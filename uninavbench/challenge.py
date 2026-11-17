import os
import json
import yaml
import time
import gzip
import logging
import prior
import queue
import threading
import zlib
import re
from importlib.metadata import version
from ai2thor.controller import Controller
from typing import Dict, Any
from .startx import startx
from .judge import create_judge
from .data import default_data_dir
from ai2thor.util.metrics import compute_single_spl,get_shortest_path_to_object,get_shortest_path_to_point
from .utils import *

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "procthor.yaml")
PROCTHOR_REVISION = "439193522244720b86d8c81cde2e51e3a4d150cf"
AI2THOR_COMMIT_ID = "f0825767cd50d69f666c7f282e54abfe58f1e917"
logger = logging.getLogger(__name__)

ALLOWED_ACTIONS =  ["Stop","MoveAhead","RotateLeft", "RotateRight", "LookUp", "LookDown" ]
# object-state events of the dynamic strategies
MEMORY_CAPTIONING = 11
STATE_ACTIONS = {"ToggleObjectOn", "ToggleObjectOff", "OpenObject", "CloseObject", "SliceObject", "DirtyObject", "CleanObject"}


# A scheduled relocation can land on the agent or wall it into a pocket; it is then moved to free floor.
TRAPPED_FRACTION = 0.5       # trapped: the agent reaches less than this part of the episode's initial floor
RELOCATION_CANDIDATES = 50   # nearest floor positions tried


def relocates(events, start, end):
    """Whether events[start:end] move an object. State changes (a door opened for a few minutes) do not."""
    return any(v["action"] not in STATE_ACTIONS for _, _, group in events[start:end] for v in group)


def free_agent(controller, floor):
    """Moves a trapped agent to the nearest free position of ``floor`` (the reachable positions of the
    episode's initial scene). The agent is trapped when it collides with an object or reaches less than
    TRAPPED_FRACTION of ``floor``. Returns {"from": position, "to": position}, or None if it is free."""
    agent = controller.last_event.metadata["agent"]
    # the reported horizon drifts (60.00002), but Teleport only accepts [-30, 60]
    pose = dict(rotation=agent["rotation"], horizon=round(agent["cameraHorizon"]), standing=True)

    why = {}

    def free(position):
        event = controller.step(action="Teleport", position=position, **pose)
        if not event.metadata["lastActionSuccess"]:
            why["teleport"] = event.metadata.get("errorMessage", "")[:200]
            return False
        reach = controller.step(action="GetReachablePositions").metadata["actionReturn"] or []
        why["reachable"] = f"{len(reach)} of {len(floor)}"
        return len(reach) >= TRAPPED_FRACTION * len(floor)

    here = agent["position"]
    if not floor or free(here):
        return None
    nearest = sorted(floor, key=lambda p: (p["x"] - here["x"]) ** 2 + (p["z"] - here["z"]) ** 2)
    for position in nearest[:RELOCATION_CANDIDATES]:
        if free(position):
            moved_to = controller.last_event.metadata["agent"]["position"]
            logger.info("Agent trapped by a scheduled relocation at %s: moved to %s", here, moved_to)
            return {"from": here, "to": moved_to}
    controller.step(action="Teleport", position=here, forceAction=True, **pose)
    logger.warning("Agent trapped at %s and no free position nearby (%s)", here, why)
    return None


class DynamicReplayError(RuntimeError):
    """A scheduled scene change could not be replayed; evaluation must stop."""


def replay_step(controller, scheduled, group, **action):
    context = (f"Dynamic replay failed at time {scheduled['time']} (group {group}): "
               f"{action['action']} on {action.get('objectId', scheduled['objectId'])}")
    try:
        event = controller.step(**action)
    except Exception as exc:
        raise DynamicReplayError(f"{context}: {exc}") from exc
    if event.metadata.get("lastActionSuccess") is not True:
        raise DynamicReplayError(f"{context}: {event.metadata.get('errorMessage') or 'lastActionSuccess=False'}")
    return event
# Custom function to replace the buggy get_shortest_path_to_point from ai2thor
def get_shortest_path_to_point_custom(controller, initial_position, target_position, allowed_error=0.0001):
    """
    Computes the shortest path to a point from an initial position.
    This is a fixed version that works with AI2-THOR 5.0+

    :param controller: agent controller
    :param initial_position: dict(x=float, y=float, z=float) with initial position
    :param target_position: dict(x=float, y=float, z=float) representing target position
    :param allowed_error: tolerance for path finding
    :return: path corners as list of points
    """
    try:
        kwargs = dict(
            action="GetShortestPathToPoint",
            position=initial_position,
            target=target_position,
            allowedError=allowed_error,
        )
        event = controller.step(kwargs)
        if event.metadata["lastActionSuccess"]:
            return event.metadata["actionReturn"].get("corners", [])
        else:
            logger.error(f"Failed to find path to point {target_position}: {event.metadata.get('errorMessage', 'Unknown error')}")
            return []
    except Exception as e:
        logger.error(f"Error in get_shortest_path_to_point_custom: {str(e)}")
        return []
class ProcthorChallenge:
    def __init__(self, cfg_file=None, agent_class=None, agent_kwargs=None, data_pth=None, render_depth=True, judge=None):
        self.agent_class = agent_class
        self.agent_kwargs = agent_kwargs or {}
        self.target_objects = []
        self.config = self.load_config(cfg_file or DEFAULT_CONFIG, render_depth)
        self.setup_env()
        self.llm = judge if judge is not None else create_judge(self.config)
        self.data_pth = data_pth or default_data_dir()
        if not os.path.isdir(os.path.join(self.data_pth, "attributes")):
            raise FileNotFoundError(f"UniNavBench data not found at '{self.data_pth}'. Download it with "
                                    "`uninavbench download --split test`, or pass data_pth / set UNINAVBENCH_DATA")
        self.result_pth = None
        try:
            self.dataset = prior.load_dataset("procthor-10k", revision=self.config["procthor_revision"])
        except Exception as e:
            # GitHub unreachable: fall back to the locally cached revision
            logger.warning("Online load failed (%r); using the same pinned offline revision %s.",
                           e, self.config["procthor_revision"])
            self.dataset = prior.load_dataset("procthor-10k", revision=self.config["procthor_revision"], offline=True)
        self.controller_kwargs = {
            "width": self.config["width"],
            "height": self.config["height"],
            **self.config["initialize"],
            "commit_id": self.config["ai2thor_commit_id"],
        }
        self.environment = {
            "procthor_revision": self.config["procthor_revision"],
            "ai2thor_commit_id": self.config["ai2thor_commit_id"],
            "ai2thor_package_version": version("ai2thor"),
            "prior_package_version": version("prior"),
        }
    @staticmethod
    def load_config(cfg_file, render_depth):
        logger.info("Loading configuration from: %s" % cfg_file)
        with open(cfg_file, "r") as f:
            config = yaml.safe_load(f.read())
        if "thor_build_id" in config:
            raise ValueError("thor_build_id is ambiguous and was previously ignored. Replace it with "
                             "procthor_revision (house dataset SHA) and ai2thor_commit_id (simulator SHA); "
                             "see uninavbench/configs/procthor.yaml.")
        for key, default in (("procthor_revision", PROCTHOR_REVISION), ("ai2thor_commit_id", AI2THOR_COMMIT_ID)):
            config.setdefault(key, default)
            if not isinstance(config[key], str) or not re.fullmatch(r"[0-9a-f]{40}", config[key]):
                raise ValueError(f"{key} must be a full 40-character commit SHA")
        for key in ("commit_id", "branch", "local_executable_path", "local_build"):
            if key in config["initialize"]:
                raise ValueError(f"initialize.{key} bypasses the simulator pin; use ai2thor_commit_id instead")
        if render_depth:
            config["initialize"]["renderDepthImage"] = True
        return config

    @staticmethod
    def setup_env():
        if "DISPLAY" not in os.environ:
            xthread = threading.Thread(target=startx)
            xthread.daemon = True
            xthread.start()
            import time
            # XXX change this to use xdpyinfo
            time.sleep(4)
    
    @staticmethod
    def load_split(pth, split, level, scene,start_episode_id,end_episode_id,task_type):
        if task_type not in ("nav", "eqa"):
            raise ValueError(f"Unknown task type: {task_type}")
        dir = os.path.join(pth,split,scene,"episodes",level)
        gz_path = os.path.join(dir, f"{task_type}.json.gz")
        json_path = os.path.join(dir, f"{task_type}.json")
        if os.path.exists(gz_path):
            with gzip.open(gz_path, "rt", encoding="utf-8") as f:
                episode = json.load(f)
        elif os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                episode = json.load(f)
        else:
            raise FileNotFoundError(f"Neither {gz_path} nor {json_path} exists")
        return episode[start_episode_id:end_episode_id]

    @staticmethod
    def load_dynamic_events(strategy_path):
        """Returns the dynamic strategy as a list of (time, key, actions) sorted by time."""
        with open(strategy_path, "r", encoding="utf-8") as f:
            dynamic_strategy = json.load(f)
        events = [(int(group[0]['time']), int(key), group) for key, group in dynamic_strategy.items() if group]
        return sorted(events, key=lambda e: (e[0], e[1]))

    @staticmethod
    def apply_dynamic_events(controller, events, next_event, scene_time):
        """Applies every pending event whose time <= scene_time. Returns (next_event, ids of changed objects)."""
        moved = set()
        while next_event < len(events) and events[next_event][0] <= scene_time:
            for value in events[next_event][2]:
                changed = {value['objectId']}
                group = events[next_event][1]
                if value['action'] in STATE_ACTIONS:
                    replay_step(controller, value, group, action=value["action"], objectId=value["objectId"], forceAction=True)
                elif value.get('resolved'):
                    # Recorded poses are snapshots, not new physical drop requests.
                    # InitialRandomSpawn(placeStationary=True) in the generator makes
                    # eligible bodies kinematic, including objects outside objectIds.
                    # Restore stable poses so gravity cannot invalidate those snapshots.
                    for pose in value['resolved']:
                        replay_step(controller, value, group, action="TeleportObject", objectId=pose["objectId"], position=pose["position"],
                                        rotation=pose["rotation"], forceAction=True, forceKinematic=True)
                        changed.add(pose["objectId"])
                elif value['action'] == 'PlaceObjectAtPoint':
                    replay_step(controller, value, group, action=value["action"], objectId=value["objectId"], position=value["target_position"],rotation=value["target_rotation"])
                elif value['action'] == 'InitialRandomSpawn':
                    # seed from the event itself so every run produces the same placement
                    seed = zlib.crc32(f"{value['objectId']}|{value['time']}".encode("utf-8"))
                    replay_step(controller, value, group, action="InitialRandomSpawn",
                        randomSeed=seed,
                        objectIds=[value['objectId']],
                        receptacleObjectIds=[value['rectangleobj']],
                        numPlacementAttempts=10,
                        forceVisible=True,
                        allowFloor=True,
                        renderImage=False,
                        allowMoveable=True)
                else:
                    raise DynamicReplayError(f"Unsupported scheduled action {value['action']} at time {value['time']}")
                moved.update(changed)
            next_event += 1
        return next_event, moved

    @staticmethod
    def resolve_image_path(data_pth, episode, index):
        image_dir = os.path.join(data_pth,"attributes",episode["scene_type"],str(episode["scene_id"]),"images")
        image = os.path.join(image_dir, episode["image_name"][index])
        if not os.path.exists(image):
            # image files on disk use "_" where the episode ids use "|"
            image = os.path.join(image_dir, episode["image_name"][index].replace("|", "_"))
        return image

    def inference_worker(
        self,
        worker_ind: int,
        in_queue: queue.Queue,
        out_queue: queue.Queue,
        agent_class: Any,
        agent_kwargs: Dict[str, Any],
        controller_kwargs: Dict[str, Any],
        max_steps: int,
        origin_path:str,
        test: bool,
        episode_count: int
    ):
        require_grounding = self.config.get("eqa_require_grounding", True)
        save_top_down = self.config.get("save_top_down_view", False)
        while True:
            try:
                episode = in_queue.get(timeout=1)
                episode_count+=1
            except queue.Empty:
                break
            if not os.path.exists(origin_path):
                os.makedirs(origin_path)
            controller_kwargs['scene'] = self.dataset[episode["scene_type"]][episode["scene_id"]]
            controller = Controller(**controller_kwargs)
            try:
                agent = agent_class(**agent_kwargs)
                # Replaying recorded poses can knock a laptop or a vase hard enough to break it, which the
                # schedule never does (a broken laptop cannot be toggled). Nothing in the benchmark breaks objects.
                controller.step(action="MakeAllObjectsUnbreakable")
                task_kind = "eqa" if "answer" in episode else "nav"
                dynamic_events = []
                next_event = 0
                # scene clock (minutes of the simulated day); advances by one per agent step
                scene_time = 0
                self.result_pth = os.path.join(origin_path,task_kind,episode["scene_type"],str(episode["scene_id"]),episode['level'])
                if not os.path.exists(self.result_pth):
                    os.makedirs(self.result_pth)
                bind_environment = getattr(agent, "bind_environment", None)
                if bind_environment is not None:
                    bind_environment(controller, os.path.join(self.result_pth, str(episode_count)))
                if episode['level'] != 'static' :
                    current_pth = os.path.join(self.data_pth,episode["scene_type"],str(episode["scene_id"]),"dynamic_strategy")
                    dynamic_events = self.load_dynamic_events(os.path.join(current_pth,f"strategy_final_{episode['level']}.json"))
                top_down_path = os.path.join(self.result_pth,"top_down_view/")
                if save_top_down and not os.path.exists(top_down_path):
                    os.makedirs(top_down_path)
                total_steps = 0
                episode_metrics = {}
                episode_results = {}
                # objectIds the agent has seen so far in this episode (visual grounding for EQA)
                seen_ids = set()
                for index,ins in enumerate(episode["instruction"]) :
                    logger.info("Task Start instruction:{} scene:{} initial_position:{} rotation:{}".format(episode["instruction"][index],episode["scene_type"]+"_"+str(episode["scene_id"]),episode["initial_position"],episode["initial_orientation"]))
                    image_nav = False
                    if "image_name" in episode and episode["image_name"][index] is not None:
                            image = self.resolve_image_path(self.data_pth, episode, index)
                            image_nav = True
                    if index == 0 :
                        teleport_action = {
                            "action": "TeleportFull",
                            **episode["initial_position"],
                            "rotation": {"x": 0, "y": episode["initial_orientation"], "z": 0},
                            "horizon": episode["initial_horizon"],
                            "standing": True
                        }
                        controller.step(action=teleport_action)
                        # floor of the initial scene: where a trapped agent may be moved to
                        floor = controller.step(action="GetReachablePositions").metadata["actionReturn"] or []
                        if save_top_down:
                            draw_top_down_frame(controller,top_down_path,0,self.target_objects)
                    ### dynamic scene update: EQA questions are asked at a given time of day, fast-forward the scene to it
                    if "time" in episode:
                        question_time = episode["time"][index] if isinstance(episode["time"], list) else episode["time"]
                        scene_time = max(scene_time, int(question_time))
                    first = next_event
                    next_event, _ = self.apply_dynamic_events(controller, dynamic_events, next_event, scene_time)
                    relocation = free_agent(controller, floor) if relocates(dynamic_events, first, next_event) else None
                    event = controller.step(dict(action="GetReachablePositions"))
                    reachable_voxel_centers = event.metadata['actionReturn']
                    agent_meta = event.metadata["agent"]
                    episode_metrics[index] = {
                        "trajectory" : [{
                            **agent_meta["position"],
                            "rotation": agent_meta["rotation"]["y"],
                            "horizon": agent_meta["cameraHorizon"]
                        }],
                        "actions_taken" : [],
                        "agent_relocations": [dict(step=0, **relocation)] if relocation else []
                    }
                    batch_steps = 0
                    ### given the gt target object and give the shortestespath from the current position
                    self.target_objects,answer = get_target_answer(event.metadata["objects"], episode, index, scene_time)
                    path = []
                    for obj in self.target_objects:
                        try:
                            p = get_shortest_path_to_object(controller = controller,object_id=obj['objectId'], initial_position= controller.last_event.metadata['agent']['position'],initial_rotation=controller.last_event.metadata['agent']['rotation'])
                        except ValueError as e:
                            ### find the nearest reachable_voxel_centers
                            logger.warning(f"Failed to get path to object {obj['name']}: {str(e)}")
                            ### find the nearest reachable_voxel_centers
                            try:
                                centers = np.array([[pos['x'],pos['y'],pos['z']] for pos in reachable_voxel_centers])
                                obj_p = np.array([obj['position']['x'],obj['position']['y'],obj['position']['z']])
                                idd = np.argmin(np.linalg.norm(centers-obj_p,axis= 1))
                                # Use custom function instead of the buggy ai2thor version
                                p = get_shortest_path_to_point_custom(
                                    controller=controller,
                                    initial_position=controller.last_event.metadata['agent']['position'],
                                    target_position=reachable_voxel_centers[idd],
                                    allowed_error=0.001
                                )
                            except Exception as fallback_error:
                                logger.error(f"Failed to get path to nearest point for {obj['name']}: {str(fallback_error)}")
                                # If path finding fails completely, use empty path as fallback
                                p = []
                        path.append(p)
                    ##################
                    stopped = False
                    predict = None
                    start_task = getattr(agent, "start_task", None)
                    if start_task is not None:
                        start_task(ins, reachable_voxel_centers)
                    last_event = controller.last_event
                    seen_ids.update(obj['objectId'] for obj in last_event.metadata["objects"] if obj['visible'])
                    while batch_steps < max_steps:
                        ### modify 1 "object_goal type"
                        action,predict = agent.act({
                            "instruction" : ins,
                            "image_task": image if image_nav else None,
                            "event":last_event,
                            "step":total_steps
                        })
                    
                        if action not in ALLOWED_ACTIONS:
                            raise ValueError("Invalid action: {action}".format(action=action))
                        logger.info("Agent action: {action}".format(action=action))
                        if action == "Stop" :
                            stopped = True
                            break
                        start = time.time()
                        last_event = controller.step(action=action)
                        end = time.time()
                        logger.debug("total_step: %d action success: %s time: %.3f", total_steps, last_event.metadata["lastActionSuccess"], end-start)
                        episode_metrics[index]["trajectory"].append({
                            **last_event.metadata["agent"]["position"],
                            "rotation": last_event.metadata["agent"]["rotation"]["y"],
                            "horizon": last_event.metadata["agent"]["cameraHorizon"]
                        })
                        episode_metrics[index]["actions_taken"].append({
                            "action": action,
                            "success": last_event.metadata["lastActionSuccess"]
                        })
                        total_steps += 1
                        batch_steps += 1
                        scene_time += 1

                        ### dynamic scene update
                        first = next_event
                        next_event, moved = self.apply_dynamic_events(controller, dynamic_events, next_event, scene_time)
                        if moved:
                            relocation = free_agent(controller, floor) if relocates(dynamic_events, first, next_event) else None
                            if relocation:
                                episode_metrics[index]["agent_relocations"].append(dict(step=batch_steps, **relocation))
                                agent_meta = controller.last_event.metadata["agent"]
                                episode_metrics[index]["trajectory"].append({
                                    **agent_meta["position"],
                                    "rotation": agent_meta["rotation"]["y"],
                                    "horizon": agent_meta["cameraHorizon"]
                                })
                            # re-render so the agent observes the updated scene
                            last_event = controller.step(action="Done")
                            update_environment = getattr(agent, "update_environment", None)
                            if update_environment is not None:
                                last_event = controller.step(action="GetReachablePositions")
                                update_environment(last_event.metadata["actionReturn"])
                            if moved & {obj['objectId'] for obj in self.target_objects}:
                                self.target_objects, answer = get_target_answer(last_event.metadata["objects"],episode,index,scene_time)
                        seen_ids.update(obj['objectId'] for obj in last_event.metadata["objects"] if obj['visible'])
                        ### get top_down view image (each total_steps)
                        if save_top_down:
                            draw_top_down_frame(controller,top_down_path,total_steps,self.target_objects)
                    target_id = 0
                    if not test:
                        self.target_objects,answer = get_target_answer(last_event.metadata["objects"], episode, index, scene_time)
                        if answer is None:
                            visible_ids = [idd for idd,target_obj in enumerate(self.target_objects) if target_obj["visible"]]
                            if visible_ids:
                                target_id = visible_ids[0]
                            episode_metrics[index]["success"] = 1 if stopped and visible_ids else 0
                        else:
                            logger.debug("target answer: %s", answer)
                            # Memory Captioning asks about the day's history, not about what is visible now
                            grounded = (not require_grounding or episode['task_type'][index] == MEMORY_CAPTIONING
                                        or any(i == episode['id'][index] or i.startswith(episode['id'][index] + "|") for i in seen_ids))
                            if predict is None or not grounded:
                                episode_metrics[index]["success"] = 0.0
                            else:
                                score = self.llm.score(ins,answer,predict)
                                episode_metrics[index]["success"] = (score-1) / 4
                        ########
                    if not test:
                        shortest_path = path[target_id] if target_id < len(path) else []
                        episode_results[index] = {
                            "object_goal": str(index)+"_"+episode["instruction"][index],
                            "success": episode_metrics[index]["success"],
                            "stopped": stopped,
                            "gt_answer":answer,
                            "predict":predict,
                            "task_type":episode['task_type'][index],
                            "scene_time": scene_time,
                            "start_episodes":total_steps-batch_steps,
                            "end_episodes": total_steps,
                            "actions":episode_metrics[index]["actions_taken"],
                            # None when no shortest path could be computed; excluded from the SPL average
                            "spl": compute_single_spl(episode_metrics[index]['trajectory'],shortest_path,episode_metrics[index]["success"]) if shortest_path else None

                        }
                        logger.info("sub-task %d done: success=%s spl=%s", index, episode_results[index]["success"], episode_results[index]["spl"])
                    else:
                        episode_results[index] = None
                    with open(os.path.join(self.result_pth,f'episode_results_{episode_count}.json'), 'w') as f:
                        json.dump(episode_results, f) 
                    with open(os.path.join(self.result_pth,f'episode_metrics_{episode_count}.json'), 'w') as f:
                        json.dump(episode_metrics, f) 
                out_queue.put((episode_count, episode_metrics, episode_results))
                del agent
            finally:
                controller.stop()
    def inference(self, episodes, nprocesses=1, test=False,episode_id = 0):
        send_queue = queue.Queue()
        receive_queue = queue.Queue()
        expected_count = len(episodes)
        for e in episodes:
            send_queue.put(e)
        self.inference_worker(
            worker_ind=0,  
            in_queue=send_queue,  
            out_queue=receive_queue,  
            agent_class=self.agent_class,
            agent_kwargs=self.agent_kwargs,
            controller_kwargs=self.controller_kwargs,
            max_steps=self.config["max_steps"],
            origin_path=self.config["result_path"],
            test=test,
            episode_count = episode_id
        )

        metrics = {"episodes": {}, "environment": dict(self.environment)}
        episode_results = []
        
        while len(episode_results) < expected_count:
            ep_idx, episode_metrics, episode_result = receive_queue.get(timeout=1000)
            metrics["episodes"][ep_idx] = episode_metrics
            episode_results.append(episode_result)

        metrics["ep_len"] = sum(len(m["actions_taken"]) for em in metrics["episodes"].values() for m in em.values()) / max(len(metrics["episodes"]), 1)
        if not test:
            # averaged over sub-tasks, not episodes
            subtasks = [v for r in episode_results for v in r.values()]
            metrics["num_subtasks"] = len(subtasks)
            metrics["success"] = sum(v["success"] for v in subtasks) / max(len(subtasks), 1)
            spls = [v["spl"] for v in subtasks if v["spl"] is not None]
            metrics["spl"] = sum(spls) / max(len(spls), 1)
            metrics["num_subtasks_without_path"] = len(subtasks) - len(spls)
            per_type = {}
            for v in subtasks:
                per_type.setdefault(str(v["task_type"]), []).append(v)
            metrics["per_task_type"] = {
                tt: {
                    "num_subtasks": len(vs),
                    "success": sum(v["success"] for v in vs) / len(vs),
                    "spl": sum(v["spl"] for v in vs if v["spl"] is not None) / max(sum(v["spl"] is not None for v in vs), 1),
                }
                for tt, vs in sorted(per_type.items())
            }

        if not test:
            logger.info("Total Episodes: {episode_count} Sub-tasks: {n} Success:{success} SPL:{spl} Episode Length:{ep_len}".format(episode_count=len(episodes), n=metrics["num_subtasks"], success=metrics["success"], spl=metrics["spl"], ep_len=metrics["ep_len"]))
        else:
            logger.info("Total Episodes: {episode_count} Episode Length:{ep_len}".format(episode_count=len(episodes), ep_len=metrics["ep_len"]))
        return metrics


    def evaluate(self, split="test", level="all", scene=0, task="nav", start=0, end=None, output_dir=None, test=False):
        """Evaluates the agent on episodes [start, end) of one scene, level and task family ("nav" / "eqa")."""
        episodes = self.load_split(self.data_pth, split, level, str(scene), start, end, task)
        if output_dir is not None:
            self.config["result_path"] = output_dir
        return self.inference(episodes, test=test, episode_id=start)
