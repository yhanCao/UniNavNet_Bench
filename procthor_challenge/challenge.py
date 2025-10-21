import os
import sys
import json
import yaml
import time
import gzip
import random
import logging
import prior
import queue
import threading
from llm.score_eqa import LLMQwQ
from ai2thor.controller import Controller
from typing import Dict, Any
from procthor_challenge.startx import startx
from ai2thor.util.metrics import compute_single_spl,get_shortest_path_to_object,get_shortest_path_to_point
from procthor_challenge.utils import *
logger = logging.getLogger(__name__)
ch = logging.StreamHandler(sys.stdout)
ch.flush = sys.stdout.flush
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

ALLOWED_ACTIONS =  ["Stop","MoveAhead","RotateLeft", "RotateRight", "LookUp", "LookDown" ]
class ProcthorChallenge:
    def __init__(self, cfg_file, agent_class, agent_kwargs,data_pth, render_depth=True):
        self.agent_class = agent_class
        self.agent_kwargs = agent_kwargs
        self.target_objects = []
        self.config = self.load_config(cfg_file, render_depth)
        self.setup_env()
        self.llm = LLMQwQ()
        self.current_scene = None
        self.data_pth = data_pth
        self.result_pth = None
        self.reachable_positions_per_scene = {}
        self.dataset = prior.load_dataset("procthor-10k")
        self.controller_kwargs = {
            "width": self.config["width"],
            "height": self.config["height"],
            **self.config["initialize"]
        }
    @staticmethod
    def load_config(cfg_file, render_depth):
        logger.info("Loading configuration from: %s" % cfg_file)
        with open(cfg_file, "r") as f:
            config = yaml.safe_load(f.read())
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
        dir = os.path.join(pth,split,scene,"episodes",level)
        if task_type == "eqa":
            with gzip.GzipFile(os.path.join(dir,"eqa.json.gz"), "r") as f:
                episode = json.loads(f.read().decode("utf-8"))
            episodes = episode[start_episode_id:end_episode_id]
        elif task_type == "nav":
            with gzip.GzipFile(os.path.join(dir,"nav.json.gz"), "r") as f:
                episode = json.loads(f.read().decode("utf-8"))
            episodes = episode[start_episode_id:end_episode_id]
        return episodes
    
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
            agent = agent_class(**agent_kwargs)
            dynamic_strategy = {}
            self.result_pth = os.path.join(origin_path,episode["scene_type"],str(episode["scene_id"]),episode['level'])
            if not os.path.exists(self.result_pth):
                os.makedirs(self.result_pth)
            if episode['level'] != 'static' :
                current_pth = os.path.join(self.data_pth,episode["scene_type"],str(episode["scene_id"]),"dynamic_strategy")
                with open(os.path.join(current_pth,f"strategy_final_{episode['level']}.json"), "r", encoding="utf-8") as f:
                    dynamic_strategy = json.load(f)
                dynamic_strategy = {int(key): value for key, value in dynamic_strategy.items()}
            top_down_path = os.path.join(self.result_pth,"top_down_view/")
            if not os.path.exists(top_down_path):
                os.makedirs(top_down_path)
            total_steps = 0
            episode_metrics = {}
            episode_results = {}
            for index,ins in enumerate(episode["instruction"]) :
                logger.info("Task Start instruction:{} scene:{} initial_position:{} rotation:{}".format(episode["instruction"][index],episode["scene_type"]+"_"+str(episode["scene_id"]),episode["initial_position"],episode["initial_orientation"]))
                image_nav = False
                if "image_name" in episode and episode["image_name"][index] is not None:
                        image = os.path.join(self.data_pth,"attributes",episode["scene_type"],str(episode["scene_id"]),"images",episode["image_name"][index])
                        image_nav = True
                if index == 0 :
                    teleport_action = {
                        "action": "TeleportFull",
                        **episode["initial_position"],
                        "rotation": {"x": 0, "y": episode["initial_orientation"], "z": 0},
                        "horizon": episode["initial_horizon"],
                        "standing": True
                    }
                    episode_metrics[index] = {
                    "trajectory" : [{
                        **episode["initial_position"],
                        "rotation" : float(episode["initial_orientation"]),
                        "horizon" : episode["initial_horizon"]
                    }],
                    "actions_taken" : []
                    }
                    controller.step(action=teleport_action)
                    draw_top_down_frame(controller,top_down_path,0,self.target_objects)
                else :
                    episode_metrics[index] = {
                    "trajectory" : [{
                       **event.metadata["agent"]["position"],
                        "rotation": event.metadata["agent"]["rotation"]["y"],
                        "horizon": event.metadata["agent"]["cameraHorizon"]
                    }],
                    "actions_taken" : []
                    }
                ### dynamic scene update
                batch_steps = 0
                event = controller.step(dict(action="GetReachablePositions"))
                reachable_voxel_centers = event.metadata['actionReturn']
                ####plot reachable_voxel_centers
                ####
                ### 需要发生在物体更新后
                ### given the gt target object and give the shortestespath from the current position
                self.target_objects,answer = get_target_answer(event.metadata["objects"], episode, index)
                paths = {}
                path = []
                for obj in self.target_objects:
                    try:
                        p = get_shortest_path_to_object(controller = controller,object_id=obj['name'], initial_position= controller.last_event.metadata['agent']['position'],initial_rotation=controller.last_event.metadata['agent']['rotation'])
                        print(p)
                    except ValueError as e:
                        ### find the nearest reachable_voxel_centers
                        centers = np.array([[p['x'],p['y'],p['z']] for p in reachable_voxel_centers])
                        obj_p = np.array([obj['position']['x'],obj['position']['y'],obj['position']['z']])
                        idd = np.argmin(np.linalg.norm(centers-obj_p,axis= 1))
                        p = get_shortest_path_to_point(controller = controller, initial_position= controller.last_event.metadata['agent']['position'],target_position=reachable_voxel_centers[idd],allowed_error=0.001)
                    
                    path.append(p)
                    # event = controller.step(action="GetShortestPath",objectId=obj['name'])
                    # if event.metadata['lastActionSuccess']:
                    #     path.append(event.metadata['actionReturn'])
                    # else:
                    #     print("No possible path")
                    #     print(event.metadata['errorMessage'])
                paths[index] = path
                episode['path'] = paths
                ##################
                stopped = False
                last_event = controller.last_event
                while batch_steps < max_steps and stopped is False:
                    event = last_event
                    ### modify 1 "object_goal type"
                    action,predict = agent.act({
                        "instruction" : ins,
                        "image_task": image if image_nav else None,
                        "event":event,
                        "step":total_steps
                    })
                    
                    if action not in ALLOWED_ACTIONS:
                        raise ValueError("Invalid action: {action}".format(action=action))
                    logger.info("Agent action: {action}".format(action=action))
                    stopped = action == "Stop"
                    if stopped :
                        break
                    start = time.time()
                    last_event = controller.step(action=action)
                    end = time.time()
                    print("total_step:",total_steps,"action success:",last_event.metadata["lastActionSuccess"],"total time:",end-start)
                    episode_metrics[index]["trajectory"].append({
                        **event.metadata["agent"]["position"],
                        "rotation": event.metadata["agent"]["rotation"]["y"],
                        "horizon": event.metadata["agent"]["cameraHorizon"]
                    })
                    episode_metrics[index]["actions_taken"].append({
                        "action": action,
                        "success": event.metadata["lastActionSuccess"]
                    })
                    
                    ### dynamic scene update
                    if total_steps > 0 and len(dynamic_strategy)> 0 and total_steps in [int(act[0]['time']) for act in dynamic_strategy.values()]: 
                        d = dynamic_strategy[list(dynamic_strategy.keys())[[int(act[0]['time']) for act in dynamic_strategy.values()].index(total_steps)]]
                        for value in d :
                            if value['action'] == 'PlaceObjectAtPoint':
                                event = controller.step(action=value["action"], objectId=value["objectId"], position=value["position"])
                            else :
                                event = controller.step(action="InitialRandomSpawn",
                                    randomSeed=random.randint(0, 1_000_000_000),
                                    objectIds=[value['objectId']],
                                    receptacleObjectIds=[value['rectangleobj']],
                                    numPlacementAttempts=10,
                                    forceVisible=True,
                                    allowFloor=True,
                                    renderImage=False,
                                    allowMoveable=True)
                            changed = False
                            for obj in self.target_objects:
                                if value['objectId'] == obj['name']:
                                    changed = True
                                    break
                            if changed:
                                self.target_objects, answer = get_target_answer(event.metadata["objects"],episode,index,total_steps)
                                path = []
                                for obj in self.target_objects:
                                    path.append(controller.step(action="GetShortestPath",objectId=obj['name']).metadata["actionReturn"])
                                episode['path'][total_steps] = path
                        event = controller.step(dict(action="GetReachablePositions"))
                        reachable_voxel_centers = event.metadata['actionReturn']
                        ###update agent

                    total_steps += 1
                    batch_steps += 1
                    ### get top_down view image (each total_steps)
                    draw_top_down_frame(controller,top_down_path,total_steps,self.target_objects)
                stopped = True
                target_id = 0
                if not test:
                    self.target_objects,answer = get_target_answer(event.metadata["objects"], episode, index)
                    if answer is None:
                        for idd,target_obj in enumerate(self.target_objects):
                            target_visible = target_obj["visible"]
                            if target_visible:
                                target_id = idd
                                break
                        episode_metrics[index]["success"] = 1 if stopped and target_visible else 0
                    else:   
                        print("target answer:",answer)
                        score = float(self.llm.scoring_eqa(ins,answer,predict))
                        episode_metrics[index]["success"] = (score-1) / 5
                        ### query LLM to give the confidence score
                    ########
                if not test:
                    episode_results[index] = {
                        "object_goal": str(index)+"_"+episode["instruction"][index],
                        "success": episode_metrics[index]["success"],
                        "gt_answer":answer,
                        "predict":predict,
                        "task_type":episode['task_type'][index],
                        "start_episodes":total_steps-batch_steps,
                        "end_episodes": total_steps,
                        "actions":episode_metrics[index]["actions_taken"],
                        "spl": compute_single_spl(episode_metrics[index]['trajectory'],episode['path'][index][target_id],episode_metrics[index]["success"])

                    }
                    print("found goal:",total_steps,"object type:",episode["instruction"][index],episode_results[index])
                else:
                    episode_results[index] = None
                if not os.path.exists(os.path.join(self.result_pth,'episode_results')):
                    os.makedirs(os.path.join(self.result_pth,'episode_results'))
                with open(os.path.join(self.result_pth,f'episode_results.json'), 'w') as f:
                    json.dump(episode_results, f) 
                if not os.path.exists(os.path.join(self.result_pth,'episode_metrics')):
                    os.makedirs(os.path.join(self.result_pth,'episode_metrics'))
                with open(os.path.join(self.result_pth,f'episode_metrics.json'), 'w') as f:
                    json.dump(episode_metrics, f) 
            out_queue.put((episode["instruction"], episode_metrics, episode_results))
            del agent
            controller.stop()
        print(f"Worker {worker_ind} Finished.")
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

        metrics = {"episodes" : {}}
        episode_results = []
        
        while len(episode_results) < expected_count:
            # try:
            ep_id, episode_metrics, episode_result = receive_queue.get(timeout=1000)
            metrics["episodes"][episode_id] = episode_metrics
            
            episode_results.append(episode_result)

        metrics["ep_len"] = sum([r[list(r.keys())[-1]]['end_episodes'] for r in episode_results]) / len(metrics["episodes"])
        if not test:
            metrics["success"] = sum([v["success"] for r in episode_results for v in r.values() ]) / len(episode_results)
            # metrics["spl"] = ai2thor.util.metrics.compute_spl(episode_results)
            metrics['spl'] = sum([v["spl"] for r in episode_results for v in r.values() ]) / len(episode_results)

        if not test:
            logger.info("Total Episodes: {episode_count} Success:{success} SPL:{spl} Episode Length:{ep_len}".format(episode_count=len(episodes), success=metrics["success"], spl=metrics["spl"], ep_len=metrics["ep_len"]))
        else:
            logger.info("Total Episodes: {episode_count} Episode Length:{ep_len}".format(episode_count=len(episodes), ep_len=metrics["ep_len"]))
        return metrics


    def _change_scene(self, scene):
        if self.current_scene != scene:
            self.current_scene = scene
            self.controller.reset(scene)
            logger.info("Changed to scene: '{scene}'".format(scene=scene))

    def move_to_point(self, datapoint):
        self._change_scene(datapoint["scene"])
        logger.info("Moving to position: {p}, y-rotation: {rot}, horizon: {hor}".format(
            p=datapoint["initial_position"],
            rot=datapoint["initial_orientation"],
            hor=datapoint["initial_horizon"]
        ))
        return self.controller.step(
            action="TeleportFull",
            x=datapoint["initial_position"]["x"],
            y=datapoint["initial_position"]["y"],
            z=datapoint["initial_position"]["z"],
            rotation={"x" : 0, "y" : datapoint["initial_orientation"], "z" : 0},
            horizon=datapoint["initial_horizon"],
            standing=True
        )

    def move_to_random_dataset_point(self, dataset, scene, object_type):
        if scene in dataset:
            if object_type in dataset[scene]:
                datapoint = random.choice(dataset[scene][object_type])
                return self.move_to_point(datapoint)
            else:
                logger.warning(
                    "No object of type: '{object_type}' for scene: '{scene}', in dataset".format(
                        object_type=object_type,
                        scene=scene
                    )
                )
                return None
        else:
            logger.warning("No scene: '{scene}' in dataset".format(scene=scene))
            return None

    def move_to_random_point(self, scene, y_rotation=0, horizon=0):
        if "test" in scene:
            raise RuntimeError(
                "Moving to random points is not posible in test scenes"
            )
        reachable_positions = self._get_reachable_positions_in_scene(scene)
        p = random.choice(reachable_positions)
        return self.move_to_point({
            "initial_position": p,
            "initial_orientation": y_rotation,
            "initial_horizon": horizon,
            "scene" : scene
        })

    def _get_reachable_positions_in_scene(self, scene):
        self._change_scene(scene)
        if scene not in self.reachable_positions_per_scene:
            event_reachable = self.controller.step({
                "action" : "GetReachablePositions",
                "gridSize" : self.config["initialize"]["gridSize"]
            })
            self.reachable_positions_per_scene[scene] = event_reachable.metadata["actionReturn"]
        return self.reachable_positions_per_scene[scene]