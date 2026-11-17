"""End-to-end check of an installed uninavbench against the dynamic dataset.

    python tests/verify_dynamic.py DATA_ROOT [--split test --scene 0 --episodes 2 --out report.json]

1. Scene changes: after the evaluator applies dynamic events, every object-state event is reflected in the
   simulator and every relocated object is at its recorded pose.
2. Navigation: an oracle agent (privileged access to the simulator and the targets) walks to a pose from
   which a target is visible and stops -> success expected; agents that stop at once or act randomly
   should mostly fail.
3. Active-EQA: the oracle walks to the queried object and answers with the ground truth valid at the
   current scene time -> correct; answering with the initial (t=0) answer must fail exactly when the
   answer has changed by then.
The oracle hooks into the evaluator with monkeypatching; this is for testing only.
"""
import argparse
import collections
import json
import math
import os
import random
import sys

import uninavbench
import uninavbench.challenge as challenge_mod
from uninavbench import ALLOWED_ACTIONS, STATE_ACTIONS, Agent, ExactMatchJudge, ProcthorChallenge, answer_at_time

STATE_ATTR = {"ToggleObjectOn": ("isToggled", True), "ToggleObjectOff": ("isToggled", False),
              "OpenObject": ("isOpen", True), "CloseObject": ("isOpen", False), "SliceObject": ("isSliced", True),
              "DirtyObject": ("isDirty", True), "CleanObject": ("isDirty", False)}

HOOK = {"controller": None, "bench": None, "record": None}
ORACLE_LOGS = []
EVENT_CHECK = collections.Counter()
EVENT_FAIL = []


# ---------------------------------------------------------------- hooks
class CapturingController(challenge_mod.Controller):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        HOOK["controller"] = self


_orig_get_target_answer = challenge_mod.get_target_answer


def recording_get_target_answer(event_objects, episode, index, current_time=0):
    objs, answer = _orig_get_target_answer(event_objects, episode, index, current_time)
    HOOK["record"] = dict(episode=episode, index=index, time=current_time, answer=answer)
    return objs, answer


_orig_apply = ProcthorChallenge.apply_dynamic_events


def checking_apply(controller, events, next_event, scene_time):
    start = next_event
    next_event, moved = _orig_apply(controller, events, next_event, scene_time)
    if next_event > start:
        objs = {o["objectId"]: o for o in controller.step(action="Done").metadata["objects"]}
        # the last event touching an object decides its expected final state / pose
        final = {}
        for _, _, group in events[start:next_event]:
            for ev in group:
                if ev["action"] in STATE_ACTIONS:
                    final[(ev["objectId"], STATE_ATTR[ev["action"]][0])] = (ev, STATE_ATTR[ev["action"]][1])
                    # a recorded pose is the snapshot before this change (opening a book can move it)
                    final.pop((ev["objectId"], "pose"), None)
                for pose in ev.get("resolved") or []:
                    final[(pose["objectId"], "pose")] = (ev, pose)
        for (oid, what), (ev, expected) in final.items():
            o = objs.get(oid)
            if what == "pose":
                ok = o is not None and math.dist([o["position"][k] for k in "xyz"], [expected["position"][k] for k in "xyz"]) < 0.05
            else:
                ok = o is not None and o[what] == expected
            EVENT_CHECK["pose" if what == "pose" else "state"] += 1
            if not ok:
                EVENT_FAIL.append(dict(time=ev["time"], action=ev["action"], objectId=oid, what=what,
                                       expected=expected if what != "pose" else expected["position"],
                                       got=None if o is None else (o["position"] if what == "pose" else o[what])))
    return next_event, moved


challenge_mod.Controller = CapturingController
challenge_mod.get_target_answer = recording_get_target_answer
ProcthorChallenge.apply_dynamic_events = staticmethod(checking_apply)


# ---------------------------------------------------------------- oracle
HEADING = {0: (0, 1), 90: (1, 0), 180: (0, -1), 270: (-1, 0)}
MAX_REPLANS = 3


class OracleAgent(Agent):
    """Walks (BFS on the reachable grid) to a pose from which a target is visible, then stops.
    For Active-EQA it answers with the ground truth valid at the current scene time
    (or at t=0 when stale=True)."""

    def __init__(self, stale=False):
        self.stale = stale
        self.key = None
        self.plan = []
        self.steps = 0
        self.start_time = 0
        self.targets_at_plan = None
        self.replans = 0
        self.last_position = None
        self.log = ORACLE_LOGS

    def reset(self):
        pass

    def _new_subtask(self):
        rec = HOOK["record"]
        key = (id(rec["episode"]), rec["index"])
        if key != self.key:
            self.key, self.plan, self.steps, self.start_time = key, [], 0, rec["time"]
            self.targets_at_plan = None
            self.replans = 0
            self.log.append(dict(index=rec["index"], task_type=rec["episode"]["task_type"][rec["index"]], planned=False))

    def _targets(self):
        return [o["objectId"] for o in HOOK["bench"].target_objects]

    def _plan(self, event):
        c = HOOK["controller"]
        agent = event.metadata["agent"]
        here = dict(x=agent["position"]["x"], y=agent["position"]["y"], z=agent["position"]["z"],
                    rotation=dict(x=0, y=agent["rotation"]["y"], z=0), horizon=agent["cameraHorizon"], standing=True)
        reach = c.step(action="GetReachablePositions").metadata["actionReturn"] or []
        grid = {(round(p["x"] / 0.25), round(p["z"] / 0.25)) for p in reach}
        goals = {}
        for oid in self._targets():
            e = c.step(action="GetInteractablePoses", objectId=oid, rotations=[0, 90, 180, 270],
                       horizons=[-30, 0, 30, 60], standings=[True])
            for p in e.metadata["actionReturn"] or []:
                k = (round(p["x"] / 0.25), round(p["z"] / 0.25), int(round(p["rotation"])) % 360)
                goals.setdefault(k, int(round(p["horizon"])))
        c.step(action="TeleportFull", **here)  # the queries may move the agent
        self.targets_at_plan = {o["objectId"]: o["position"] for o in HOOK["bench"].target_objects}
        self.log[-1]["planned"] = True
        self.log[-1]["goal_poses"] = len(goals)
        start = (round(here["x"] / 0.25), round(here["z"] / 0.25), int(round(here["rotation"]["y"] / 90) * 90) % 360)
        prev = {start: None}
        queue = collections.deque([start])
        found = None
        while queue:
            s = queue.popleft()
            if s in goals:
                found = s
                break
            x, z, r = s
            dx, dz = HEADING[r]
            for action, nxt in (("MoveAhead", (x + dx, z + dz, r)), ("RotateRight", (x, z, (r + 90) % 360)),
                                ("RotateLeft", (x, z, (r - 90) % 360))):
                if nxt not in prev and (nxt[0], nxt[1]) in grid:
                    prev[nxt] = (s, action)
                    queue.append(nxt)
        if found is None:
            return []
        actions = []
        s = found
        while prev[s] is not None:
            s, a = prev[s][0], prev[s][1]
            actions.append(a)
        actions.reverse()
        dh = goals[found] - int(round(here["horizon"]))
        actions += ["LookDown" if dh > 0 else "LookUp"] * (abs(dh) // 30)
        return actions

    def _answer(self):
        rec = HOOK["record"]
        gt = rec["episode"]["answer"][rec["index"]]
        return answer_at_time(gt, 0 if self.stale else self.start_time + self.steps)

    def act(self, obs):
        self._new_subtask()
        event = obs["event"]
        is_eqa = "answer" in HOOK["record"]["episode"]
        visible = {o["objectId"] for o in event.metadata["objects"] if o["visible"]}
        targets = set(self._targets())
        if any(v == t or v.startswith(t + "|") for v in visible for t in targets):
            self.log[-1]["stopped_visible"] = True
            return "Stop", (self._answer() if is_eqa else None)
        moved = self.targets_at_plan is not None and any(
            o["objectId"] not in self.targets_at_plan or math.dist([o["position"][k] for k in "xz"],
                                                                   [self.targets_at_plan[o["objectId"]][k] for k in "xz"]) > 0.1
            for o in HOOK["bench"].target_objects)
        position = event.metadata["agent"]["position"]
        # the evaluator moved a trapped agent: the plan starts from the old position
        relocated = self.last_position is not None and math.dist([position[k] for k in "xz"], [self.last_position[k] for k in "xz"]) > 0.3
        self.last_position = position
        if (not self.plan and self.targets_at_plan is None) or moved or relocated or (event.metadata["lastActionSuccess"] is False and self.plan):
            self.plan = self._plan(event)
        elif not self.plan and self.replans < MAX_REPLANS:
            # the plan ended without the target in view: the scene changed on the way
            # (a door closed, an occluder moved in), so the planned pose no longer sees it
            self.replans += 1
            self.plan = self._plan(event)
        if not self.plan:
            self.log[-1]["no_plan"] = True
            return "Stop", (self._answer() if is_eqa else None)
        self.steps += 1
        return self.plan.pop(0), None


class StopAgent(Agent):
    def reset(self):
        pass

    def act(self, obs):
        return "Stop", None


class RandomAgent(Agent):
    def reset(self):
        pass

    def act(self, obs):
        return random.choice(ALLOWED_ACTIONS[1:]), None


# ---------------------------------------------------------------- driver
def norm(value):
    return json.dumps(uninavbench.judge._normalize(value))


def evaluate(data, agent_class, agent_kwargs, split, scene, level, task, episodes, max_steps, out_dir, judge="exact"):
    bench = ProcthorChallenge(None, agent_class, agent_kwargs, data, render_depth=False,
                              judge=ExactMatchJudge() if judge == "exact" else None)
    bench.config["max_steps"] = max_steps
    HOOK["bench"] = bench
    metrics = bench.evaluate(split, level, scene, task, 0, episodes, output_dir=out_dir)
    return metrics



def count_never_visible(logs):
    """Sub-tasks whose target cannot be seen from any reachable pose. Memory captioning asks
    about the day's history and the evaluator does not require seeing its object, so it is exempt."""
    return sum(1 for l in logs if l.get("goal_poses") == 0 and l["task_type"] != challenge_mod.MEMORY_CAPTIONING)


def validation_errors(report, oracle_min_success=1.0):
    """Release gates; negative-control task failures are expected."""
    errors = []
    if report.get("execution_error"):
        errors.append(report["execution_error"])
    if not report["runs"]:
        errors.append("No validation runs completed")
    for row in report["runs"]:
        name = f"{row['level']}/{row['task']}/{row['agent']}"
        if row["subtasks"] == 0:
            errors.append(f"{name}: no sub-tasks evaluated")
        if row["n_event_failures"]:
            errors.append(f"{name}: {row['n_event_failures']} replay mismatches")
        if row.get("target_never_visible", 0):
            errors.append(f"{name}: oracle target never visible")
        if row["agent"] == "oracle" and row["success"] < oracle_min_success:
            errors.append(f"{name}: oracle success {row['success']} below {oracle_min_success}")
    return errors


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--scene", default="0")
    ap.add_argument("--levels", nargs="+", default=["static", "small", "all"])
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--out", default="verify_report.json")
    ap.add_argument("--result_dir", default="verify_results")
    ap.add_argument("--judge", default="exact", choices=["exact", "llm"], help="llm: the configured LLM judge")
    ap.add_argument("--tasks", nargs="+", default=["nav", "eqa"])
    ap.add_argument("--agents", nargs="+", default=["oracle", "stale_oracle", "stop", "random"])
    ap.add_argument("--oracle-min-success", type=float, default=1.0,
                    help="Minimum positive-oracle success for validation (default: 1.0).")
    args = ap.parse_args(argv)
    if args.episodes < 1 or not 0 <= args.oracle_min_success <= 1:
        ap.error("--episodes must be positive and --oracle-min-success must be in [0, 1]")
    random.seed(0)
    report = {"uninavbench": uninavbench.__file__, "runs": []}
    try:
        for level in args.levels:
            for task in args.tasks:
                for name, cls, kw, steps in (("oracle", OracleAgent, {}, 300), ("stale_oracle", OracleAgent, {"stale": True}, 300),
                                             ("stop", StopAgent, {}, 1), ("random", RandomAgent, {}, 100)):
                    if (name == "stale_oracle" and task == "nav") or name not in args.agents:
                        continue
                    ORACLE_LOGS.clear()
                    EVENT_CHECK.clear()
                    EVENT_FAIL.clear()
                    m = evaluate(args.data, cls, kw, args.split, args.scene, level, task, args.episodes, steps, args.result_dir, args.judge)
                    report["environment"] = m["environment"]
                    results = []
                    for n in range(1, args.episodes + 1):
                        path = os.path.join(args.result_dir, task, args.split, str(args.scene), level, f"episode_results_{n}.json")
                        with open(path) as f:
                            results += list(json.load(f).values())
                    extra = {}
                    if task == "eqa":
                        same = [r for r in results if norm(r["predict"]) == norm(r["gt_answer"])]
                        diff = [r for r in results if norm(r["predict"]) != norm(r["gt_answer"])]
                        extra = dict(answer_equals_gt=len(same), answer_differs_from_gt=len(diff),
                                     success_when_equal=round(sum(r["success"] for r in same) / max(len(same), 1), 3),
                                     success_when_different=round(sum(r["success"] for r in diff) / max(len(diff), 1), 3))
                    else:
                        extra = dict(stopped=sum(r["stopped"] for r in results))
                    if name in ("oracle", "stale_oracle"):
                        # sub-tasks whose target cannot be seen from any reachable pose
                        extra["target_never_visible"] = count_never_visible(ORACLE_LOGS)
                        extra["oracle_failed_types"] = dict(collections.Counter(
                            str(r["task_type"]) for r in results if r["success"] < 1))
                    row = dict(level=level, task=task, agent=name, subtasks=m["num_subtasks"], success=m["success"],
                               spl=round(m["spl"], 4), per_task_type={k: round(v["success"], 3) for k, v in m["per_task_type"].items()},
                               **extra, events_checked=dict(EVENT_CHECK), event_failures=EVENT_FAIL[:5], n_event_failures=len(EVENT_FAIL))
                    print(json.dumps(row), flush=True)
                    report["runs"].append(row)
    except Exception as exc:
        report["execution_error"] = f"{type(exc).__name__}: {exc}"
    errors = validation_errors(report, args.oracle_min_success)
    report["validation"] = {"passed": not errors, "errors": errors,
                            "oracle_min_success": args.oracle_min_success}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    for error in errors:
        print(f"[validation failed] {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
