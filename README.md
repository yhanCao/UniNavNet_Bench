# OmniDy-Bench: A Unified Benchmark for Life-long Dynamic Embodied Navigation

## :robot: OmniDy-Bench
put the teaser image

## :hammer: Installation

```bash
pip install uninavbench            # or, from a clone of this repository: pip install .
```

This installs the `uninavbench` Python package and the `uninavbench` command. The evaluator uses AI2-THOR 5.0 and the `procthor-10k` houses from `prior` (a cached copy is used when GitHub is unreachable). Then download the dataset (below) and point the benchmark to it:
```bash
export UNINAVBENCH_DATA=/path/to/Dynamic_Strategy
```

### Quick start
Write an agent (the benchmark has no planner of its own):
```python
from uninavbench import Agent, Benchmark

class MyAgent(Agent):
    def reset(self):
        pass

    def act(self, obs):
        # obs["instruction"]: instruction or question; obs["image_task"]: goal image path (image-goal nav) or None
        # obs["event"]: ai2thor Event (RGB obs["event"].frame, depth obs["event"].depth_frame); obs["step"]
        return "MoveAhead", None          # (action, answer); answer is scored for Active-EQA sub-tasks

bench = Benchmark(agent_class=MyAgent)    # data from $UNINAVBENCH_DATA, or Benchmark(data_pth=...)
metrics = bench.evaluate(split="test", level="all", scene=0, task="nav", start=0, end=10)
print(metrics["success"], metrics["spl"], metrics["per_task_type"])
```
Actions: `Stop`, `MoveAhead`, `RotateLeft`, `RotateRight`, `LookUp`, `LookDown`. A sub-task ends when the agent returns `Stop` (or after `max_steps`).

From the command line, an agent is a module (or `.py` file) with a `build()` function returning `(agent_class, agent_kwargs, render_depth)`, see `uninavbench/agents/random_agent.py`:
```bash
uninavbench run   -a my_agent.py -p test -l all -i 0 -t nav -es 0 -ee 2000 -o metrics.json.gz
uninavbench batch -a my_agent.py -p test --levels static small all --tasks nav eqa -n 4
```

### Active-EQA judge
Answers are scored by an LLM through any OpenAI-compatible chat API (`judge` in the config), e.g.
```bash
export UNINAVBENCH_JUDGE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export UNINAVBENCH_JUDGE_MODEL=qwen3.8-flash
export UNINAVBENCH_JUDGE_API_KEY=...              # never put keys in the config file
export UNINAVBENCH_JUDGE_EXTRA_BODY='{"enable_thinking": false}'
```
`UNINAVBENCH_JUDGE=exact` switches to exact string matching (offline tests).

### Checking an installation
`tests/verify_dynamic.py DATA_ROOT` runs privileged oracle agents through the installed package and checks that dynamic events are applied to the simulator, that navigation success matches target visibility after `Stop`, and that Active-EQA is scored against the answer valid at the current scene time.

## :floppy_disk: Dataset
Download the dataset and put it into the `dataset/` dir.
```
git lfs install
git clone https://huggingface.co/datasets/yhncao/UniDy_Bench
```
The code expects the following layout (pass another root with `-d` / `DATA=`):
```bash
  ├── OMNIDY_BENCHMARK/
  │  ├── dataset
  │  │  ├── Dynamic_Strategy/
  │  │  │  ├── attributes/
  │  │  │  │  ├── {train,val,test}/
  │  │  │  │  │   ├── {scene_id}/
  │  │  │  │  │   │   ├── description.json        # VLM attributes of every object (color, material, ...)
  │  │  │  │  │   │   ├── index.json              # number of images of every object
  │  │  │  │  │   │   ├── images/
  │  │  │  │  │   │   │   ├── Pen_surface_3_29_1.png   # objectId with "|" -> "_", then the view index
  │  │  │  ├── {train,val,test}/
  │  │  │  │  ├── {scene_id}/
  │  │  │  │  │   ├── dynamic_strategy/
  │  │  │  │  │   │   ├── strategy_final_small.json   # timeline of scene changes, Surface-Dynamic
  │  │  │  │  │   │   ├── strategy_final_all.json     # timeline of scene changes, Fully-Dynamic
  │  │  │  │  │   ├── episodes/
  │  │  │  │  │   │   ├── {static,small,all}/
  │  │  │  │  │   │   │   ├── nav.json
  │  │  │  │  │   │   │   ├── eqa.json
```
Episode files are plain JSON (`.json`); a `.json.gz` next to it is read instead when present.

### :heavy_exclamation_mark: Note
We select 500 train scenes, 200 validation scenes and 100 test scenes (800 scenes, 4,218 rooms, at least three rooms each) from ProcTHOR-10K.
Each scene has three levels of dynamic degree:
1. `static` — the initial scene, nothing changes.
2. `small` (Surface-Dynamic) — objects on surfaces are moved, and object states change.
3. `all` (Fully-Dynamic) — objects on surfaces and on the floor (furniture) are moved, and object states change.

Scene changes follow a schedule of human activities spanning 1–3 days (3 days for 82% of the scene levels). Time is measured in minutes from 00:00 of day 1 (0–4319) and advances by one minute per agent step. A dynamic strategy contains
- **object relocations** (`InitialRandomSpawn`, `PlaceObjectAtPoint`), including carrying portable objects between rooms;
- **object-state changes**: `ToggleObjectOn/Off` (TV, laptop, lamps), `OpenObject/CloseObject` (fridge, cabinets, drawers, books, laptop), `SliceObject` (bread, apple, tomato, lettuce, potato, egg), `DirtyObject/CleanObject` (dishes, cookware, bed).

| Level | Events | Relocations | State changes | Schedule length (scenes: 1 / 2 / 3 days) |
| ----- | -----: | ----------: | ------------: | ----- |
| small | 125,021 | 65,469 | 59,552 | 113 / 20 / 667 |
| all   | 120,867 | 65,180 | 55,687 | 123 / 34 / 643 |

### :clipboard: Tasks
Every episode is a long-horizon sequence of **10 sub-tasks** in the same scene; the scene keeps changing between sub-tasks.

| `task_type` | Task | Category | Share of sub-tasks |
| :-: | ----- | ----- | -----: |
| 1 | Object Navigation (ObjNav) | Navigation | 10.0% |
| 2 | Scene-Graph / Relationship Navigation | Navigation | 10.0% |
| 3 | Demand-Driven Navigation | Navigation | 10.0% |
| 4 | Attribute-based Navigation | Navigation | 10.0% |
| 5 | Image-goal Navigation | Navigation | 10.0% |
| 6 | Attribute Recognition | Active-EQA | 10.4% |
| 7 | State Recognition | Active-EQA | 4.7% |
| 8 | Object Recognition | Active-EQA | 2.2% |
| 9 | Object Localization | Active-EQA | 12.1% |
| 10 | Object Counting | Active-EQA | 15.5% |
| 11 | Memory Captioning | Active-EQA | 5.0% |

In the dynamic levels the answers of Active-EQA questions depend on the time the question is answered: 35% of the dynamic EQA sub-tasks have, at question time, an answer different from the initial scene (State Recognition 16–19%, Counting 38–40%, Object Localization 39–42%, Object Recognition 60–62%, Memory Captioning 56–59%). In the dynamic levels 44% of the navigation sub-tasks target an object that is moved or changes state during the schedule, and every navigation episode starts at a random time of the schedule (`time`), so the agent navigates while the scene changes.

Every target can be seen by the evaluated agent (from a reachable position, with the evaluator's camera and `visibilityDistance` 1.0 m): navigation targets during the whole day, Active-EQA objects at question time. Sub-tasks with hidden targets (inside closed receptacles, occluded, or out of reach) were replaced by others of the same type, and so were navigation targets visible from almost anywhere (Wall, Floor, Doorframe). Relation-navigation instructions of the dynamic levels describe relations that hold during the whole schedule, and attribute / state questions only ask about an object that is the only one of its category in the named room. Episode targets cover 97 object categories.

### :round_pushpin: Nav

| Split | Level | Scenes | Episodes | Files |
| ----- | ----- | -----: | -----: | ----- |
| Train | static / small / all | 500 | 1,000k per level | `dataset/Dynamic_Strategy/train/{scene_id}/episodes/{level}/nav.json` |
| Val   | static / small / all | 200 | 400k per level | `dataset/Dynamic_Strategy/val/{scene_id}/episodes/{level}/nav.json` |
| Test  | static / small / all | 100 | 200k per level | `dataset/Dynamic_Strategy/test/{scene_id}/episodes/{level}/nav.json` |

### :mag_right: A-EQA

| Split | Level | Scenes | Episodes | Files |
| ----- | ----- | -----: | -----: | ----- |
| Train | static / small / all | 500 | 1,000k per level | `dataset/Dynamic_Strategy/train/{scene_id}/episodes/{level}/eqa.json` |
| Val   | static / small / all | 200 | 400k per level | `dataset/Dynamic_Strategy/val/{scene_id}/episodes/{level}/eqa.json` |
| Test  | static / small / all | 100 | 200k per level | `dataset/Dynamic_Strategy/test/{scene_id}/episodes/{level}/eqa.json` |

In total: 9.6M episodes (4.8M navigation, 4.8M Active-EQA) and 96M sub-tasks; 2,000 episodes per scene, level and task family.

Each file is a list of episodes; every list-valued field has one entry per sub-task.

<details>
<summary><b>Episode Structure</b></summary>
<p>

```javascript
Nav Episode:
{
    'instruction': ['Find the Toilet showing in the image.', 'Find the AlarmClock showing in the image.',
                    'Find the GarbageCan in the Bathroom', 'I need something to put pepper on my food.', ...],
    'image_name': ['Toilet_5_0_1.png', 'AlarmClock_surface_4_12_3.png', None, None, ...],   // attributes/{split}/{scene_id}/images/
    'id': ['Toilet|5|0', 'AlarmClock|surface|4|12', 'GarbageCan|5|3', None, ...],            // target objectId
    'object_type': [None, None, None, ['SaltShaker'], ...],                                  // target categories when no id
    'task_type': [5, 5, 2, 3, ...],
    'level': 'all',
    'initial_horizon': 0,
    'initial_orientation': 0,
    'initial_position': {'x': 3.75, 'y': 0.9009993672370911, 'z': 4.0},
    'scene_type': 'test',
    'scene_id': 0,
    'time': 1138                       // dynamic levels: start time (minutes from day 1, 00:00)
}

A-EQA Episode:
{
    'time': [..., 236, ..., 436, ..., 768],          // minute of the day each question is asked (sorted)
    'instruction': [..., 'Is the Plate in the Kitchen dirty ?', ...,
                    'How many CreditCards are there in the Bedroom?', ...,
                    "Has the 'tidy up bedroom' activity happened yet today?"],
    'task_type': [..., 7, ..., 10, ..., 11],
    'id': [..., 'Plate|surface|6|26', ..., 'CreditCard|surface|4|14', ..., 'RemoteControl|surface|4|8'],
    'answer': [...,
               {'0': 'no', '480': 'yes', ...},     // time-indexed: the answer is the entry with the
               {'0': 1, ..., '640': 2, ...},       // latest time <= the current scene time
               {'0': 'No.', '600': 'Yes.'}],
    'object_type': [None, ...],
    'level': 'all', 'initial_horizon': 0, 'initial_orientation': 0,
    'initial_position': {'x': 0.25, 'y': 0.9009997844696045, 'z': 5.0},
    'scene_type': 'test', 'scene_id': 0
}

Dynamic strategy event (strategy_final_{small,all}.json, {"0": [event], "1": [event], ...}):
{'time': 570, 'activity': 'watch TV', 'action': 'ToggleObjectOn', 'objectId': 'Television|7|0|2',
 'state_change': True, 'success': True, ...}
{'time': 392, 'activity': 'wake up', 'action': 'InitialRandomSpawn', 'objectId': 'Cup|surface|6|23',
 'rectangleobj': 'SideTable|4|3', 'recommended_room': 0, 'carry': True, 'success': True,
 'resolved': [{'objectId': 'Cup|surface|6|23', 'position': {...}, 'rotation': {...}}]}   // pose to reproduce
```
</details>


## 🎯 Evaluation
```bash
python3 runner.py -a agents.random_agent -c ./configs/procthor.yaml -d ./dataset/Dynamic_Strategy -p test -l all -i 0 -t nav -es 0 -ee 2000
```
`-p` split, `-l` level (`static`/`small`/`all`), `-i` scene id, `-t` task family (`nav`/`eqa`), `-es`/`-ee` episode range, `-o` output metrics file.

To evaluate every scene of a split, all levels and both task families:
```bash
SPLIT=test LEVELS="static small all" TASKS="nav eqa" NPROC=4 DATA=./dataset/Dynamic_Strategy bash batch_run.sh
```
Each job writes `Result/metrics/{split}_{scene}_{level}_{task}.json.gz` (finished jobs are skipped, so a run can be resumed) and logs to `Result/logs/`. Under slurm, submit `batch_run.sh` as an array job (`sbatch --array=0-N`); each array task runs one job.

**Protocol.**
- The scene clock (minutes from day 1, 00:00) advances by one minute per agent step. A navigation episode of a dynamic level starts at its `time`; before an EQA sub-task starts, the scene is fast-forwarded to the question's `time`. Every dynamic event whose time has been reached is applied; relocations are reproduced exactly from their recorded `resolved` poses.
- A navigation sub-task succeeds when the agent issues `Stop` while a target object (or a piece of it, once sliced) is visible.
- An Active-EQA answer is scored by an LLM judge from 1 to 5 against the ground truth valid at the current scene time, mapped to (score − 1) / 4. The answer counts only if the queried object has been seen during the episode (`eqa_require_grounding` in the config, default `true`); Memory Captioning, which asks about the day's history, is exempt.
- SR and SPL are averaged over sub-tasks; per-task-type results are reported under `per_task_type`. For every sub-task the shortest path starts from the agent's position at the end of the previous sub-task.

`runner.py`, `batch_run.sh` and the `procthor_challenge` / `llm` / `agents` modules are kept as thin wrappers around the package for existing scripts.

### After installing and running the demo, you should see log messages that resemble the following:
```
2025-09-29 17:04:58,089 [INFO] procthor_challenge.challenge - Task Start instruction:What texture is the Laptop in Bedroom ? scene:train_1 initial_position:{'x': 5.0, 'y': 0.9009993672370911, 'z': 15.25} rotation:0
2025-09-29 17:04:58,423 [INFO] procthor_challenge.challenge - Agent action: LookUp
total_step: 0 action success: True total time: 0.09546089172363281
2025-09-29 17:04:58,641 [INFO] procthor_challenge.challenge - Agent action: LookDown
total_step: 1 action success: True total time: 0.07021379470825195
```
