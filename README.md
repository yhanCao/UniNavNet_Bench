# UniNavBench

Long-horizon embodied navigation and Active Embodied Question Answering (Active-EQA) in indoor scenes that keep changing. 800 ProcTHOR houses (AI2-THOR 5) follow 1–3 day schedules of human activities: objects are moved between rooms and receptacles, food is sliced, the TV is switched on and off, dishes get dirty and are washed. Every episode is a sequence of 10 sub-tasks in the same changing scene.

This repository downloads the dataset, replays the scene changes while an agent acts, and scores navigation and Active-EQA.

1. [Dataset](#1-dataset)
2. [Running the benchmark](#2-running-the-benchmark)
3. [ST-CogNav baseline](#3-st-cognav-baseline)
4. [Adding your own agent](#4-adding-your-own-agent)

## 1. Dataset

### Splits and tasks

| Split | Scenes | Navigation episodes | Active-EQA episodes |
| ----- | -----: | -----: | -----: |
| train | 500 | 3 × 1,000k | 3 × 1,000k |
| val   | 200 | 3 × 400k | 3 × 400k |
| test  | 100 | 3 × 200k | 3 × 200k |
| mini_test (subset of test) | 3 | 3 × 6k | 3 × 6k |

Each scene has three dynamic levels: `static` (no change), `small` (objects on surfaces move, object states change) and `all` (furniture moves as well). An episode has 10 sub-tasks of one family, navigation or Active-EQA:

| `task_type` | Task | Example |
| :-: | ----- | ----- |
| 1 | Object navigation | Find the Book |
| 2 | Relation navigation | Find the CellPhone which is on the SideTable |
| 3 | Demand-driven navigation | I need something to cook something in. |
| 4 | Attribute navigation | Find the green, plastic Cup |
| 5 | Image-goal navigation | Find the Fridge showing in the image. |
| 6 | Attribute recognition | What color is the Painting in the LivingRoom ? |
| 7 | State recognition | Is the Plate in the Kitchen dirty ? |
| 8 | Object recognition | What is on the TVStand in the Bedroom? |
| 9 | Object localization | Where is my Watch? |
| 10 | Object counting | How many Books are there in the Bedroom? |
| 11 | Memory captioning | Has the 'make breakfast' activity happened yet today? |

Navigation and Active-EQA sub-tasks each make up half of the dataset. In the dynamic levels 35% of the Active-EQA answers differ, at question time, from the initial scene, and 44% of the navigation targets are moved or change state during the schedule. Every navigation target can be seen from some reachable pose at every time of the schedule, and no scheduled change cuts off part of the floor.

### Download

The dataset is hosted on the Hugging Face Hub at [yhncao/UniNavBench](https://huggingface.co/datasets/yhncao/UniNavBench) (current manifest version `2026.11.16.1`). After [installing the package](#installation):

```bash
uninavbench download --split mini_test        # ~90 MB, three test scenes: 0, 4, 6
uninavbench download --split test             # ~4.3 GB  -> ~/.cache/uninavbench/data
uninavbench download --split train val test   # everything (~44 GB)
uninavbench download --split test --scenes 0 4 7
```

Every shard is checked against the sha256 of the release manifest. Sizes are download sizes: the shards stay in the Hugging Face cache (`~/.cache/huggingface`) after extraction, so plan for about twice that disk space, or delete the cached shards afterwards.

- `--data_pth DIR` extracts elsewhere; then `export UNINAVBENCH_DATA=DIR` for evaluation.
- `--source modelscope` downloads from ModelScope (`pip install 'uninavbench[modelscope]'`); `HF_ENDPOINT` can point to a Hugging Face mirror.
- `--from_dir DIR` reads shards from a local copy of the release.

`mini_test` contains test scenes 0, 4 and 6 with all levels and both task families, for installation checks and development. It extracts into `test/`, so evaluate it with `-p test`. It is not an independent evaluation split; report results on the full test split.

### Layout

```text
<data>/{train,val,test}/<scene>/episodes/{static,small,all}/{nav,eqa}.json.gz
<data>/{train,val,test}/<scene>/dynamic_strategy/strategy_final_{small,all}.json
<data>/attributes/{train,val,test}/<scene>/images/*.png      # goals of image-goal navigation
```

An episode file is a list of episodes. Each episode holds the start pose (`initial_position`, `initial_orientation`, `initial_horizon`), the scene `time` (minutes from 00:00 of day 1; the start time of a dynamic navigation episode, the time of each Active-EQA question), and per sub-task the `instruction`, `task_type`, target (`id` / `object_type`), goal `image_name`, and for Active-EQA the `answer` (a dict from scene time to the answer valid from then on). A schedule (`strategy_final_*.json`) lists the timed object relocations and state changes with the recorded resulting poses. Repairs of the released data are described on the [dataset page](https://huggingface.co/datasets/yhncao/UniNavBench); [docs/release.md](docs/release.md) explains how the shards are built.

## 2. Running the benchmark

### Installation

```bash
git clone https://github.com/yhanCao/UniNavNet_Bench.git && cd UniNavNet_Bench
pip install .
```

Requirements:

- Linux with Python 3.9+ (tested on 3.9 and 3.10), preferably in a virtual environment. The package is not on PyPI yet.
- `git` on the `PATH` and network access to GitHub: the ProcTHOR houses (~120 MB) are fetched with `git` on first use (`git-lfs` is installed automatically).
- A display: a desktop session, or an X server on a headless machine.
- About 1 GB for the AI2-THOR simulator, downloaded to `~/.ai2thor` on first use.

On first use AI2-THOR prints `There has been an update to ProcTHOR-10K that must be used with AI2-THOR version 5.0+`. The installed version is already 5.0.0 and the ProcTHOR revision is pinned on purpose, so the warning can be ignored.

### Evaluate an agent

```bash
# one scene, level and task family; episodes [es, ee)
uninavbench run -a examples/agents/random_agent.py -p test -i 0 -l all -t nav -es 0 -ee 10
uninavbench run -a examples/agents/random_agent.py -p test -i 0 -l all -t eqa -es 0 -ee 10   # needs a judge, below

# every scene of a split, all levels and task families, 4 simulators in parallel (resumable)
uninavbench batch -a examples/agents/random_agent.py -p test --levels static small all --tasks nav eqa -n 4
```

`-a` takes a `.py` file or an importable module (see [Adding your own agent](#4-adding-your-own-agent)); without it a built-in random agent runs. `run` writes the summary metrics to `-o` (default `metrics.json.gz`) and per-episode trajectories and results to `-r` (default `results/`); `batch` writes one metrics file per scene, level and task to `--out` and logs to `--logs`, and exits with a nonzero status if any run fails. `-c` selects another benchmark config (default [uninavbench/configs/procthor.yaml](uninavbench/configs/procthor.yaml)).

Active-EQA answers are scored by an LLM judge, any OpenAI-compatible chat API:

```bash
export UNINAVBENCH_JUDGE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export UNINAVBENCH_JUDGE_MODEL=gpt-6.2-sol
export UNINAVBENCH_JUDGE_API_KEY=...
export UNINAVBENCH_JUDGE_EXTRA_BODY='{"enable_thinking": false}'   # optional, passed to the API
```

`UNINAVBENCH_JUDGE=exact` scores by exact string match instead (no API; for tests).

### Evaluation protocol

- Actions: `Stop`, `MoveAhead` (0.25 m), `RotateLeft`, `RotateRight` (90°), `LookUp`, `LookDown` (30°). A sub-task ends with `Stop` or after 200 steps.
- The scene clock counts minutes from 00:00 of day 1 and advances by one minute per step. A navigation episode of a dynamic level starts at its `time`; before an Active-EQA sub-task the scene is fast-forwarded to the question's time. Scheduled relocations and state changes are applied when the clock reaches them.
- Navigation succeeds when the agent issues `Stop` while a target is visible (within 1 m). SPL uses the shortest path from where the previous sub-task ended.
- An Active-EQA answer is scored from 1 to 5 by the judge against the answer valid at the current scene time and mapped to (score − 1) / 4. It counts only if the queried object was seen during the episode (except memory captioning).
- If a scheduled relocation lands on the agent or walls it into a pocket (it then reaches less than half of the episode's initial floor), the agent is moved to the nearest free position, also at episode starts. Relocations are recorded per sub-task in `agent_relocations`; state changes such as an open door do not trigger them.
- Objects are unbreakable during evaluation; replaying recorded poses could otherwise break a laptop or a vase, which the schedule never does.
- A failed scheduled simulator action aborts the run with its time, action and object.
- Success and SPL are averaged over sub-tasks, overall and per task type.

The evaluation pins ProcTHOR revision `439193522244720b86d8c81cde2e51e3a4d150cf` and AI2-THOR build `f0825767cd50d69f666c7f282e54abfe58f1e917` (config fields `procthor_revision`, `ai2thor_commit_id`); every metrics file records both and the installed package versions. See [docs/evaluation-reliability.md](docs/evaluation-reliability.md).

### Checking an installation

```bash
python tests/verify_dynamic.py ~/.cache/uninavbench/data --split test --scene 0 --episodes 2
```

runs privileged oracle agents through the installed package: the scheduled changes must appear in the simulator, an oracle that walks to a visible target and stops must succeed, an agent that stops at once must fail, and Active-EQA must be scored against the answer valid at the current time. The unit tests run with `python -m pytest`.

## 3. ST-CogNav baseline

The `st_map` package is the ST-CogNav navigation agent (commit `5ac9398`), packaged as **ST-Map**: OpenSeeD detection and CLIP features, RGB-D voxel mapping, a scene graph with temporal object states and a history of removed objects, LLM selection of the next graph node, VLM verification of the goal, and a local planner. It uses the simulator's reachable positions and scene bounds for its map, so it is a privileged geometric baseline.

**Environment.** A CUDA machine with Python 3.10, PyTorch, the `st-map` extra, detectron2 / pytorch3d / gradslam / panopticapi built from source, the OpenSeeD deformable-attention extension, and the OpenSeeD checkpoint `openseed_swinl_pano_sota.pt`. Step-by-step instructions are in [docs/st-map.md](docs/st-map.md).

```bash
pip install '.[st-map]'
```

**Configuration.** Copy [st_map/configs/default.yaml](st_map/configs/default.yaml) and set the checkpoint and the LLM / VLM endpoints (any OpenAI-compatible API; one multimodal model can serve both):

```yaml
openseed_checkpoint: /path/to/openseed_swinl_pano_sota.pt
llm: {base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1", model: qwen3.8-flash}
vlm: {base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1", model: qwen3.8-flash}
```

**Run.**

```bash
export ST_MAP_CONFIG=/path/to/st-map.yaml
export ST_MAP_LLM_API_KEY=... ST_MAP_VLM_API_KEY=...
uninavbench run -a uninavbench.agents.st_map -p test -i 0 -l all -t nav -es 0 -ee 1
```

Notes:

- Speed: about 0.5 s per step on an RTX 4090 with 16.5 GB of GPU memory (one instance per 24 GB GPU).
- Tokens: thinking is off by default; the planner sees one line per candidate node and replies with a short JSON, and goal verification sends crops of the candidate scaled to 320 px. Token counts per call kind are written to `ST-Map/usage.json`.
- Memory and disk use stay bounded over an episode (a few views per object, a capped history archive, unused frames deleted). `debug_artifacts: true` additionally saves per-step data and plots.
- Active-EQA: the agent searches for the object a question mentions and answers once, when it stops or on its last step, from the current frame and close-ups of the target. This answer step is a minimal addition, not part of ST-CogNav.

## 4. Adding your own agent

An agent subclasses `uninavbench.Agent` and returns one action per step:

```python
# my_agent.py
from uninavbench import ALLOWED_ACTIONS, Agent


class MyAgent(Agent):
    def __init__(self, model_path=None):
        self.model_path = model_path            # load models here: one instance per episode

    def reset(self):
        pass                                    # required by the interface

    def act(self, observations):
        instruction = observations["instruction"]   # instruction (navigation) or question (Active-EQA)
        goal_image = observations["image_task"]     # goal image path for image-goal navigation, else None
        event = observations["event"]               # ai2thor Event: RGB event.frame, depth event.depth_frame,
                                                    # pose event.metadata["agent"]
        step = observations["step"]                 # steps taken so far in the episode
        action, answer = "MoveAhead", None
        return action, answer                   # action from ALLOWED_ACTIONS; answer for Active-EQA, else None


def build():
    # (agent class, constructor kwargs, render a depth image?)
    return MyAgent, {"model_path": "weights.pt"}, True
```

```bash
uninavbench run -a my_agent.py -p test -i 0 -l all -t nav -es 0 -ee 10
uninavbench batch -a my_agent.py -p test -n 4
```

- **Lifecycle.** The evaluator creates a new agent for every episode and calls `act` until the sub-task ends; the next sub-task of the episode continues with the same agent, the scene as it is and a new `instruction`, so memory carried across sub-tasks is allowed and useful. A sub-task ends when `act` returns `Stop` or after 200 steps.
- **Answers.** For Active-EQA the answer returned with the last action of a sub-task is scored: return it together with `Stop`, or on the 200th step if the agent does not stop.
- **Depth.** `build()`'s third value turns depth rendering on (`event.depth_frame`); leave it off if the agent does not need it.
- **Python API.** The same agent runs without the command line:

  ```python
  from uninavbench import Benchmark
  bench = Benchmark(agent_class=MyAgent, agent_kwargs={"model_path": "weights.pt"})
  metrics = bench.evaluate(split="test", level="all", scene=0, task="nav", start=0, end=10)
  print(metrics["success"], metrics["spl"], metrics["per_task_type"])
  ```

- **Optional hooks.** `start_task(instruction, reachable_positions)` is called at the start of each sub-task; `bind_environment(controller, output_dir)` once per episode with the AI2-THOR controller and an output directory; `update_environment(reachable_positions)` after scheduled scene changes. They give privileged access to the simulator (ST-Map uses them); say so when reporting results of an agent that uses them.
- **Example.** [examples/agents/random_agent.py](examples/agents/random_agent.py) is a complete, runnable agent to start from.

Report success and SPL on the full test split, overall and per task type and level, together with the `environment` block of the metrics files.

## License

The evaluation code is released under the Apache License 2.0 (see [LICENSE](LICENSE)). The dataset annotations, dynamic schedules and goal images on the Hugging Face Hub are released under Apache-2.0. The ProcTHOR houses and the AI2-THOR simulator are downloaded from their upstream sources and remain under their own licenses. Third-party code vendored in `st_map/` keeps its original notices (see [st_map/NOTICE](st_map/NOTICE)).

## Citation

<!-- TODO(release): fill in once the paper is accepted -->
```bibtex
@article{uninavbench,
  title   = {TODO},
  author  = {TODO},
  journal = {The International Journal of Robotics Research},
  year    = {TODO}
}
```
