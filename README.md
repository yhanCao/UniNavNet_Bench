# UniNavBench

Long-horizon embodied navigation and Active Embodied Question Answering (Active-EQA) in indoor scenes that keep changing. 800 ProcTHOR houses (AI2-THOR 5) follow 1–3 day schedules of human activities: objects are moved between rooms and receptacles, food is sliced, the TV is switched on and off, dishes get dirty and are washed. Every episode is a sequence of 10 sub-tasks in the same changing scene.

This repository loads the dataset, replays the scene changes while your agent acts, and scores navigation and Active-EQA. Plug in your own agent, or use the optional **ST-Map** navigation baseline, packaged from ST-CogNav commit `5ac9398`. See [ST-Map setup and usage](docs/st-map.md) for its separate CUDA/model environment, configuration, and limitations.

## Installation

From a clone of this repository:

```bash
git clone https://github.com/yhanCao/UniNavNet_Bench.git && cd UniNavNet_Bench
pip install .
```

Requirements:

- Linux with Python 3.9+ (installation tested on 3.9 and 3.10). Use a virtual environment to keep AI2-THOR's dependencies off your system Python. The package is not on PyPI yet; `pip install uninavbench` will work once it is published.
- `git` on the `PATH` and network access to GitHub: the ProcTHOR houses (~120 MB) are fetched from GitHub with `git` on first use. `git-lfs` is installed automatically if it is missing.
- A display: a desktop session, or an X server on a headless machine.
- About 1 GB of disk space for the AI2-THOR simulator, which is downloaded to `~/.ai2thor` on first use, plus the dataset (see below).

On first use AI2-THOR prints `There has been an update to ProcTHOR-10K that must be used with AI2-THOR version 5.0+`. The installed version is already 5.0.0 and the ProcTHOR revision is pinned on purpose (below), so this warning can be ignored.

The default evaluation pins ProcTHOR revision `439193522244720b86d8c81cde2e51e3a4d150cf` and AI2-THOR build `f0825767cd50d69f666c7f282e54abfe58f1e917`. Metrics include both identifiers and installed package versions. Custom configs use separate `procthor_revision` and `ai2thor_commit_id` fields; replace the previously ignored `thor_build_id` key. See [failure handling and version pins](docs/evaluation-reliability.md).

## Dataset

```bash
uninavbench download --split test             # ~4.3 GB  -> ~/.cache/uninavbench/data
uninavbench download --split mini_test        # ~90 MB, three test scenes: 0, 4, 6
uninavbench download --split train val test   # everything (~44 GB)
uninavbench download --split test --scenes 0 4 7
```

Sizes are download sizes. The shards stay in the Hugging Face cache (`~/.cache/huggingface`) after extraction, so plan for about twice that disk space, or delete the cached shards once the data is extracted.

> **Dataset release complete.** All 33 shards for `train`, `val`, `test`, and `mini_test` are available in [yhncao/UniNavBench](https://huggingface.co/datasets/yhncao/UniNavBench), including the navigation completeness repair for train scenes 220 and 229 (manifest version `2026.11.14.1`). The Python package is not on PyPI yet; install it from this source checkout. See *Publishing the dataset* below.

`mini_test` is a convenience subset of `test` containing scenes 0, 4 and 6, with all three dynamic levels and both task families. It extracts into `test/` and `attributes/test/`, so evaluate it with `-p test` (for example, `uninavbench run -a examples/agents/random_agent.py -p test -i 0 -es 0 -ee 1`). The full test split still contains all 100 scenes; this subset is not an independent evaluation split.

The shards are hosted on the Hugging Face Hub (`yhncao/UniNavBench`); `--source modelscope` downloads from ModelScope instead (`pip install 'uninavbench[modelscope]'`), and `HF_ENDPOINT` can point to a Hugging Face mirror. Each shard is checked against the sha256 of the release manifest. Use another location with `--data_pth DIR` and `export UNINAVBENCH_DATA=DIR`, or `--from_dir` to read the shards straight from a local directory:

```bash
uninavbench download --split test --from_dir release/ --data_pth ~/.cache/uninavbench/data
```

| Split | Scenes | Navigation episodes | Active-EQA episodes |
| ----- | -----: | -----: | -----: |
| train | 500 | 3 × 1,000k | 3 × 1,000k |
| val   | 200 | 3 × 400k | 3 × 400k |
| test  | 100 | 3 × 200k | 3 × 200k |
| mini_test (subset of test) | 3 | 3 × 6k | 3 × 6k |

Three dynamic levels per scene: `static` (no change), `small` (objects on surfaces move, object states change), `all` (furniture moves as well).

## Quick start

`examples/agents/random_agent.py` is an editable example in this repository. It picks a random next action every step and occasionally stops and answers. The installed package also includes `uninavbench.agents.random_agent`, the default agent when `-a` is omitted; it moves randomly until the step limit.

```bash
uninavbench run -a examples/agents/random_agent.py -p test -l all -i 0 -t nav -es 0 -ee 10
uninavbench run -a examples/agents/random_agent.py -p test -l all -i 0 -t eqa -es 0 -ee 10   # needs a judge, see below
uninavbench batch -a examples/agents/random_agent.py -p test --levels static small all --tasks nav eqa -n 4
```

Or from Python:

```python
from uninavbench import Agent, Benchmark

class MyAgent(Agent):
    def reset(self):
        pass

    def act(self, obs):
        # obs["instruction"]  instruction (navigation) or question (Active-EQA)
        # obs["image_task"]   goal image path for image-goal navigation, else None
        # obs["event"]        ai2thor Event: RGB obs["event"].frame, depth obs["event"].depth_frame
        # obs["step"]         steps taken so far in the episode
        return "MoveAhead", None   # (action, answer); the answer given with the last action is scored

bench = Benchmark(agent_class=MyAgent)
metrics = bench.evaluate(split="test", level="all", scene=0, task="nav", start=0, end=10)
print(metrics["success"], metrics["spl"], metrics["per_task_type"])
```

Actions: `Stop`, `MoveAhead` (0.25 m), `RotateLeft`, `RotateRight` (90°), `LookUp`, `LookDown` (30°). A sub-task ends with `Stop` or after 500 steps. For the command line an agent is a module or `.py` file whose `build()` returns `(agent_class, agent_kwargs, render_depth)`.

## Tasks

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

## Evaluation protocol

- The scene clock counts minutes from 00:00 of day 1 and advances by one minute per agent step. A navigation episode of a dynamic level starts at its `time`; before an Active-EQA sub-task the scene is fast-forwarded to the question's time. Scheduled object relocations and state changes are applied when the clock reaches them.
- Navigation succeeds when the agent issues `Stop` while a target is visible (within 1 m). SPL uses the shortest path from where the previous sub-task ended.
- If a scheduled relocation lands on the agent or walls it into a pocket (it then reaches less than half of the episode's initial floor), the agent is moved to the nearest free position. This also covers episode starts: start poses are chosen in the initial scene, and furniture may have moved onto them by the episode's start time. Relocations are recorded per sub-task in `agent_relocations`; state changes such as an open door do not trigger them.
- Objects are unbreakable during evaluation. Replaying the recorded poses could otherwise break a laptop or a vase, which the schedule never does.
- A failed scheduled simulator action aborts evaluation with its time/action/object context; batch evaluation returns a nonzero exit status if any worker fails.
- An Active-EQA answer is scored from 1 to 5 by an LLM judge against the answer valid at the current scene time and mapped to (score − 1) / 4; it counts only if the queried object was seen during the episode (except memory captioning).
- Success and SPL are averaged over sub-tasks, overall and per task type.

The judge is any OpenAI-compatible chat API:

```bash
export UNINAVBENCH_JUDGE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export UNINAVBENCH_JUDGE_MODEL=qwen3.8-flash
export UNINAVBENCH_JUDGE_API_KEY=...
export UNINAVBENCH_JUDGE_EXTRA_BODY='{"enable_thinking": false}'   # optional, passed to the API
```

`UNINAVBENCH_JUDGE=exact` scores by exact string match instead (no API; for tests). Other settings are in `uninavbench/configs/procthor.yaml` (`-c` to use another file).

## Checking an installation

```bash
python tests/verify_dynamic.py ~/.cache/uninavbench/data --split test --scene 0 --episodes 2
```

runs privileged oracle agents through the installed package: the scheduled changes must appear in the simulator, an oracle that walks to a visible target and stops must succeed, an agent that stops at once must fail, and Active-EQA must be scored against the answer valid at the current time.

## Publishing the dataset (maintainers)

```bash
python scripts/build_release.py /path/to/Dynamic_Strategy release/ --resume  # train/val/test + mini_test
huggingface-cli upload yhncao/UniNavBench release/ . --repo-type dataset
modelscope upload yhncao/UniNavBench release/ --repo-type dataset            # mirror
```

`UNINAVBENCH_REPO` (or `--repo`) selects another dataset repository.

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
