# ST-Map

ST-Map packages the complete navigation/mapping algorithm from ST-CogNav commit
`5ac93986c2db14d9f25143d86ed0b55ad7f8735c` inside UniNavBench. It does not require
`RobotHOR/robothor_dynamic_CogNav` at runtime. `st_map/UPSTREAM.json` records the
original source hashes; `st_map/NOTICE` describes provenance and integration changes.

It retains OpenSeeD detection, CLIP features, RGB-D voxel fusion, object association,
temporal state sequences, removal to a history archive, instruction-related history,
optional VLM history captions, relevant graph nodes/frontiers in prompts, LLM node
selection, VLM goal verification, and local navigation. The original class named
`SimpleRandomAgent` is a model-based planner despite its historical name.

## Environment

Use a dedicated Linux CUDA environment with Python 3.10 (the upstream setup uses
PyTorch 2.6.0, torchvision 0.21.0, CUDA 12.4). The base UniNavBench package still
supports Python 3.9+; the full optional model environment has tighter compatibility
requirements. Importing `st_map` or calling `build()` does not load models.

```bash
conda create -n st-map python=3.10
conda activate st-map
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install '.[st-map]'
```

Some upstream dependencies require source installation and a matching CUDA compiler:

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install 'git+https://github.com/facebookresearch/pytorch3d.git'
pip install 'git+https://github.com/gradslam/gradslam.git@conceptfusion'
pip install 'git+https://github.com/cocodataset/panopticapi.git'
```

OpenSeeD's deformable-attention CUDA source is included in the installed package.
Compile its extension after installing PyTorch and setting CUDA_HOME appropriately:

```bash
ST_MAP_OPS_DIR=$(python -c 'from st_map.config import resource_path; print(resource_path("openseed/body/encoder/ops"))')
cd "$ST_MAP_OPS_DIR"
python -m pip install . --no-build-isolation
```

For a read-only installation, copy this directory to a writable build directory
first. Compiling this extension does not compile the rest of ST-Map.

Download/provide the upstream OpenSeeD checkpoint `openseed_swinl_pano_sota.pt`.
The text embedding model, CLIP weights and OpenSeeD tokenizer are retrieved by their
respective libraries on first use (or point configuration at local cached assets).
Weights, datasets, original output artifacts and upstream credentials are excluded.
The optional extra alone does **not** install the four source dependencies or the
CUDA extension above.

## Configuration and evaluation

Copy `st_map/configs/default.yaml` to a location of your choice, configure your
LLM/VLM models and endpoints, then run:

```bash
export ST_MAP_CONFIG=/absolute/path/to/st-map.yaml
export ST_MAP_OPENSEED_CHECKPOINT=/absolute/path/to/openseed_swinl_pano_sota.pt
export ST_MAP_LLM_API_KEY=...   # only if required by your server
export ST_MAP_VLM_API_KEY=...
uninavbench run -a uninavbench.agents.st_map -p test -l all -i 0 -t nav -es 0 -ee 1
# Equivalent entry points: -a st_map or -a examples/agents/st_map.py
```

Any OpenAI-compatible endpoint works; one multimodal model can serve both roles:

```yaml
llm: {base_url: "https://api.openai.com/v1", model: gpt-6.2-sol}
vlm: {base_url: "https://api.openai.com/v1", model: gpt-6.2-sol}
```

with the key in `ST_MAP_LLM_API_KEY` and `ST_MAP_VLM_API_KEY`. `extra_body` passes provider-specific
request fields (for example `{reasoning_effort: low}` for OpenAI reasoning models). `enable_thinking` is a
DashScope / Qwen field: it is only sent when set, e.g. `enable_thinking: false` to turn Qwen3 thinking off.
Requests adapt to the provider: a server that rejects `max_tokens` gets `max_completion_tokens`, and one that
rejects a temperature gets its default.

### Token use

Thinking or reasoning multiplies output tokens: keep it off or low (`enable_thinking: false` on DashScope,
`extra_body: {reasoning_effort: low}` on OpenAI). With a reasoning model, the output caps below also have to
leave room for its reasoning tokens (or set them to `null`). The planner prompt describes each candidate node on one line (frontier / explored, path
distance and direction, nearby objects) and asks for `{"reason": <= 15 words, "result": node}`, capped at
`llm.max_tokens` (64). Target verification sends up to three views of the candidate, each cropped to the
object's detection box plus `vlm.crop_margin` and scaled to `vlm.image_max_side` (320 px, about 100 image
tokens instead of about 335 for a full frame), and accepts `vlm.max_tokens` (8) of output. Token counts per
call kind (`llm`, `verify`, `answer`, `caption`) are written to `ST-Map/usage.json` with each EQA answer
and are available as `st_map._vendor.vlm.client.USAGE`.

### Bounded memory

Memory and disk use do not grow with the episode length. Each object keeps its `memory.views_per_object`
detections with the largest pixel area plus the latest, its last `memory.class_votes` labels and
`memory.segments_per_object` state segments, and in its state record only the current step and the last
step it was visible. The history archive keeps the `memory.archive_limit` most recent removed objects
without point clouds, masks or features. Saved frames and camera poses that no object refers to are
deleted, and visited places are kept as grid cells. Per-step RGB-D dumps, detection overlays and path plots
are written only with `debug_artifacts: true`.

### Active-EQA answers

For a question (an instruction ending in `?`), the agent answers once, when it stops or at step
`max_task_steps` (200, the benchmark's limit): the VLM receives the question, the current frame and up to
two close-ups of the verified target (or of the object most similar to the question), and replies in at
most 10 words.

`ST_MAP_LLM_BASE_URL`, `ST_MAP_LLM_MODEL`, `ST_MAP_VLM_BASE_URL`,
`ST_MAP_VLM_MODEL`, `ST_MAP_EMBEDDING_MODEL`, `ST_MAP_CLIP_PRETRAINED`, and
`ST_MAP_OUTPUT_DIR` override configuration fields. YAML paths are relative to the
YAML file. History captioning defaults off; when enabled its `memory.caption_vlm`
client has its own configuration. `mapping_config` and `openseed_config` may point
to custom YAML files, otherwise installed resources are used. No global working
directory changes or `sys.path` aliases are required.

The benchmark binds a new agent once per episode, then calls `start_task` for every
subtask, including repeated instructions. Maps, archives and the global step clock
persist within an episode. Navigation goals, local plans and target similarity are
reset per subtask. Camera correction actions consume benchmark steps as ordinary
actions. Each episode writes artifacts under its result directory's `ST-Map/`.
After scheduled scene changes, the optional `update_environment` hook refreshes
reachability and discards obsolete action plans. Existing agents require no
lifecycle changes.

The upstream geometry assumes 640×480 RGB-D images, 0.25 m grid spacing and 90°
rotation steps; incompatible simulator settings are rejected. CUDA is required.
As in the original implementation, the planner uses simulator reachability and
scene object bounds to establish the map extent. This is a privileged geometric
baseline, not a claim of exclusively RGB-D input. It does not use target IDs or
reference answers for planning.

The upstream planner has no EQA policy: it searches for the object a question mentions as it would for a
navigation target. The answer step above is a minimal addition for evaluation, not part of ST-CogNav.

## Python library

```python
from st_map import STMapAgent
from uninavbench import Benchmark

benchmark = Benchmark(agent_class=STMapAgent, agent_kwargs={"config": "/path/to/st-map.yaml"})
metrics = benchmark.evaluate(split="test", level="all", scene=0, task="nav", start=0, end=1)
```

For an existing simulator, construct `STMapAgent(config=..., controller=controller)`,
call `start_task(instruction, reachable_positions)`, and pass the normal observation
dictionary to `act()`. Call `start_task` at each task boundary; `reset()` clears the
whole episode and requires a new environment binding. `agent.memory` exposes the
underlying mapper, including `objects`, `history_archive`, and
`retrieveHistoryContext(image_directory)` after setting its instruction embedding.

`from st_map import STMap` provides a standalone mapper facade:
`STMap(fov, size, min_bound, config=...)`. Its `backend` is the original torch module;
methods and read access delegate to it. `size`/`min_bound` follow upstream's (x,z,y)
voxel convention. `getGlobalMap` preserves the upstream tensor/observation interface.
This low-level interface needs a target embedding before object association; use
`mapper.textmodel.compute_target_feat(instruction)` before processing frames.

`st_map.memory.TemporalMemoryMixin` and `st_map.graph` expose temporal-memory methods
and prompt-subgraph utilities independently of detection models and CUDA. They use
the original object dictionaries and NetworkX graphs.

## Validation and remaining limits

Run `python -m pytest tests`. Lightweight tests cover memory archival/state
transitions, histogram mismatch, prompt filtering, nearest frontiers, lifecycle
persistence, configuration and optional imports. These do not replace evaluation
with the actual model stack. Full navigation quality and CUDA-extension compatibility
must be checked in a provisioned GPU environment with the checkpoint and model
services. Missing assets or unavailable CUDA produce explicit startup errors.

Integration fixes include bounded LLM retries, rejecting node selections outside
the supplied candidates, the upstream empty-plan goal-rotation argument error,
refreshing stored reachable coordinates after furniture moves, removing global
autograd disabling, and avoiding separate backbone initialization
from a URL already covered by the OpenSeeD checkpoint.
