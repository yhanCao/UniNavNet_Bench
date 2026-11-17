# ST-Map integration validation

Validated on 2026-11-14 against pinned upstream commit
`5ac93986c2db14d9f25143d86ed0b55ad7f8735c`.

- Full repository test suite: **20 passed**. Covers memory state classification,
  dual depth/appearance evidence, archive removal, similarity alignment, history
  retrieval, RGB histogram comparison, graph filtering/frontiers, task persistence,
  repeated instructions, dynamic geometry refresh, legacy-agent compatibility,
  configuration, namespace resolution and credential exclusion.
- Complete vendored navigation backend imports successfully in the existing
  `dy_ai2thor` environment: Python 3.10.18, PyTorch 2.6.0+cu124,
  torchvision 0.21.0+cu124, Open3D 0.19.0, Detectron2 0.6, timm 1.0.19,
  sentence-transformers 5.1.0, open-clip-torch 3.1.0.
- Namespaced deformable-attention wrapper executes the installed CUDA extension
  directly on a synthetic tensor and returns the expected GPU result. This check
  does not use the CPU fallback.
- Importing the backend preserves the application's autograd setting.
- Built wheel and sdist include mapper, memory, graph, Agent entry point, default
  config, OpenSeeD YAML/category list, source provenance and CUDA build sources.
- Installed wheel imports, Agent builders and resource lookups work from `/tmp`,
  outside the source checkout, without detector weights or CUDA in the lightweight
  verification environment.

The full detector/embedding/CLIP model initialization, LLM/VLM calls and a scored
simulator episode were not run. The upstream default checkpoint path is absent;
provide `ST_MAP_OPENSEED_CHECKPOINT` and configure model services before running
an episode. The import/CUDA checks establish integration compatibility, not
navigation accuracy or benchmark performance. The original planner has no EQA
answer-generation policy.

The test run reports the existing dataset tar-extraction deprecation warning and
an upstream CUDA-wrapper docstring escape warning; neither fails validation.
