# train/220 and train/229 navigation completeness repair

The six navigation JSON files were repaired on 2026-11-14. Each now contains
2000 episodes with 10 aligned subtasks each, and 4000 tasks of each navigation
type (1–5). EQA and dynamic strategies were not modified.

| Scene | Level | Added relation subtasks |
| --- | --- | ---: |
| 220 | static | 108 |
| 220 | small | 56 |
| 220 | all | 74 |
| 229 | static | 96 |
| 229 | small | 56 |
| 229 | all | 77 |

Total: 467 relation-navigation subtasks; 37 complete the six short last episodes
and 430 form 43 additional episodes. All preexisting task prefixes and episode
metadata are unchanged. The additional tail episodes contain relation navigation
only, to preserve both the original data and the exact global task-type quotas.

The original generator's relation branch skips selections where no suitable
object is found; the final grouping does not resample this shortfall. The backup
has the same missing tasks. This repair does not claim to recover the lost random
selections: it resamples existing relation tasks from the same scene and level,
consistent with the generator's sampling-with-replacement policy.

Each selected target is certified visible throughout the existing visibility
cache. Targets affected by any scheduled event are excluded. For spatial relations,
all reference categories affected by scheduled events are excluded conservatively;
self-reference relations are excluded. Original initial poses are reused, and new
episode start times follow the previous deterministic dynamic-time rule. This is
validation against the existing scene visibility evidence and schedule, not a new
full simulator sweep.

Original bytes, cache snapshots, source locations of every added task, random seed
recipe, before/after hashes and the applied report are retained by the maintainers.
The per-task provenance of the added tasks is published on the Hub as
`repairs/train_220_229_20261114.json`. The maintainers' repair directory contains:

- `original/`: exact original six files.
- `visibility-cache/`: the visibility evidence used by the repair.
- `repair-report.json`: complete per-task provenance and checksums.
- `release/train-005.tar`: repaired full shard, with only the six nav members replaced.
- `release/manifest.json`: local manifest for this single repaired shard.
- `release/train-005.repair.json`: shard checksum and list of replaced members.

An independent pass checked all six repaired files for counts, aligned arrays,
unchanged original prefixes/metadata and provenance. Six repair regression tests
cover quotas, idempotence, prefix preservation and rejection of invisible/moved
objects or moved relation references. The repaired tar's six entries were read back
and compared byte-for-byte with their expected compressed payloads.

The repaired shard was integrated after the release builder finished. The full
release was published to Hugging Face in commit
[`387dbd2d745aee3a0fb837030400d21358aa1677`](https://huggingface.co/datasets/yhncao/UniNavBench/commit/387dbd2d745aee3a0fb837030400d21358aa1677),
with manifest version `2026.11.14.1`. All 33 remote shards passed size and SHA-256
verification, covering train 500, val 200, test 100 and mini_test 3 scenes.
The canonical local shard and manifest now also include the repair; the earlier
shard is retained by the maintainers.
To download just these two repaired scenes:

```bash
uninavbench download --split train --scenes 220 229
```

The reproducible repair and archive-staging implementations are
`scripts/repair_nav_completeness.py` and `scripts/patch_nav_release.py`.
