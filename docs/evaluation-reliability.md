# Evaluation failure handling and version pins

The default environment is fixed to:

- ProcTHOR house-loader revision: `439193522244720b86d8c81cde2e51e3a4d150cf`.
- AI2-THOR executable commit: `f0825767cd50d69f666c7f282e54abfe58f1e917`.
- Python packages: `ai2thor==5.0.0`, `prior==1.0.3`.

These are the local cached loader and installed executable used in the previous
verification environment. They provide an explicit reproducible evaluation
baseline; they are not proof that every historical data-generation run used this
exact environment.

`procthor_revision` is passed to PRIOR for both online and offline loads.
`ai2thor_commit_id` is passed to every evaluation Controller. Both fields can be
changed in custom YAML, but must be full immutable commit SHAs. Evaluation metrics
record these identifiers and the installed Python package versions in `environment`.
The dynamic validation report includes the same environment information.

The old `thor_build_id: ab3cacd...` setting was ignored and confuses two different
identifiers: that SHA identifies an older ProcTHOR revision. Remove that key from
custom configurations and use the two fields in the shipped YAML. Overrides of
commit/branch/local-executable settings inside `initialize` are rejected so they
cannot silently bypass the simulator pin.

Scheduled state changes, recorded-pose teleports and fallback placements must
return `lastActionSuccess=True`. Raised simulator errors or unsuccessful returns
raise `DynamicReplayError` with the scheduled time, group, action and object. The
evaluation stops and its controller closes. Events are not silently skipped or
retried after partially applying a group. Resolved teleports report every affected
object ID so target observations/planners refresh when secondary objects move.

This checks execution failures; it does not establish that a successful action
achieved the exact recorded pose. The existing verification driver still compares
positions/states, retaining its 5 cm positional tolerance. The previously observed
vertical discrepancy was subsequently traced to missing kinematic state during
resolved-pose replay and fixed; see [pose replay diagnosis](pose-replay-diagnosis.md).

`uninavbench batch` exits 1 if any subprocess fails; successful and skipped work
exits 0. Both `python -m uninavbench` and the console entry point propagate that
status. The dynamic verification driver exits 1 for replay mismatches, simulator
or evaluation exceptions, invisible oracle targets, empty runs, or insufficient
positive-oracle success. It saves errors in the JSON report under `validation`.
Execution exceptions are also saved under `execution_error`.

Random, stop and stale-oracle negative controls may legitimately fail their tasks;
those task failures alone do not fail validation. The positive oracle defaults to
100% required success. `--oracle-min-success` can set an explicit [0,1] threshold,
which is included in the report. This does not relax the replay mismatch gate.

```bash
python -m pytest tests
python tests/verify_dynamic.py /path/to/data --scene 0 --episodes 1 \
  --levels static small all --tasks nav --agents oracle --out report.json
```

A nonzero status is a failed validation even if the navigation success printed
beforehand was 100%.

## Verification on 2026-11-15

- Full regression suite: **61 passed** (26 existing tests plus 35 new cases).
- Real pinned simulator: a state toggle and a recorded-pose teleport succeeded;
  an invalid object ID raised the contextual replay error. Controller cleanup
  completed afterward.
- Real test/0 static navigation oracle: 10/10 subtasks succeeded, SPL 0.8103,
  validation passed, process exit status **0**.
- Real test/0 small navigation oracle: 10/10 subtasks succeeded, SPL 0.8200,
  11 state and 25 pose checks. The existing GarbageCan vertical offset was
  reproduced (about 0.0638 m); the validation report failed and the process
  exited **1**. This is the intended gate behavior, not a resolved pose discrepancy.
- Wheel and sdist built successfully. A wheel installed outside the checkout
  exposes the pinned YAML, declares exact simulator/PRIOR package dependencies,
  and propagates a failed batch worker through the installed console entry point.

The validation reports and build logs were kept by the maintainers and are not part of this repository.

## Follow-up: recorded-pose physics state

Resolved-pose teleports now pass `forceKinematic=True` so snapshot coordinates
remain stable, matching the recording pipeline. The complete diagnosis and its
controlled experiments are in [pose replay diagnosis](pose-replay-diagnosis.md).
The same test/0 small/all navigation and EQA oracle cases now pass: 40/40
subtasks, 86 state checks and 214 pose checks, zero mismatches, exit status 0.
The full unit-test suite now has 62 passing cases. The earlier failed gate
results above are retained as historical before-fix evidence.
