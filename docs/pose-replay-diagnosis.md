# Cause and fix of the test/0 GarbageCan vertical discrepancy

Investigated on 2026-11-15 using the pinned ProcTHOR/AI2-THOR environment.
The affected example is test/0, small, time 660, `GarbageCan|5|3` (`bin_12`).

## What differed

The saved resolved transform has y=0.034396056 m. The original evaluator's
unconstrained teleport settles at y=-0.029394299 m, a vertical difference of
0.063790355 m. x/z agree within floating-point precision. Object position is the
asset's transform origin; a negative origin y alone does not mean the mesh is
below the floor. The settled bounding-box bottom is approximately zero.

## Root cause

The recording pipeline uses InitialRandomSpawn with placeStationary=True
(the API default) and allowMoveable=True. In the pinned simulator implementation,
RandomSpawnRequiredSceneObjects sets eligible rigid bodies to kinematic before
filtering by objectIds. Consequently, an unrelated successful random placement
can freeze the GarbageCan too. Later PlaceObjectAtPoint preserves that existing
kinematic state and leaves the GarbageCan at y=0.034396056 m. The generator reads
this actual metadata position into resolved; the number is not a transcription
error.

The old evaluator replaces these operations with TeleportObject calls that
restore position/rotation but do not restore the rigid-body state. From a fresh
scene the GarbageCan remains subject to gravity, so the same transform falls by
about 6.4 cm. Thus the mismatch is an incomplete recording/replay physics contract,
triggered by the simulator API's broader side effect, rather than numeric loss
or a coordinate-frame conversion error.

Primary sources for the exact executable commit:

- [InitialRandomSpawn defaults and dispatch](https://github.com/allenai/ai2thor/blob/f0825767cd50d69f666c7f282e54abfe58f1e917/unity/Assets/Scripts/BaseFPSAgentController.cs#L3347).
- [Kinematic assignment before the objectIds filter](https://github.com/allenai/ai2thor/blob/f0825767cd50d69f666c7f282e54abfe58f1e917/unity/Assets/Scripts/PhysicsSceneManager.cs#L594).
- [TeleportObject forceKinematic behavior](https://github.com/allenai/ai2thor/blob/f0825767cd50d69f666c7f282e54abfe58f1e917/unity/Assets/Scripts/PhysicsRemoteFPSAgentController.cs#L620).

## Controlled reproduction

| Operation in a fresh copy of the scene | Resulting origin y (m) | Bounding-box bottom y (m) |
| --- | ---: | ---: |
| Old teleport, physics enabled | -0.029394299 | ~0 |
| Same teleport, physics paused | 0.034396056 | 0.063791156 |
| Successful unrelated random placement, then original PlaceObjectAtPoint | 0.034396056 | 0.063793123 |
| Recorded-pose teleport with forceKinematic=True | 0.034396056 | 0.063793063 |

Each resulting position stayed unchanged through six subsequent Done actions.
The third case reproduces the recorded value exactly without assigning that
value to the GarbageCan directly. This distinguishes a physics-state mismatch
from a wrong recorded coordinate.

## Fix and limits

Resolved poses now use TeleportObject(forceAction=True, forceKinematic=True).
This defines them as stable scene snapshots, as expected by scheduled navigation
and EQA replay. It does not pause global physics, change raw placement fallback
behavior, alter coordinates, or relax the existing 5 cm verification tolerance.
Dataset shard contents and recorded answers remain unchanged.

This restores snapshot reproducibility; it does not make the original floating
placement physically grounded. A future dataset that requires natural settling
would need generation-side physical-state control and regenerated dependent
annotations/visibility evidence rather than a hardcoded y-coordinate correction.

The experiment artifacts (`experiment.json`, `frozen-state-confirmed.json`,
`summary.json`, `fixed-gate.json`) were kept by the maintainers. Regression coverage asserts that all resolved objects receive
stable teleports while original placement operations retain their API behavior.

## Verification after the fix

- Full unit/regression suite: **62 passed**.
- Real pinned simulator, test/0, small/all × nav/eqa: **40/40 oracle subtasks**
  succeeded, **86 state checks and 214 pose checks**, **zero mismatches**.
- The verification driver exited **0** with its original 5 cm positional tolerance.
- This is sampled validation of the diagnosed mechanism, not a full-corpus replay
  certification. No dataset coordinates or answers were modified.
