# shapley_eval compatibility changelog

This note is intended for presentation/review so the rewrite can be explained as a controlled extension, not a workflow break.

## What stayed the same

- Existing collaborator workflow is preserved as default (`method="snapshot"`).
- Existing `compute_coalition_utilities(...)` call without method args still works.
- Existing output schemas are unchanged:
  - `coalition_utilities.csv`: `coalition,size,val_mean_dice,val_tc,val_wt,val_et`
  - `allocation_summary.csv`: `client,shapley,least_core`
- Existing `run_shapley_from_coalitions(...)` flow remains valid.

## What changed

- Added explicit utility-generation modes in `compute_coalition_utilities(...)`:
  - `snapshot` (default): average final local snapshots per coalition.
  - `or`: one-round approximation using stored per-round singleton updates.
  - `mr`: multi-round approximation using stored per-round coalition trajectories.
- Added performance options:
  - `max_samples` to cap validation examples during coalition evaluation.
  - `save_every` to reduce CSV write frequency.
  - evaluator/model reuse to avoid unnecessary repeated construction/copying.
- Added `run_shapley_eval(...)` as a one-call orchestrator wrapper.

## Why this was done

- Keep current collaborator experiments runnable with no API break.
- Restore ability to run approximation modes consistent with OR/MR definitions.
- Reduce runtime/memory overhead in post-hoc coalition evaluation.

## Minimal migration pattern

```python
accuracy_dict = compute_coalition_utilities(...)
```

still works and is equivalent to:

```python
accuracy_dict = compute_coalition_utilities(..., method="snapshot")
```

To run OR/MR explicitly:

```python
compute_coalition_utilities(..., method="or", round_artifacts_dir="submodels/round_artifacts")
compute_coalition_utilities(..., method="mr", round_artifacts_dir="submodels/round_artifacts")
```
