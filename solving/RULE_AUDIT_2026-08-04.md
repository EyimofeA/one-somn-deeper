# Rule and upstream audit — 2026-08-04 (Author: Codex)

## Scope

Read-only comparison of the local competition checkout at
`8a3c78d6eae4047b07cd8c617c1b311f544a0e9f` against upstream
`e32c2f985f8ed4107c96d00271448777954ecc0c` (`2026-08-03T23:44:35Z`,
`Multiple backward passes supported`). The local checkout remains at the old
pin; only its remote-tracking reference was fetched. The live problem page was
read on 2026-08-04.

## Live rules and scoring

- Live problem page: Hard ranks certified in-distribution Max T, then certified
  OOD-N Max T, then accuracy at each first uncertified rung. The recurrence may
  change on Hard. Source: https://onelayerdeeper.ai/problem (read 2026-08-04).
- The upstream deadline remains Monday, August 31, 2026, 10:00 PM PT.
- No score/rung generation or data files changed in `e32c2f9`.

## Upstream delta: `8a3c78d` → `e32c2f9`

Changed files:

```
README.md
benchmark/__init__.py
benchmark/api.py
benchmark/runner.py
benchmark/validation.py
service/views.py
tests/test_contract.py
tests/test_runner_budget.py
```

### New API fields

`OptimizerBundle` now supports:

- `backward_passes_per_step: int = 1` (evaluator caps this at 8);
- `between_backward_passes(BackwardPassContext)`;
- `should_reuse_batch(BatchReuseContext) -> bool` (evaluator caps a batch at
  eight optimizer updates).

New public context types are `BackwardPassContext(completed_steps, pass_index,
total_passes)` and `BatchReuseContext(completed_steps, current_batch_uses,
loss)`. The runner clears/clips gradients separately for every backward pass;
these are not accumulated gradients. All extra passes consume the existing
wall-clock budget.

### Legality changes

Rule 4 now explicitly permits these bounded evaluator-owned passes and batch
reuse. Rule 12 requires one finite differentiable custom loss for each such
pass. Rule 14 now explicitly forbids submission code from invoking
`Tensor.backward`, `torch.autograd.backward`, or `torch.autograd.grad`, and
forbids callbacks from initiating nested model/loss calls, optimizer/scheduler
steps, or hidden training work. The documented intermediate
gradient/parameter/optimizer-state transformation is the only exception.

The prior restrictions remain: random initialization, no hard-coded algorithm
or weights, GPU-only state/computation, no data augmentation, evaluator-owned
outer loop, and 500M model-state ceiling.

## Impact on local work

- No current source is automatically invalidated. A static scan of all local
  `submission.py` files found no use of the newly prohibited derivative entry
  points or hidden-loop callback names.
- Existing validation artifacts are **stale**, because they were produced at
  `8a3c78d`; any future submission must validate at `e32c2f9` or newer.
- The completed Hard Fable v2 job `602bf7f1-eab7-46c2-91e8-e4a4a010f9d7` is
  validly evaluated under the then-live service, but it is not competitive:
  no T=1 rung was certified, and the live leaderboard showed `mof` at #19/19
  with 0.0000% at both T=1 tie-break profiles (read 2026-08-04).
- The new hooks are a possible future experimental variable, not an approved
  architecture change. They must not be combined with the current Task B
  capability ladder.

## Required checks before competition-facing code changes

1. Approve whether to move the local competition checkout to `e32c2f9`.
2. The executor records a single ladder condition and its prediction in
   `solving/experiments/predictions.md` before any training. This is now an
   executor-owned protocol requirement, not a human-confirmation gate.
