# Cheating boundary — what is / is not allowed

Source of truth: competition [README](https://github.com/tilde-research/one-layer-deeper) + [problem page](https://onelayerdeeper.ai/problem) + `submission_validation.py`. This note is our working digest.

## Forbidden (ban / invalid score territory)

- **Math oracles in the forward pass** — φ(N), factoring N, CRT, closed-form `pow(x, 2**T, N)` as the “model”
- **Hard-coded weights** — `torch.load` of a solved checkpoint; answer lookup tables keyed by (N, x, T)
- **Inspecting / shipping the training set** from inside the submission; reading dataset files at eval time
- **Custom training loop / your own backward** — evaluator owns one-forward / one-backward
- **CPU offloading** to dodge the GPU budget
- **Data augmentation** beyond what the evaluator samples
- **Exploiting Hard metric recording**
- **Importing** repo `model` / `optim` / `data` packages or installing extra packages at runtime

## Allowed (and expected)

- Any **architecture** under ≤500M persistent params (shared weights count once)
- **Recurrence / loops / ACT / routing / memory tokens**
- **Optimizer + LR schedule** via `OptimizerBundle`
- **Custom loss** `(logits, labels, aux) → scalar` (evaluator still does backward)
- Lower **batch size** / **max_steps** than the manifest ceiling
- Different depth at `train()` vs `eval()` via `self.training`
- Random init + learning on the evaluator’s stream

## Grey zone — ask before doing

- Heavy **hand-designed digit arithmetic circuits** that are “learned” in name only (often fails Hard if the recurrence changes)
- Encoding **T** as an explicit loop count that assumes “squaring” semantics (Hard may change the task)

**Rule of thumb:** if it would still make sense on a *different* serial recurrence with the same token format, it is probably fine. If it only works because you reimplemented modular squaring, it is cheating.

## Evaluator privilege (not cheating for them)

Labels may use:

```text
φ(N) = (p−1)(q−1)
e = 2ᵀ mod φ(N)
y = xᵉ mod N
```

You never receive p, q. Replicating that inside `submission.py` is the canonical foul.
