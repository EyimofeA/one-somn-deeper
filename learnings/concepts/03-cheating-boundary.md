# Cheating boundary — what is / is not allowed

Sources: competition [README](https://github.com/tilde-research/one-layer-deeper) + website rules + `submission_validation.py`. Discord beta chat is summarized in [`14-discord-beta-meta.md`](14-discord-beta-meta.md) — useful, not always binding.

## Forbidden (ban / invalid score territory)

- **Math oracles in the forward pass** — φ(N), factoring N, CRT, closed-form `pow(x, 2**T, N)` as the “model”
- **Hard-coded weights** — `torch.load` of a solved checkpoint; non-random init that is already a solved circuit
- **Hard-coded algorithm in the forward pass** — README rule 03: outputs must come from the *learned* model. Discord examples that organizers treat as cheats: one-hot digit arithmetic FFN lookup tables chained into exact `r² mod N` for T steps; programmatic solvers with a tiny “learned” shim
- **Inspecting / shipping the training set** from inside the submission; reading dataset files at eval time
- **Custom training loop / your own backward** — evaluator owns one-forward / one-backward
- **CPU offloading** to dodge the GPU budget
- **Data augmentation** on Hard (leaderboard) runs — stated in Discord and added to website rules
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

## Grey zone — ask / re-check before Hard

- **Hand-designed digit arithmetic circuits** that train only in name (fails Hard if recurrence is mismatched; Easy/Medium can still hit 100%)
- Encoding **T** as an explicit loop count that assumes Easy “squaring” semantics
- **Aux loss on intermediate hidden states** to match algorithmic steps — Discord: allowed under *current beta* rules per mcleish7; may change
- Forward **branching on T** without computing y — asked in Discord; no clear ruling in our paste

**Rule of thumb (az / organizers’ direction):** if it only works because you reimplemented the Easy/Medium recurrence, it is a solver. Hard is a **slight secret variant** of that recurrence so those solvers break. Prefer architectures that could still train if the serial step changed.

## Easy 100% ≠ research win

Discord consensus: Easy/Medium can be solved exactly with designed weights / solvers. Public Hard LB tops are tiny exact-% after resets. Our sandbox optimizes for **learned** depth/composition that might transfer to Hard — not for Easy vanity 100%.

## Evaluator privilege (not cheating for them)

Labels may use φ(N) shortcuts. You never receive p, q. Replicating that inside `submission.py` is the canonical foul.
