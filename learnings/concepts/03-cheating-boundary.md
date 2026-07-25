# Cheating boundary — what is / is not allowed

Sources: competition [README](https://github.com/tilde-research/one-layer-deeper) @ `79f0a09` + website rules + `submission_validation.py`. Discord beta chat is summarized in [`14-discord-beta-meta.md`](14-discord-beta-meta.md) — useful, not always binding.

Rule numbers below match the **2026-07-24** README (rules 1–16). Older notes that say “Rule 10 = solvers” are stale — solvers / data inspection are now **Rule 14**.

## Forbidden (ban / invalid score territory)

- **Math oracles in the forward pass** — φ(N), factoring N, CRT, closed-form `pow(x, 2**T, N)` as the “model” (Rules 7–8 + 14)
- **Hard-coded weights** — `torch.load` of a solved checkpoint; non-random init that is already a solved circuit (**Rule 6**)
- **Hard-coded algorithm in the forward pass** — outputs must come from the *learned* model (**Rule 7**). Discord examples that organizers treat as cheats: one-hot digit arithmetic FFN lookup tables chained into exact `r² mod N` for T steps; programmatic solvers with a tiny “learned” shim
- **Broken autograd / non-end-to-end learning** — final logits must stay on the autograd graph with an unbroken gradient path from loss to the predicting parameters (**Rule 8**)
- **CPU offloading** — model state and computation must stay on GPU throughout train/eval (**Rule 9**)
- **Inspecting / shipping the training set** from inside the submission; reading dataset files at eval time (**Rule 14**)
- **Custom training loop / your own backward** — evaluator owns one-forward / one-backward (**Rule 14**)
- **Manifest overrides** (**Rule 14**)
- **Exploiting Hard metric recording** (**Rule 16**)
- **Importing** repo `model` / `optim` / `data` packages or installing extra packages at runtime (**Rule 2**)
- **Data augmentation** on Hard (leaderboard) runs — stated in Discord and added to website rules

## Allowed (and expected)

- Any **architecture** under ≤500M model-state ceiling — **trainable params** are capped; persistent buffers and frozen state still count (**Rule 5**)
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

Discord consensus: Easy/Medium can be solved exactly with designed weights / solvers. Public Hard LB tops are tiny after resets; ranking is now **certified Max T**, not mean exact %. Our sandbox optimizes for **learned** depth/composition that might transfer to Hard — not for Easy vanity 100%.

## Evaluator privilege (not cheating for them)

Labels may use φ(N) shortcuts. You never receive p, q. Replicating that inside `submission.py` is the canonical foul.
