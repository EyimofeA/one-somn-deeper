# ASSUMPTIONS — `2026-07-22_hard_sample_v0`

Sample Hard-oriented card. Not scored. Not a claim that this beats the board.

## FACT (from primary packet / README / problem page)

| ID | Fact |
|----|------|
| F1 | Hard may change aspects of the recurrence; do not assume repeated squaring. |
| F2 | Official reply: people have guessed a “slightly new family”; some guesses worked, some did not. Exact family unconfirmed. |
| F3 | Submission = one `submission.py`, `SUBMISSION` contract, ≤500M persistent state, ≤256 KiB. |
| F4 | Evaluator owns loop; `forward` → `(logits, auxiliary)`; optional `training_loss`. |
| F5 | Hard: 3600s train, 1800s eval, 1 accepted attempt/UTC day, H100. |
| F6 | Easy/Medium public practice uses digit prompts with markers N/X/T (TOKEN_IDS). |
| F7 | Rule 10 bans task-specific solvers / dataset inspection / custom train loops. |

## ASSUMPTION (ours — may be wrong)

| ID | Assumption | Rests on |
|----|------------|----------|
| A1 | Hard still uses a **serial composition** of one unknown step on digit-like inputs with a depth/count field. | F1–F2, problem framing |
| A2 | Prompt still exposes field markers compatible with ids 2/3/4 and digit ids 7–16 (same public TOKEN_IDS). If Hard changes vocab, field/place helpers degrade gracefully but may be useless. | F6; **unconfirmed for Hard** |
| A3 | A **weight-tied block applied K times** is more transferable across step families than a hardwired square/cube circuit. | F1–F2, F7 |
| A4 | Place-within-number + field embeddings help digit algorithms for **nearby** modular maps (square, cube, affine-ish), not only Easy squaring. | A1–A2 |
| A5 | Re-adding the embedded prompt each loop (input injection) reduces drift of “which problem” across depth. | UT/DEQ practice; not Hard-proven |
| A6 | d=128, K=8 is a compromise: more depth than Easy d=32 K=4, far below the d=2048 Hard run that hit train EM 100% / eval 0% in our log. | local log fact; A3 |
| A7 | Clamped cosine over ~120×train_seconds avoids LR restart under long Hard clocks. | Medium schedule bug in our metrics history |
| A8 | Higher weight decay (1.0) and 0.5× token-emb init scale bias away from pure memorization under 3600s. | untested on Hard |
| A9 | Fixed K (not loops=T) avoids coupling depth to a possibly redefined T semantics. | F1 |

## Explicit non-goals

- No φ(N), no `%` on task values, no sympy, no guessed closed form for cube/affine.
- No claim about leaderboard rank.
- Do not `one-layer submit --tier hard` without principal approval (1/day).

## How to validate (not submit)

```bash
one-layer validate solving/experiments/2026-07-22_hard_sample_v0/submission.py
# optional local smoke if competition/ clone present:
# python -m benchmark.runner --manifest benchmark/manifests/smoke_cpu.json \
#   --submission-file ../solving/experiments/2026-07-22_hard_sample_v0/submission.py
```
