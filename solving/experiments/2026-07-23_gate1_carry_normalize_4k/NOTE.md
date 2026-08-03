# gate1_carry_normalize_4k

**Author:** Codex
**Implementation:** Boyle (`implement_gate1_square`)

**CHANGE:** runtime `max_steps` only: 1,000 → 4,000. Submission, generator,
all 8,000/2,000 data rows, optimizer, batch sizes, seed, evaluation/log
interval, and 600-second safety ceiling are frozen from
`gate1_carry_normalize`.

The manifest retains the same `data/generated/gate1_carry_normalize` root.
This card tests whether the still-rising carry-normalization learning curve
converges under a larger fixed step budget.

**PREDICTION:** agent-authored under the human's explicit supervision override;
recorded in [`../predictions.md`](../predictions.md).

The model ran for 4,000 fixed optimizer steps on the A6000 in 137.1 seconds.
Held-out exact match rose from 30.35% at step 1,000 to 80.55% at step 4,000;
the peak occurred at the final evaluation. Final train-batch exact match was
89.06%, held-out loss was 0.103, and train/test remained coupled.

**RESULT:** refuted, classified by Codex under the human's override.

**Interpretation (Codex):** Carry normalization clearly generalizes to unseen
sequences, but did not reach the predicted ≥95% or the exactness required for
composition. Final exact match by column count was c1 100.0%, c2 97.83%, c3
95.56%, c4 89.44%, c5 86.39%, c6 67.78%, and c7 53.89%. Error compounds with
carry-chain length, supporting a shared LSD→MSD scan rather than another generic
budget extension.
