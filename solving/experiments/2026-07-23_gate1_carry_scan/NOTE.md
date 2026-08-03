# gate1_carry_scan

**Author:** Codex
**Implementation:** Boyle (`implement_gate1_square`)

**CHANGE:** replace parallel plain-Transformer decoding with one learned,
weight-shared LSD→MSD carry-state scan. Generator, all 8,000/2,000 rows,
4,000-step runtime, optimizer hyperparameters, seed, batches, and vocabulary
are frozen from `gate1_carry_normalize_4k`.

Each three-token total block is embedded without numeric conversion, encoded by
one shared block MLP, and passed through one d=32 GRUCell. The same cell advances
all seven possible columns and performs exactly two learned end-input flush
steps. Its `c+2` emissions are placed at the evaluator's last `c+2` valid prompt
positions. The tied digit/output embedding keeps the model compact at 11,104
state elements.

Motivation: the 4,000-step anchor ended at c1 100.0%, c2 97.83%, c3 95.56%,
c4 89.44%, c5 86.39%, c6 67.78%, and c7 53.89%; see
[`../2026-07-23_gate1_carry_normalize_4k/NOTE.md`](../2026-07-23_gate1_carry_normalize_4k/NOTE.md).

**PREDICTION:** agent-authored under the human's explicit supervision override;
recorded in [`../predictions.md`](../predictions.md).

The model ran for 4,000 fixed optimizer steps on the A6000 in 130.2 seconds.
Final held-out exact match was 79.45%; train-batch exact match was 78.91%.
Held-out loss was 0.201 and the peak occurred at the final step.

**RESULT:** refuted, classified by Codex under the human's override.

**Interpretation (Codex):** The scan improved the longest chains slightly
(c6 71.94%, c7 56.11% versus 67.78% / 53.89% for the Transformer) but lost
accuracy at c3–c5 and did not flatten the curve. Reusing a continuous state is
insufficient; state error still compounds with chain length.
