# gate1_carry_normalize

**Author:** Codex
**Implementation:** Boyle (`implement_gate1_square`)

**CHANGE:** diagnostic data/target only: aligned local products → exact
base-10 normalization of supplied pre-carry column totals. The plain d=32
four-layer RoPE submission, optimizer, and runtime are byte-frozen from
`gate1_aligned_products`.

Each total is a three-digit block; blocks and output digits are LSD-column
first. Bounds use balanced operand lengths and contributor count ×81:
`c=1..7` gives contributor patterns `[1]`, `[1,1]`, `[1,2,1]`,
`[1,2,2,1]`, `[1,2,3,2,1]`, `[1,2,3,3,2,1]`,
`[1,2,3,4,3,2,1]`; the corresponding bounds are `[81]`, `[81,81]`,
`[81,162,81]`, `[81,162,162,81]`, `[81,162,243,162,81]`,
`[81,162,243,243,162,81]`, and `[81,162,243,324,243,162,81]`.
Targets have fixed length `c+2`, including two flushed carry digits.

Train/test counts by c are `64/16`, `736/184`, then `1440/360` for each
of c=3..7: 8,000 train and 2,000 test rows. Every c occurs in both splits;
prompts are unique and disjoint. Arithmetic exists only in the generator.

**PREDICTION:** agent-authored under the human's explicit supervision override;
recorded in [`../predictions.md`](../predictions.md).

The model ran for 1,000 fixed optimizer steps on the A6000 in 35.7 seconds.
Final train-batch exact match was 29.69%; held-out exact match was 30.35%.
Held-out loss fell continuously from 2.81 to 0.67, and the highest held-out
accuracy occurred at the final evaluation.

**RESULT:** refuted, classified by Codex under the human's override.

**Interpretation (Codex):** The predicted ≥95% was not reached. Unlike exact
squaring, train and held-out improved together without a generalization gap,
and the curve had not plateaued. A bounded step-budget extension is required
before concluding that a shared carry scan is necessary.
