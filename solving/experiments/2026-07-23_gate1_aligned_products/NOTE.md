# gate1_aligned_products

**Author:** Codex
**Implementation:** Boyle (`implement_gate1_square`)

**CHANGE:** diagnostic mapping/data only: one digit pair product → the
fixed-width two-digit product block for every digit of x, in input order and
without cross-position carry. The model and optimizer are byte-identical to
`gate1_digit_product`.

**PREDICTION:** recorded by the human in [`../predictions.md`](../predictions.md).

The split holds out whole x sequences across every multiplier b. Training has
3,000 rows from 300 sequences of lengths 1–3; same-length test has 600 rows
from 60 disjoint sequences; OOD has 1,000 rows from 100 length-4 sequences.
Every ordered local pair (digit,b) ∈ {0,…,9}² occurs in training. Labels use
two digits per local product, so `x=372,b=4` maps to `122808`. Training is
complete.

The unchanged model ran for 1,000 fixed optimizer steps on the A6000 in 45.0
seconds. Train and held-out same-length exact match both reached 100%. Length-4
OOD exact match peaked at 10.0% at step 1 and ended at 7.3%; OOD loss rose from
2.10 at step 100 to 2.59 at step 1,000, while same-length test loss fell to
0.00016.

**RESULT:** confirmed, classified by Codex at the human's request.

**Interpretation (Codex):** Complete local-product coverage is sufficient at
trained positions but is not reused at the unseen fourth position. The current
failure is positional length generalization before carry propagation.
