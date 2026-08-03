# gate1_square

**Author:** Codex
**Implementation:** Boyle (`implement_gate1_square`)

**CHANGE:** diagnostic target only: `copy_x` → exact decimal `square_x_no_mod`; model and optimizer are byte-identical to `gate0_copy` / `claude_std_rope_e1`.

**PREDICTION (human):** “didgit multiplication is hard iirc for normal
transformers. everything can be learnt well but im sure we arent generalizaing”

Fixed N=10403 and T=1; the x splits are identical to `gate0_copy`: 800 train
and 199 test rows from x=1..999, plus 1,000 unique four-digit OOD x rows.
The local diagnostic target is x² without modular reduction.

The unchanged d=32, four-layer RoPE Transformer ran for 1,000 fixed optimizer
steps on the A6000 in 78.4 seconds. Train exact match reached 98.05%. Held-out
same-length exact match peaked at 7.04% at step 900 and ended at 5.53%.
Four-digit OOD exact match was 0.00% at every evaluation. OOD loss rose from
2.30 at step 100 to 6.72 at step 1,000 while train loss fell to 0.019.

**RESULT (human):** confirmed.

**Interpretation (Codex):** The model can nearly fit the finite square table but
does not learn an exact reusable squaring rule. Gate 1 fails, so the next bounded
diagnostics should separate digit-pair products, place alignment, and carry
propagation before attempting modular reduction or held-out T.
