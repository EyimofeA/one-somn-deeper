# T=1 phase square/reduce information-flow test

Research-only final-label diagnostic. The two arms have identical parameters,
depth, optimizer, data, and runtime. The sole intervention is whether `N` is
visible during the four-step square phase; both arms see `N` during the
four-step reduction phase. No arithmetic trace, quotient, residue, or
intermediate target enters the model or loss.

Prediction and result are registered in
[`../predictions.md`](../predictions.md). Raw reports live under the ignored
`diagnostics/artifacts/t1_phase_square_reduce_2026-08-08/` directory.

## Result

Refuted across three matched seeds. Both arms reached 100% train exact.

| Arm | Held-out-x exact (seeds 0/1/2) | Median | Unseen-N exact (seeds 0/1/2) | Median |
|---|---:|---:|---:|---:|
| Factored | 11.76 / 12.18 / 11.34% | 11.76% | 17.76 / 17.06 / 17.29% | 17.29% |
| Entangled | 13.03 / 11.76 / 14.71% | 13.03% | 17.99 / 16.82 / 17.29% | 17.29% |

Hiding `N` in the square phase gives zero median unseen-`N` improvement and
slightly worse held-out-`x`. Phase-level information restriction is therefore
not sufficient to identify an arithmetic square representation in this
unconstrained latent cell.
