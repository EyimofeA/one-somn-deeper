# T=1 pair-fold square tape → learned reduction

Research-only, final-label-only test replacing the failed generic square phase
with the strongest structural lesson from the multiplication gates: learned
digit-pair categories, fixed pair-to-column routing, a shared within-column
fold, and an LSD-first learned carry scan. A learned `N`-conditioned local
reduction cell receives that tape. Pair products, carries, raw squares,
quotients, residues, and traces are never computed for the model or loss.

Prediction and result are registered in
[`../predictions.md`](../predictions.md). Raw reports live under the ignored
`diagnostics/artifacts/t1_pairfold_square_reduce_2026-08-08/` directory.

## Result

Refuted across three seeds; every run reached 100% train exact.

| Split | Seed 0 | Seed 1 | Seed 2 | Median |
|---|---:|---:|---:|---:|
| Held-out x | 10.50% | 10.50% | 11.34% | 10.50% |
| Unseen N | 16.82% | 15.89% | 16.36% | 16.36% |

This is worse than the generic factored tape's 11.76% / 17.29% medians.
Pair routing does not recover the strong directly supervised raw-square
mechanism when credit arrives only through the final modular label.
