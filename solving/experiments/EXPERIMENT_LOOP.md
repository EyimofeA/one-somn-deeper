# Experiment scoreboard

Canonical living status: **[`../STATUS.md`](../STATUS.md)**.  
Layout / commits: [`LAYOUT.md`](LAYOUT.md).  
Predictions: [`predictions.md`](predictions.md).  
Ops: [`OPS.md`](OPS.md).

## Parent Easy ladder (facts only)

| Phase | Best mean | Card |
|-------|-----------|------|
| b0/b1/b2 | 1.00% | `b0_transformer` |
| maxed small | 1.33% | `b0_transformer_max` |
| d32 K1 | 2.70% | `scale_tf_d32` |
| d32 K4 | 5.50% | `depth_d32_k4` |
| d32 K2 | 6.20% | `depth_d32_k2` |
| UT K2 | 6.50% | `depth_d32_k2_ut` |
| UT K2→eval4 | **6.80%** | `depth_d32_k2_ut_evalk4` |
| UT K4 e5 | **1.00%** | `depth_d32_k4_ut` |

Rejected: N-FiLM, soft-ACT, midloop, short-T_max cosine on Medium.

## Plots

[`figures/PLOTS_INDEX.md`](figures/PLOTS_INDEX.md) — open PNGs in IDE.
