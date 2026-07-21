# Guiding principle — Karpathy recipe

Source: [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) (2019).

## How we map it here

| Karpathy stage | Our move |
|----------------|----------|
| 1. Become one with the data | `data_samples/`, concepts/01–02 |
| 2. Skeleton + dumb baselines | b0/b1/b2 → `*_max` on Easy e1 |
| 3. Overfit / simple→complex | depth loops → UT depth-emb; then Medium when funnel ready |
| Visualize everything | `figures/` dashboard from JSONL; no silent complexity |
| One change at a time | experiment cards; ablate K or width, not both |

## Where we are (2026-07-21)

- Skeleton + baselines + K-sweep + UT depth-emb: **done**
- Current reference baseline for further work: **`depth_d32_k4_ut`** (best e5) and **`depth_d32_k2_ut`** (best e1)
- **Not yet:** aux losses, diffusion, Medium — Karpathy says earn complexity; train loss still plateaus ~1.7–2.2 with batch exact noisy → diagnose/overfit path before stacking losses
- Scaling-law day (when we grow width/params): plot score vs params / steps under fixed clock — see glossary note on “scaling laws”

## Non-negotiables for us

- Hypothesis before submit
- Plot every returned scalar we can
- Prefer Easy ablations over unverified Medium fancy
- No checkpoint fine-tune (API does not return weights) — “iterate” = new `submission.py`
- Reserve **10 Easy/day** for Claude Code when principal says so (`DAILY_QUOTA.md`)

Leaks: we cannot “overfit one batch” on the hosted runner the way Karpathy means; approximate with smoke_cpu + watching train-batch exact rise under 60s.
