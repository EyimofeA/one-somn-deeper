# 2026-07-29 Easy e5 Submissions

## Scoreboard

| # | Submission | e5 Score | vs Baseline (1.00%) |
|---|-----------|----------|---------------------|
| — | UT K4 baseline | 1.00% | reference |
| 1 | STE + input injection | 0.54% | input injection hurts |
| 2 | STE + injection + carry aux + wd=0.3 + progressive | 0.92% | near baseline |
| 3 | UT K4 + carry aux + wd=0.3 (no STE) | 0.25% | wd=0.3 breaks plain UT |

## Score Chart

![scores](solving/experiments/figures/2026-07-29_e5_scores.png)

## Loss Curves

![loss](solving/experiments/figures/2026-07-29_loss_curves.png)

## Key finding

- **STE discrete bottleneck is necessary for higher weight decay** — without it, wd=0.3 kills learning
- **Input injection is harmful** — the model gets distracted by seeing the input again
- **Carry auxiliary with the current proxy (digit % 10) is too weak** — doesn't teach real computation
- **Best near-baseline: STE + wd=0.3 + progressive loss** — next card should isolate STE alone

## Next card

Plain STE bottleneck, wd=0.2, no injection, no aux. Tests whether STE alone beats 1.00% baseline.