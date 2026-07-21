# Experiment loop

One card per scored run. Metrics in `metrics/`. Plots in `figures/` (open PNGs in the IDE if chat does not embed).

## Template

```text
ID / Hypothesis / Arch / Ablation vs / Tier / Job / mean·test·ood / steps / L0→Lend / Next
```

## Limits (quick)

Easy 60s · 60/day · Medium 600s · 6/day · Hard 3600s · 1/day. No checkpoint return. `compile` off on Easy manifests.

## Phase 1 — v1 baselines (d=128, AdamW flat, bs=512)

| ID | Arch | mean | test | ood | steps | L0→Lend |
|----|------|------|------|-----|-------|---------|
| 001 | b0 Transformer | 1.00% | 2.0% | 0% | 261 | 83.8→1.63 |
| 002 | b1 MLP×3 | 0.30% | 0.7% | 0% | 287 | 80.4→1.87 |
| 003 | b2 BiGRU | 0.70% | 1.3% | 0% | 258 | 21.6→1.21 |

## Phase 1b — maxed small (d=64, AdamW 3e-3 + warmup/cosine, bs=256)

| ID | Arch | mean | test | ood | steps | L0→Lend |
|----|------|------|------|-----|-------|---------|
| 004 | b0_transformer_max | **1.33%** | 2.7% | 0% | **557** | 43.5→1.64 |
| 005 | b1_mlp_max | **1.00%** | 2.0% | 0% | **585** | 44.7→1.85 |
| 006 | b2_rnn_max | **1.00%** | 2.0% | 0% | **555** | 13.9→1.29 |

Jobs: 004 `d1a89a12…` · 005 `20b6faf8…` · 006 `e13aab41…`

**Reference baselines for later ablations:** use `*_max` (more steps, better scores). OOD still 0% → next axis = depth/loops on Transformer_max.
