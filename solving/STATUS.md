# Status (living)

Last updated: 2026-07-22 (submissions migrated to dated experiment dirs).

## Hard leaderboard

We are **#11 at 0.03%** (`mof` / Claude Hard run). Top is **0.40%** (az). Nobody has solved the task.

Protocol: [`../RESEARCH_PROTOCOL.md`](../RESEARCH_PROTOCOL.md).  
Lecture: [`../learnings/readings/one-layer-deeper-notes.md`](../learnings/readings/one-layer-deeper-notes.md).  
Path D short form: [`../learnings/concepts/18-lipschitz-quantize-progressive.md`](../learnings/concepts/18-lipschitz-quantize-progressive.md).

## Best scored cards (learned line)

| Axis | Card | Score | Note |
|------|------|-------|------|
| Easy e1 | `depth_d32_k2_ut_evalk4` | **6.80%** (n=3, σ≈0) | **Invalid ranking signal** |
| Easy e5 | `depth_d32_k4_ut` | **1.00%** | Prefer over e1 |
| Medium m5 | `depth_d32_k4_ut_optsched` | **~0.20%** | Schedule-safe |
| Hard H1 | `claude_hard_h1` | **0.03%** | Train 100% / eval 0% |

## Active submissions

Symlinks → `experiments/2026-07-21_<name>/`. Full history: all `2026-07-21_*` dirs.

| Path | Role |
|------|------|
| `depth_d32_k4_ut_optsched/` | Medium/Hard schedule-safe UT K4 |
| `depth_d32_k2_ut_evalk4/` | Easy e1 peak (weak gate) |
| `depth_d32_k4_ut/` | Easy e5 peak |
| `claude_pv_k4_ut/` | Place-value UT |
| `claude_hard_h1/` | Hard artifact — do not widen |

## Next (Part 8 — you pick; agent implements)

Write PREDICT in [`experiments/predictions.md`](experiments/predictions.md) before any run.

1. Measure μ+λ on local data (no GPU) — gates Path G bonus  
2. Count digits of N on h1/m5 — gates Path E  
3. Progressive loss (one-change card)  
4. STE quantize between steps  
5. Input inject each loop  

## Ops

[`experiments/OPS.md`](experiments/OPS.md) · [`experiments/LAYOUT.md`](experiments/LAYOUT.md) · [`RESEARCH_LOG.md`](RESEARCH_LOG.md)
