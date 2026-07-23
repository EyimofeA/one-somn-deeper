# Status (living)

Last updated: 2026-07-23 (Gate 1 carry normalization at 4,000 steps: **89.06%
train-batch / 80.55% held-out**, up from 30.35% held-out at step 1,000. The
peak remained at the final step. Prediction **refuted** because it did not reach
95%, but carry normalization demonstrably generalizes).

## P2 grokking ladder (the active gate — see `claude code fable/FULL_TRANSCRIPT.md`)

| Rung | Setup (all T=1) | Status |
|------|------|--------|
| 1 | fixed single N, unseen x | **N=323: peak 8.6% but decays to ~2% by end of anneal; N=1073: peak 3.5%→1.5%** — real but non-monotone, never consolidates (`2026-07-23_t1only_fixedn_wd01/`) |
| 2 | multi-N seen at train, unseen x | **floor, 0.5-0.75%** (`2026-07-22_t1only_probe_*` — relabeled from "rung 3", see correction) |
| 3 | held-out N (`split_group=modulus`) | not run — strictly harder than rung 2, gate (≥5%) nowhere in sight |

Failure point is **below rung 2**: the one-step map is barely learnable even
per-modulus. No Hard-tier architecture work is informative until rung 1 clears a
real number. wd=1.0 refuted at d=32 (never fits train, `_wd1/`).

Digit micro-scan gate: a learned LSD→MSD scan with a discrete 16-state carry
reached 100% train but only 1.49% peak held-out-x at fixed N=1073. It also cost
about 8× in optimizer-step throughput versus the d=32 N=1073 rung-1 probe. Hosted e5
completed 613 steps and scored 0.50% on both test and held-out-T OOD.

## Hard leaderboard

We are **#11 at 0.03%** (`mof` / Claude Hard run). Top is **0.40%** (az). Nobody has solved the task.

Protocol: [`../RESEARCH_PROTOCOL.md`](../RESEARCH_PROTOCOL.md).  
Lecture: [`../learnings/readings/one-layer-deeper-notes.md`](../learnings/readings/one-layer-deeper-notes.md).  
Path D short form: [`../learnings/concepts/18-lipschitz-quantize-progressive.md`](../learnings/concepts/18-lipschitz-quantize-progressive.md).
Scientific reset: [`../learnings/concepts/19-scientific-gates.md`](../learnings/concepts/19-scientific-gates.md) (Author: Codex).

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

## Next — carry primitive (agent proceeds under human override)

Write PREDICT in [`experiments/predictions.md`](experiments/predictions.md) before any run.

1. c1–c3 reached 100.0% / 97.8% / 95.6%; c6–c7 fell to 67.8% / 53.9%
2. **Next:** replace parallel carry decoding with a shared LSD→MSD scan
3. Hold data, optimizer, parameter scale, and 4,000-step budget fixed

## Ops

[`experiments/OPS.md`](experiments/OPS.md) · [`experiments/LAYOUT.md`](experiments/LAYOUT.md) · [`RESEARCH_LOG.md`](RESEARCH_LOG.md)
