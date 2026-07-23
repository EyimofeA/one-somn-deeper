# Status (living)

Last updated: 2026-07-23. Hard shot #2 `99c4d7d3` scored **0.05%** (test
0.1%/ood_t 0/ood_n_t 0). The current local mechanism work is a bounded,
non-submission carry diagnostic; its results are below.

## Current research state: digit recurrence gates

| Gate | Data and held-out split | Metric | Result | Meaning |
|---|---|---|---|---|
| 0: route tokens | copy X; 1–3 digit train, 4 digits held out | exact sequence | 100% | routing/position is not the block |
| 1a: raw square | decimal X → X² | exact sequence | 7% same-length peak, 0% 4-digit | raw product formation does not generalize |
| 1b: original digit product | held-out 10×10 pairs; four product values unseen | exact table entry | 15% | confounded by unseen decimal outputs |
| 1b′: repaired pair split | held pairs, but every test product seen in train | exact decimal product | Transformer 45% peak / 25% final | pair relation is partly learned, then forgotten through overfit |
| 1b″: bilinear digit cell | same repaired split; fixed ordinal digits + NALU interaction | exact decimal product | 30% peak / 25% final | multiplicative bias does not beat the generic baseline |
| 1b‴: fixed-step schedule | same Transformer and repaired split; step-indexed LR | exact decimal product | 25% peak / 25% final | naive fixed schedule reaches memorization too quickly |
| 1b⁗: slow fixed warmup | same Transformer and split; 400-step warmup | exact decimal product | 40% peak / 25% final | reproduces early signal but does not retain it |
| 1b⁗⁗: pairwise product-and-carry scan | all 1–2 digit operand pairs train; both operands 3 digits held out | exact 6-digit decimal product | 0.25% peak / 0.05% final | fixed schoolbook routing does not identify the local product law |
| 1b⁗⁗⁗: learned pair table | identical length split; only pair MLP → categorical learned table | exact 6-digit decimal product | 1.30% peak / 0.80% final | local categories fit, but compose poorly across length |
| 1b⁗⁗⁗⁗: one-short curriculum | pair table unchanged; train if either operand <100 | exact 6-digit decimal product | 11.25% peak / 11.15% final | nonzero long-column states transfer; unseen high×high remains the gap |
| 1c: carry | 8k train / 2k disjoint test; 1–7 LSD-first three-digit totals | exact normalized output | 98.15% peak at 8k | a shared recurrent state can carry useful state quickly |
| 1d: hard prototype state | same carry data; 64 learned prototypes after every transition | exact normalized output | 0.25% peak | argmax projection prevents learning |
| 1e: soft prototype state | same carry data; soft mixture of the same 64 prototypes | exact normalized output | 98.75% at 8k steps | viable, but slower to optimize than continuous state |

**Current bottleneck:** make a reusable digit-product transition identifiable
under held-out composition. Even correct schoolbook column routing plus a
learned carry scan fits short operands without length generalization. Do not
return to modular squaring or Hard architecture changes until this gate has a
mechanism that generalizes.

**Canonical active code:** [`research/carry_scan.py`](research/carry_scan.py)
and [`research/pairwise_product_carry_scan.py`](research/pairwise_product_carry_scan.py).
The exact A6000 logs are `results_local/gate1_quantized_carry_scan/monitor.jsonl`,
`results_local/gate1_soft_prototype_scan/monitor.jsonl`, and
`results_local/gate1_soft_prototype_8k/monitor.jsonl`, plus
`results_local/gate1_continuous_carry_8k/monitor.jsonl` on `twoA6000`.

## P2 grokking ladder (the active gate — see `claude code fable/FULL_TRANSCRIPT.md`)

| Rung | Setup (all T=1) | Status |
|------|------|--------|
| 1 | fixed single N, unseen x | **N=323: peak 8.6% but decays to ~2% by end of anneal; N=1073: peak 3.5%→1.5%** — real but non-monotone, never consolidates (`2026-07-23_t1only_fixedn_wd01/`) |
| 2 | multi-N seen at train, unseen x | **floor, 0.5-0.75%** (`2026-07-22_t1only_probe_*` — relabeled from "rung 3", see correction) |
| 3 | held-out N (`split_group=modulus`) | not run — strictly harder than rung 2, gate (≥5%) nowhere in sight |

Failure point is **below rung 2**: the one-step map is barely learnable even
per-modulus. No Hard-tier architecture work is informative until rung 1 clears a
real number. wd=1.0 refuted at d=32 (never fits train, `_wd1/`).

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
