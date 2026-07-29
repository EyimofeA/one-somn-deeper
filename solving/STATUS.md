# Status (living)

Last updated: 2026-07-24. **Upstream competition pin moved `2c56499` → `79f0a09`:**
Hard now ranks by certified Max T then OOD N Max T (ladder 1…64); Rules renumbered
with explicit bans on hard-coded weights/algorithms, broken autograd, and CPU
offload; submissions close Aug 31 10pm PT. Specs refreshed under
`solving/handoff/PRIMARY_SOURCES.md` + `learnings/concepts/{01,03,07,09,14}`.

Hard shot #2 `99c4d7d3` scored **0.05%** mean exact (test 0.1%/ood_t 0/ood_n_t 0)
under the *old* Hard metric — not the current Max T rank key. The current local
mechanism work is a bounded, non-submission carry diagnostic; its results are below.

## Task B direct reduction — completed parallel-Transformer branch (2026-07-30)

| Setting | Final held-out-u exact | Reading |
|---|---:|---|
| fixed N=1349, baseline (3 seeds) | **33.65±2.95%** | unseen-u reduction remains unsolved despite 98.03% train exact |
| two N={1349,1357}, baseline (3 seeds) | **11.27±0.63%** | adding a second N collapses generalization |
| two N, correct N broadcast | **11.55±0.30%** | no material gain vs baseline |
| two N, shuffled broadcast control | **11.35±0.39%** | confirms no semantic N-routing effect |
| fixed N, quotient auxiliary | **29.38±6.91%** | refuted vs baseline |
| fixed N, u-copy auxiliary control | **22.35±0.40%** | auxiliary head/extra loss is harmful |

Counterfactual two-N evaluation: baseline changes predictions with N on 92.55% of pairs, but responds incorrectly under both N on 92.38%; correct broadcast does not repair this. The parallel standard-Transformer branch is **falsified for Task B**. Do not scale N broadcast or quotient auxiliaries.

**Serial branch:** completed and refuted for the tested formulation. Parameter-matched K=8 learned workspace reaches 29.23±5.84% peak / 23.47±7.08% final held-out-u EM, below baseline (34.75±2.35 / 33.65±2.95) and a five-layer non-recurrent control (38.23±4.67 / 35.65±5.56). More K helps within the recurrent model but it never clears the control; do **not** K-sweep. The next justified Task-B test is input-conditioned versus shuffled-context workspace initialization at K=8, no auxiliary labels. Evidence: `diagnostics/analysis_out/task_b_n_broadcast_ablation.md`, `diagnostics/analysis_out/task_b_fixed_n_quotient_aux_ablation.md`, `diagnostics/analysis_out/task_b_serial_workspace_ablation.md`.

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
| 1b⁗⁗⁗⁗⁗: full-position baseline | pair table unchanged; 190k random 3-digit pairs, held complete pairs | exact 6-digit decimal product | 58.55% final/peak | learned multiplication composition exists once all position interactions appear |
| 1b⁗⁗⁗⁗⁗⁗: extended horizon | same full-position card; 8k/400 → 31k/1.6k effective steps | exact 6-digit decimal product | 94.85% peak / 94.5% last eval | prior residual was primarily optimization-horizon limited |
| 1b⁗⁗⁗⁗⁗⁗⁗: fan-in tag | same extended card; add learned vector for fixed `[1,2,3,2,1]` column counts | exact 6-digit decimal product | 84.75% peak | explicit count metadata harms the learned column map |
| 1b⁗⁗⁗⁗⁗⁗⁗⁗: intra-column fold | same extended card; sum → shared pair-GRU fold before carry scan | exact 6-digit decimal product | 97.60% peak / 97.4% final | preserves central pair interactions; clears local product gate |
| 2a: soft-digit recurrence | train bases 0..9 at T=1,2; all bases held at T=3 | exact four-LSD digit state | 40.0% (bases 0..3 only) | soft states fit two applications but drift under a third; narrow 100% was not sufficient evidence |
| 2b: STE state control | 2a data/test; soft state → STE one-hot state | exact four-LSD digit state | 40.0% peak through 1.2k steps | discretization does not repair failure; the missing transition is four-digit squaring itself |
| 2c: one-step 4-digit square | 8k shuffled x train / 2k held x; T=1 | exact four-LSD digit state | 85.35% peak at 3.5k | local 4-digit transition generalizes strongly but is not exact enough to compose |
| 2d: arity fold init | 2c split; fold start state keyed by term count | exact four-LSD digit state | 84.55% peak | earlier learning but lower final law; arity is already inferable from fold length |
| 2e: balanced fold tree | 2c split; sequential fold → binary tree | exact four-LSD digit state | 61.7% at 2k (stopped) | serial term order is valuable; short tree paths hurt learning |
| 2f: soft carry prototypes | 2c split; continuous carry → 64 soft prototypes | exact four-LSD digit state | 26.4% at 1.5k (stopped) | finite bottleneck blocks joint product/carry representation |
| 2g: unweighted aux columns | 2c test; train also predicts later square columns | exact four-LSD digit state | 0% by 1k (stopped) | equal-weight extra labels overwhelm the primary objective |
| 2h: weighted aux columns | 2g labels; auxiliary loss weight 0.25 | exact four-LSD digit state | 64.8% at 2k (stopped) | stable but substantially worse; extra columns divert capacity |
| 2i: digit-3 weighted loss | 2c data/model; loss weight `[1,1,1,4]` | exact four-LSD digit state | 74.55% at 2k (stopped) | residual is structural, not insufficient direct gradient |
| 2j: symmetric pair table | 2c split; table constrained `T=Tᵀ` | exact four-LSD digit state | 50.55% at 2k (stopped) | serial fold benefits from orientation-specific features |
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

## Squaring/reduction isolation (2026-07-24, separate from the ladder below)

Composed `learned_reduction_cell` (squaring+reduction, final-remainder-only
supervision) peaked 5.17% then decayed to 1.72% floor — inconclusive on which
stage fails. Isolated each stage separately, real-token format, N=323 fixed:

| Test | Data | Result | Meaning |
|---|---|---|---|
| pure squaring | x uniform 0-9999, held-out x, label=plain x² (no mod) | **18.65% peak / 18.25% final, stable** | full 8-digit squaring ceiling (not 97.8% — that was mod-10⁴ truncated); mechanism generalizes, small gap |
| pure reduction v1 | P uniform 8-digit, held-out P, label=P mod 323, direct supervision | **0.60% peak**, below 1.72% floor | reduction alone, uniform P, does not generalize at all |
| pure reduction v2 | P via reciprocal/log-uniform sampling (arXiv 2506.23679 A.1), wd=1.0, 80k steps | **78.45% peak; final window 69-84%, no decay through 80k steps** | reduction generalizes for real once P's distribution skews small (matching what x² actually produces relative to N) and given grokking-scale wd/budget; confound-checked (95.4% on trivial P<N, 75.5% on genuine P>=N) |

**Reading:** modular reduction is not inherently unlearnable — it was unlearnable
under uniform-P sampling. The composed cell's poor result and pure_reduction v1's
near-zero result likely share this cause. Next step if this thread continues:
retry the composed `learned_reduction_cell` test but check whether P (the
squaring cell's raw output) is already reciprocal-shaped in practice, or feed it
reciprocal-sampled P directly into the composed pipeline's reduction half.
Full detail: `experiments/predictions.md` 2026-07-24 (a)-(d).

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

Full sequential competition runs: [`experiments/SCOREBOARD.md`](experiments/SCOREBOARD.md) (also live canvas).

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

**Read first:** [`DESIGN_NEXT.md`](DESIGN_NEXT.md) — the current architecture thesis
(form vs. content; learn the operation + reduction instead of hardcoding squaring;
the true gate is one-step held-out-N).

Write PREDICT in [`experiments/predictions.md`](experiments/predictions.md) before any run.

1. Measure μ+λ on local data (no GPU) — gates Path G bonus  
2. Count digits of N on h1/m5 — gates Path E  
3. Progressive loss (one-change card)  
4. STE quantize between steps  
5. Input inject each loop  

## Ops

[`experiments/OPS.md`](experiments/OPS.md) · [`experiments/LAYOUT.md`](experiments/LAYOUT.md) · [`RESEARCH_LOG.md`](RESEARCH_LOG.md)
