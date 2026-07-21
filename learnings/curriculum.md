# Curriculum

Ordered learning path for One Somn Deeper.

## Phase 0 — Setup ✓

- [x] Scaffold workspace, clone competition repo
- [x] Run unit tests and CPU smoke on Mac
- [x] Install `one-layer` CLI (login deferred)

## Phase 1 — Baselines

- [x] Study `competition/submissions/baseline_adamw/submission.py`
- [x] Implement b0 / b1 / b2 baselines in `solving/submissions/`
- [x] CPU smoke each baseline locally
- [x] Easy e1 scored submits (needs `one-layer login`)
- [x] Ladder + train-curve plots

## Phase 1b — Figures (after first scored run)

Parent produces script-backed plots in `solving/experiments/figures/` (matplotlib; research-viz skill). Embed PNGs in debriefs so you actually see them.

Priority charts:

1. **Baseline ladder** — exact accuracy: b0 vs b1 vs b2 (Easy e1)
2. **Train curve** — loss / exact accuracy vs step (from metrics JSONL)
3. **Depth slice** — accuracy vs T (e5 or OOD split when available)
4. **Depth–time tradeoff** — steps/sec vs loop count K (Colab timing)
5. **Small multiples** — one panel per dataset e1–e5 once we have multiple submits

Rules: one message per figure, honest axes, colorblind-safe palette for presentation plots.

## Phase 2 — Understand failure modes

- [ ] Read `competition/data/squaring_mod.py` dataset generation
- [ ] Profile exact-accuracy on Easy tier (e1) via Colab
- [ ] Document OOD vs in-distribution gaps in `RESEARCH_LOG.md`

## Phase 3 — Architecture experiments

- [ ] Recurrence vs fixed depth under 500M parameter ceiling
- [ ] Optimizer and schedule sweeps
- [ ] Medium-tier (m1–m5) iteration on Colab

## Phase 4 — Hard preparation

- [ ] User-approved Hard submit only
- [ ] No math oracles, no hard-coded weights

## Concepts

1. [The Problem](concepts/01-the-problem.md)
2. [Where the data is](concepts/02-where-the-data-is.md) — open `solving/experiments/data_samples/` and stare at JSONL
3. [Cheating boundary](concepts/03-cheating-boundary.md)
4. [Arch, bias, optimizer](concepts/04-arch-bias-optimizer.md)
5. [Baseline arches (paper-style)](concepts/05-baseline-arches.md)
6. [Reading metrics](concepts/06-reading-metrics.md)
7. [Limits & artifacts](concepts/07-limits-and-artifacts.md)
8. [Expected loss](concepts/08-expected-loss.md)
9. [What is returned](concepts/09-what-is-returned.md)
10. [Karpathy recipe](concepts/10-karpathy-recipe.md)
11. [Ideas backlog](concepts/11-ideas-backlog.md)
12. [Current arch & naming](concepts/12-current-arch-and-naming.md) — d=width, K=loops; N-cond
13. [Decisions glossary](concepts/13-decisions-glossary.md) — FiLM, ACT/ponder, e1–e5, logging
14. [Discord / beta meta](concepts/14-discord-beta-meta.md) — Hard≠Easy solver, deadline, grey zones
15. [LR schedules under wall-clock](concepts/15-lr-schedules-wallclock.md) — no cosine sawtooth; includes the reconciliation of the two fixes (prefer wall-clock; do NOT apply optsched to Easy)
16. [Representation vs throughput](concepts/16-representation-vs-throughput.md) — place value wins; width/T-coupling/output-place lose; deriving (field, place) from `input_ids`; where the answer actually lives
17. [Exact recurrence learning](concepts/17-recurrence-generalisation.md) — **start here for what's next**; Hard groks train 100% / eval 0%, so the problem is inductive bias, not capacity

## Papers

- [Paper log](papers/README.md) — start with [T²MLR 2607.15178](papers/2607.15178-t2mlr.md)
