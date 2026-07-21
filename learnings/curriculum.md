# Curriculum

Ordered learning path for One Somn Deeper.

## Phase 0 — Setup ✓

- [x] Scaffold workspace, clone competition repo
- [x] Run unit tests and CPU smoke on Mac
- [x] Install `one-layer` CLI (login deferred)

## Phase 1 — Baselines

- [ ] Study `competition/submissions/baseline_adamw/submission.py`
- [ ] Implement b0 / b1 / b2 baselines in `solving/submissions/`
- [ ] CPU smoke each baseline locally

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
