# Experiment loop

Open figures via `figures/PLOTS_INDEX.md` (IDE, not chat).

## Limits

Easy 60s / 60 day · Medium 600s / 6 day · Hard 3600s / 1 day. No checkpoint return. No LR/grad logs.

## Phase 1 baselines → max

| ID | Arch | mean | test | ood | steps |
|----|------|------|------|-----|-------|
| 001–003 | v1 Tf/MLP/GRU | 1.0 / 0.3 / 0.7 | … | 0 | ~260 |
| 004–006 | max Tf/MLP/GRU | 1.3 / 1.0 / 1.0 | … | 0 | ~555 |

## Phase 2 — depth (shared block × K on Tf d=64)

| ID | K | mean | test | ood | steps | job |
|----|---|------|------|-----|-------|-----|
| 007 | 4 | **1.83%** | 0.7% | **3.0%** | 489 | 5a88d0fb… |
| 008 | 8 | 1.67% | 1.3% | **2.0%** | 491 | 83d8bdf5… |

First non-zero OOD. Best mean so far: **K=4**.

## Phase 2b — width scaling (K=1)

| d | mean | test | ood | steps |
|---|------|------|-----|-------|
| **32** | **2.70%** | 1.3% | **4.0%** | 539 |
| 64 | 1.30% | 2.7% | 0% | 557 |
| 96 | 1.50% | 2.0% | 1.0% | 541 |
| 128 | 1.80% | 2.7% | 1.0% | 503 |

**Best overall Easy e1 so far: d=32 single-block (2.70% mean).**

## Phase 3 — combo d32 × K=4

| ID | Arch | mean | test | ood | steps | job |
|----|------|------|------|-----|-------|-----|
| 009 | **d32 K=4** | **5.50%** | 2.0% | **9.0%** | 471 | 83e291a5… |

Beats both parents (d32 K1=2.7%, d64 K4=1.8%). Plots: `fig_combo_d32_k4_e1.png`, `fig_combo_d32_k4_train_e1.png`.

## Phase 3b — e5 transfer (same submission)

| ID | Dataset | mean | test | ood | steps | job |
|----|---------|------|------|-----|-------|-----|
| 009 | e1 fixed N=323 | **5.50%** | 2.0% | 9.0% | 471 | 83e291a5… |
| 010 | e5 10–11 bit N | **0.79%** | 1.1% | 0.5% | 2527 | 07ed6ab5… |

Same `depth_d32_k4`. More steps on e5 but much lower score → modulus generalization is the gap.

## Phase 4 — d32 K-sweep

### Easy e1

| K | mean | test | ood | steps |
|---|------|------|-----|-------|
| 1 | 2.70% | 1.3% | 4% | 539 |
| **2** | **6.20%** | 3.3% | 9% | 383 |
| 3 | 5.00% | 2.0% | 8% | 407 |
| 4 | 5.50% | 2.0% | 9% | 471 |
| 6 | 4.50% | 2.0% | 7% | 411 |
| 8 | 2.70% | 3.3% | 2% | 413 |

Peak at **K=2** on e1 (then soft decline).

### e5 gate (top K)

| K | e1 mean | e5 mean | e5 test | e5 ood |
|---|---------|---------|---------|--------|
| 2 | **6.20%** | 0.50% | 0.3% | 0.7% |
| 3 | 5.00% | 0.40% | 0.7% | 0.0% |
| 4 | 5.50% | **0.80%** | 1.1% | 0.5% |

e1 peak ≠ e5 peak. Reference for N-work stays **K=4** on e5; e1 ablations can still cite K=2.

## After each run, read

1. `learnings/concepts/09-what-is-returned.md`
2. matching `metrics/*.jsonl`
3. `figures/PLOTS_INDEX.md`
4. this file + `RESEARCH_LOG.md` + `learnings/concepts/11-ideas-backlog.md`
