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

## After each run, read

1. `learnings/concepts/09-what-is-returned.md`
2. matching `metrics/*.jsonl`
3. `figures/PLOTS_INDEX.md`
4. this file + `RESEARCH_LOG.md`
