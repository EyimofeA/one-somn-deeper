# Experiment loop

One card per scored run. Append results to `solving/RESEARCH_LOG.md`. Plots → `solving/experiments/figures/`.

## Template

```text
ID:          YYYY-MM-DD-NNN
Hypothesis:  (one sentence)
Arch:        b0 | b1 | b2 | depth_*  (+ knobs: K, lr, …)
Ablation vs: (what changes from last card)
Tier/data:   easy/e1
Submit:      one-layer submit … → job id
Metrics:     mean_exact / test / ood  (facts only)
Plot:        fig_….png
Next:        (next hypothesis)
```

## Phase 1 ladder — Easy e1 (done)

| ID | Hypothesis | Arch | Job | mean | test | ood | steps |
|----|------------|------|-----|------|------|-----|-------|
| 001 | Attention binds N/X/T better than no mixing | b0 Transformer | fddbf10e… | 1.00% | 2.0% | 0% | 261 |
| 002 | Position-wise MLP enough on Easy e1 | b1 MLP×3 | 7843e881… | 0.33% | 0.7% | 0% | (see jsonl) |
| 003 | BiGRU helps serial token structure | b2 BiGRU | 6cd36e0b… | 0.67% | 1.3% | 0% | (see jsonl) |

Same AdamW. Arch diagrams: `learnings/concepts/05-baseline-arches.md`.  
Plots: `figures/fig_baseline_ladder_e1.png`, `figures/fig_baseline_train_curves_e1.png`.
