# fable_max_nohardst

**CHANGE:** ablation — `hard_st` forced permanently `False` during training (p stays pure soft `softmax(dl/theta)` the entire run, never switches to hard straight-through). `theta` anneal unchanged from the original.

**RESULT:** refuted the hard_st hypothesis — collapse still happens.

**Detail:** First run (1,161 steps) was contaminated by GPU contention from a concurrent m5 background job — discarded, don't compare against it. Clean rerun (GPU uncontended): 1,900 steps, matching the original's 1,891 almost exactly, confirming contention was the earlier confound.

Loss descends nicely: 3.13 → 0.72 (peak train accuracy 47.9%) by step 1100-1200 — then **still collapses** at step 1300 (loss 0.72 → 2.15, accuracy → 0.6%), at essentially the same point as the unmodified original (step 1200-1300). Removing the discrete `hard_st` jump entirely did not prevent the collapse. This rules out `hard_st` as the (sole) cause — see `fable_max_notheta` for the theta ablation, and the session note for the resulting alpha hypothesis.

Metrics: stdout only. Log: `solving/RESEARCH_LOG.md` (pending).
