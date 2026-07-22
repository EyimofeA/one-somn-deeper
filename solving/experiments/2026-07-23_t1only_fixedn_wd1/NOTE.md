# t1only_fixedn_wd1 — rung 1, weight_decay 0.1 → 1.0

**CHANGE:** one variable off `2026-07-23_t1only_fixedn_wd01`: AdamW weight_decay
0.1 → 1.0 (the grokking literature's memorise→generalise knob; L4 from yesterday's
synthesis). Same datasets, same 900s budget, same monitored setup.

**RESULT:** refuted — wd=1.0 over-regularizes at d=32 and never even fits train.

| N | steps | train EM | test EM final | weight norm (vs wd0.1) |
|---|---|---|---|---|
| 323 | 82,000 | 61% | 1.72% | 15.0 (vs 35.2) |
| 1073 | 22,000 | 31% | 0.00% | 15.4 (vs 30.3) |

D1 per-position digit accuracy on the final checkpoints ≈ train-marginal baseline at
every position (N=1073: 0.07/0.14/0.08 vs baselines 0.12/0.12/0.11; N=323:
0.17/0.13/0.45 vs 0.15/0.12/0.51) — priors only, no computation.

**Interpretation (marked as interpretation):** at this width the wd=1.0 penalty
dominates before memorization completes, so the grokking precondition (memorize
first, then simplify) never establishes. If wd is to be pushed on rung 1, it needs
either more width or longer budget — not this exact config.

Metrics: `../metrics/rung1_n{323,1073}_wd1_monitor.jsonl`; checkpoints
`/tmp/rung1_n{323,1073}_wd1_monitor_final.pt` (box only).
