# fable_hard_h1_adamw

**CHANGE:** replaced fable_hard_h1's flat-lr=3e-4 WarmupSchedule with our validated AdamW (lr=3e-3, wd=0.1) + wall-clock warmup+cosine schedule.

**RESULT:** confirmed the diagnosis — it was the optimizer, not the architecture.

**Detail:** 1,595,648 params, e5, 60s, 4,311 steps. Loss actually moves now: 2.9 → 1.82 by the end (train accuracy climbing to 7%, vs. the original's flat ~1-2% ceiling for the entire run). Still far from converged in 60s — this architecture clearly needs more optimization budget than the flat single-layer/looped cards tested earlier today. test 0.58% / ood 0.33%. Confirms the original `fable_hard_h1`'s flat loss curve was a stuck optimizer, not a representational dead end — see `fable_hard_h1_muon` for a much stronger fix.

Metrics: stdout only. Log: `solving/RESEARCH_LOG.md` (pending).
