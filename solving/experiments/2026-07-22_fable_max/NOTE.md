# fable_max

**CHANGE:** new architecture — T-proportional weight-tied FiLM-conditioned loop over a register (last N-digit-count positions), digit re-quantization each loop, wall-clock schedule shared between LR and quantization sharpness/hardness. Contract-audited before running (ban list, harness interface, scheduler pattern) — clean.

**RESULT:** unclear — real learning signal, then a sharp, localized collapse.

**Detail:** 1,615,882 params (d_model=256, 2 FiLM blocks). Tested locally on e5 (60s), not submitted for real scoring.

Loss: 3.13 → 0.29 by step 1100 (train accuracy → 80.1%) — the fastest, cleanest early learning curve of anything tested today. Then **collapses hard** at step 1200-1300: loss back to ~2.18, accuracy ~0%, and it stays collapsed for the remaining ~20s (final train loss 2.176, test 0.75%, ood 0.33%).

The collapse timing is not noise: elapsed ≈38s at step 1200, horizon = 0.92×60 = 55.2s, so `progress ≈ 0.69` — just past `hard_st = prog > 0.55`, the point where `forward()` switches quantization from soft (`softmax(dl/theta)`) to hard straight-through (one-hot forward, soft gradient). That regime change lines up almost exactly with the collapse. Leading hypothesis: the model found a good soft-quantization solution and the abrupt hard-quantization cutover destroyed it before it could adapt in the remaining budget.

Not a dead end — this is the best pre-collapse learning curve seen today, on the hardest local tier available (e5, fully variable N and T). Worth fixing the transition (smoother/later `hard_st` ramp, or anneal `alpha`/`theta` more conservatively) and rerunning before judging the architecture, since the collapse looks like a schedule artifact, not evidence the design can't work.

**m5 (Medium, 600s), unmodified, run afterward:** same flat pathology as everything else tested on m5 today — 15,726 steps, loss stuck ~2.18-2.24 the entire run, never approaches the pre-collapse dip seen on e5, test 0.07% / ood 0.17%. Confirms the collapse isn't a "needs more time" story — 10x the budget just stays collapsed the whole way, doesn't recover and doesn't get a second chance at the good solution.

**Root cause, resolved (see `fable_max_nohardst`, `fable_max_notheta`, `fable_max_smooth`):** neither the discrete `hard_st` jump nor `theta`'s anneal individually causes the collapse — both ablations still collapse at the same progress point (~0.74-0.76). Leading suspect now: `alpha`, the register-blend strength, which reaches its ceiling (1.0 — full state replacement instead of a blend) at exactly that point in every variant tested. Untested as of this note.

**Not submitted for real scoring** — pending the collapse fix.

Metrics: stdout only. Log: `solving/RESEARCH_LOG.md` (pending).
