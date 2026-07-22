# t1only_probe_ut_k8

**CHANGE:** weight-tied UT K=8 loop instead of 4 untied layers, same T=1-only probe manifest.

**RESULT:** refuted — same schedule bug as K=4, same fix, same clean outcome.

**Detail:** Original run confounded by the same `t_max = seconds * 8` scheduler bug documented in `t1only_probe_ut_k4`. Patched to the wall-clock scheduler and reran: 13,856 params, 21,750 steps in 500s, train converges cleanly (loss 1.7e-5, 100% exact-match, LR anneals to ~3e-5). **test 0.75% final, 1.50% max ever seen** — same noise band as flat (0.75%) and K=4-fixed (0.50%). More rounds (K=8 vs K=4) does not help either.

Metrics: `/tmp/ut_k8_fixed_monitor.jsonl` on the L40S box, not pulled into the repo. Log: `solving/RESEARCH_LOG.md` (pending).
