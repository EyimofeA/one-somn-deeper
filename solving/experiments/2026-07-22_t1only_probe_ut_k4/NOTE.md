# t1only_probe_ut_k4

**CHANGE:** weight-tied UT K=4 loop instead of 4 untied layers, same T=1-only probe manifest.

**RESULT:** refuted — clean now, after fixing a second bug found along the way.

**Detail:** First attempt was confounded by undertraining (below). Investigating why revealed `depth_d32_k4_ut`'s scheduler was never patched with the wall-clock fix from `15-lr-schedules-wallclock.md` — it still used `t_max = seconds * 8`, calibrated for ~8 steps/s. This card runs ~48.7 steps/s on the L40S, so `CosineAnnealingLR` (uncapped past `T_max≈3800`) cycled every ~7600 steps instead of annealing once — train accuracy oscillated between ~35% and 100% for the entire 24,200-step run instead of converging. Chart: `ut_k4_t1_monitor.jsonl` run, see artifact linked in session.

Patched to the wall-clock scheduler and reran: 13,728 params, 14,500 steps in 300s, train converges cleanly and stays converged (loss 1.5e-4, 100% exact-match, LR properly anneals to ~3e-5). **test 0.50% final, 1.25% max ever seen across the whole run** — same noise band as the flat anchor's 0.75%. Looping does not close the T=1 gap even with full convergence and a correct schedule.

Metrics: `/tmp/ut_k4_t1_monitor.jsonl` (broken) and `/tmp/ut_k4_fixed_monitor.jsonl` (fixed), both on the L40S box, not pulled into the repo. Log: `solving/RESEARCH_LOG.md` (pending).
