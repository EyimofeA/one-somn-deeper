# claude_fire_e1

**CHANGE:** FIRE relative attention bias alone (bidirectional-adapted, signed distance), no RoPE/absolute/depth.

**RESULT:** refuted

**Detail:** d=32, heads=4, layers=4 (untied), 52,390 params, 1251 steps in 60s. Train hit 100% by step 900 (same fast-memorization shape as the RoPE anchor). test 0.67% / ood 3.00% — both worse than `claude_std_rope_e1` (test 2.67% / ood 7.00%) and the worst of the three ablations on test.

Metrics: not saved (stdout only). Log: `solving/RESEARCH_LOG.md` (pending).
