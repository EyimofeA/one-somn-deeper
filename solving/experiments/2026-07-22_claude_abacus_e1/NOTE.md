# claude_abacus_e1

**CHANGE:** Abacus place-value embedding alone (end-anchored, MSD-corrected), no RoPE/absolute/depth.

**RESULT:** refuted

**Detail:** d=32, heads=4, layers=4 (untied), 53,184 params, 1381 steps in 60s. Never reached full memorization (train exact-match plateaued ~44.9%, loss ~0.5 — slower to fit than the RoPE anchor, consistent with having only local place-value signal and no global span order). test 1.33% / ood 6.00% — both worse than the `claude_std_rope_e1` anchor (test 2.67% / ood 7.00%).

Metrics: not saved (stdout only). Log: `solving/RESEARCH_LOG.md` (pending).
