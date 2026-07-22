# claude_fireabacus_e1

**CHANGE:** Abacus + FIRE combined, no RoPE/absolute/depth.

**RESULT:** refuted

**Detail:** d=32, heads=4, layers=4 (untied), 54,438 params, 1225 steps in 60s. Train hit 100% by step 900. test 0.67% / ood 2.00% — worse than `claude_std_rope_e1` (test 2.67% / ood 7.00%) and the worst ood of all four cards, despite being the paper's own strongest configuration.

Metrics: not saved (stdout only). Log: `solving/RESEARCH_LOG.md` (pending).
