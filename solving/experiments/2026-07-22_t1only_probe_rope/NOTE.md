# t1only_probe_rope

**CHANGE:** none (unmodified `claude_std_rope_e1` anchor) — isolates whether a single flat block can compute one step of `u² mod N` for held-out (u, N), separate from composition and from embedding choice.

**RESULT:** confirmed — LOW branch. 0.75% test exact-match at T=1, well under the 90% threshold.

**Detail:** 51,136 params (d=32, heads=4, layers=4, RoPE — same anchor as every e1/e3/e5 card today). 2826 steps in 60s. Train hit 100% by step 1700, loss 7e-6 by the end — full memorization of the 1600-row train set. Held-out test (same T=1, different (u,N) pairs): 0.75%. The block cannot represent one step of modular squaring for unseen N even in the easiest possible version of this task (no iteration, no T-extrapolation, field identity unambiguous since every row is T=1). This rules out composition and embedding choice as the bottleneck for today's four position-encoding cards — none of them could have scored higher regardless of scheme, since the representational gap is upstream of position encoding entirely.

Metrics: not saved (stdout only). Log: `solving/RESEARCH_LOG.md` (pending).
