# claude_std_rope_e1

**CHANGE:** standard 4-layer (untied) Transformer, plain token embedding, RoPE — no weight-tying, no absolute/depth position embeddings.

**RESULT:** unclear

**Detail:** train 100% / loss 1e-5 by step 900 (1353 total steps in 60s). Held-out test (same T∈{1,2,3}, same N=323, different X) only 2.67% — model memorized the exact training set, did not learn the general digit-mod-323 map even in-distribution. ood (T=4, e1's only unseen-T split) 7.00%, below the 9.94% majority-class baseline (`16-representation-vs-throughput.md`), so no real extrapolation signal either. Below current e1 champion `depth_d32_k2_ut_evalk4` (combined 6.83%, test 4.7 / ood 9.0) on both splits.

Metrics: not saved (stdout only, single run). Log: `solving/RESEARCH_LOG.md` (pending).
