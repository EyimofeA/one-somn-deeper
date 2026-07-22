# fable_max_wd1

**CHANGE:** weight_decay 0.1 → 1.0, everything else unchanged. Monitored (periodic held-out eval) on e5, overtrained to 500s instead of 60s.

**RESULT:** confirmed a real dynamics change, refuted the hope of a better ceiling — closes the loop on today's whole fable_max thread.

**Detail:** 15,000+ steps, e5, 500s. Three phases:

1. **Deeper overfitting than wd=0.1.** Train accuracy hits 84.8% by step 1000 (33s) — much faster than the wd=0.1 original. Test loss peaks at **2.98** around step 1800 — *worse* than wd=0.1's peak of 2.52. Higher wd did not reduce peak overfitting; it let the model overfit longer before the schedule intervened.
2. **Genuine slow generalization phase, steps ~1800-9600.** Test loss improves steadily — 2.9 → 2.6 → 2.5 → 2.3 → 2.27 — while train accuracy stays high (75-95%, not degrading). This is the grokking shape (train stays fit, test slowly catches up over thousands of steps) — not seen anywhere else this session outside the original Hard run, and the first evidence wd actually changes the trajectory meaningfully.
3. **Second collapse at step 9800** (elapsed 325.7s). Train accuracy craters 94%→1%, test loss settles at **2.171** and stays flat for the remaining 5,000+ steps — the best test loss of any fable_max variant today, but by a margin of ~0.01, not a real improvement.

**The collapse is schedule-relative, not step-relative:** progress at step 9800 = 325.7 / (0.92×500) ≈ 0.708 — the same ~0.7 threshold (theta floor, alpha ceiling) that triggered the single collapse in every 60s run today. A longer horizon just buys more steps before the same fractional checkpoint, which is why wd=1.0's overfitting phase ran deeper before hitting it.

**The unifying finding:** every fable_max variant tested today — wd=0.1, wd=1.0, hard_st on/off, theta annealed/frozen, 60s or 500s — converges to test loss ≈2.17-2.18 once past the schedule's ~0.7 threshold. That's not a hyperparameter coincidence; it's a shared ceiling, and it matches the T=1 probe finding (`2026-07-22_t1only_probe_rope`): the block cannot represent one step of modular reduction for held-out N, regardless of training dynamics. Weight decay changes *how* a model gets to the ceiling (fast crude overfit vs. slow grokking-shaped climb) but not whether it can cross it. This is an architecture/mechanism question (RASP-L / inner-loop direction), not an optimizer or schedule one — no further wd/schedule tuning on this family is likely to move the needle.

Metrics: `/tmp/fable_max_wd1_overtrain500.jsonl` on the L40S box, not pulled into the repo. Log: `solving/RESEARCH_LOG.md` (pending).
