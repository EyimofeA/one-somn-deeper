# fable_max_smooth

**CHANGE:** replaced fable_max's discrete `hard_st = prog > 0.55` quantization switch with a continuous `mix = prog/0.7` interpolation between soft and hard-straight-through forward values.

**RESULT:** unclear — the cliff is gone, but so is the early learning, and there's a real confound.

**Detail:** 1,615,882 params, e5, 60s, only 1,154 steps (vs. 1,891 for the original at the same 60s — smooth is ~40% slower per step, computing `p_soft`, `hard`, and the STE blend every loop iteration instead of branching). No collapse — but also no descent at all: loss flat ~2.1-2.2 the entire run, final train ~1% accuracy, test 0.58% / ood 0.50%. Never reaches anywhere near the original's pre-collapse loss 0.29 / 80% train accuracy.

This isn't a clean refutation of the fix. The quantization schedule is **wall-clock**-based (`self.progress`, from `WallclockSchedule`), not step-based, and this variant runs meaningfully slower — so at any given step count it has had more real time elapsed, meaning `mix` is already higher (more "hardened") than the original was at the same step. Comparing step 700-800 directly: original at elapsed 22.5-25.6s was still descending fast (loss 1.46→0.96); smooth at the *same steps* had already used 36.7-41.9s and was flat at ~2.1-2.15. The smooth version may be getting pushed toward hardness before it's had comparable exploration time, not because the fix is wrong but because it's paying a compute tax the schedule doesn't account for.

Next step, not yet done: either make the schedule step-count-aware instead of pure wall-clock for this specific quantization-progress signal, or profile/cheapen the mix computation so the two variants are compute-matched, before concluding anything about whether smoothing the transition actually helps.

Metrics: stdout only. Log: `solving/RESEARCH_LOG.md` (pending).
