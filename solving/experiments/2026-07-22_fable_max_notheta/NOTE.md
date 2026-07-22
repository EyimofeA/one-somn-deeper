# fable_max_notheta

**CHANGE:** ablation — `theta` frozen at 1.0 (no temperature annealing at all). `hard_st`'s discrete `prog>0.55` jump unchanged from the original.

**RESULT:** refuted the theta hypothesis too — collapse still happens, and peak performance is worse.

**Detail:** GPU uncontended (ran right after the clean `nohardst` rerun). 1,861 steps in 60s. Loss descends more weakly than either the original or `nohardst` — never gets below ~1.89 (vs. original's 0.29, `nohardst`'s 0.72) — then **still collapses** at step 1300 (loss 1.99 → 2.15), the same point as both other variants.

Both named suspects (the discrete `hard_st` jump, `theta`'s anneal) are now ruled out individually — the collapse happens with either one disabled, at the same progress point (~0.74-0.76) in all three runs (original, `nohardst`, `notheta`). The one schedule variable common to all three and untested here: `alpha = 0.25 + 0.75*min(1, prog/0.7)`, the register-blend strength, which hits its ceiling (alpha=1.0, full replacement of the register state each loop instead of a blend) at exactly this progress point in every run. Worth testing next.

Metrics: stdout only. Log: `solving/RESEARCH_LOG.md` (pending).
