# Model-review disagreements

These are disagreements among proposals/opinions, not resolved findings.

1. **State interface:** Claude/Fable favors a token-register workspace with tied recurrence; the latent-state review question argues the central failure may be answer-aligned token positions and proposes `h(t+1)=F(h(t),N)` with a genuine latent workspace. Project evidence currently favors the criticism: final-label tied VDF and depth curricula fit training but have no held-out depth certificate (`review_packets/latent_state_second_opinion_2026-08-05/FAILED_EXPERIMENTS.md`).
2. **What recurrence should learn:** one proposal treats exactness as a reusable one-step iterator; the serial diagnostic branch decomposes Square, Compare, and Subtract. The latter has strong controlled evidence but is not yet competition-legal end-to-end at public scale (`solving/STATUS.md`).
3. **Representation:** the Claude packet proposes digit registers and optionally RNS channels; the current research direction prioritizes a continuous latent workspace. RNS remains an untested opinion, not evidence.
4. **Hard task assumptions:** the Claude packet warns Hard may alter the recurrence; no external review can resolve the hidden rule. Do not design a Hard-specific guessed solver.
5. **Optimization versus identifiability:** Fable-style suggestions emphasize optimizer, entropy, and loop scheduling; the project controls show more updates or curriculum do not repair OOD transfer (`solving/RESEARCH_LOG.md`). The present evidence weights state/transition identifiability above small optimizer changes.

## Resolution policy

No disagreement launches an experiment by itself. A human-selected review synthesis must rank the alternatives and approve one central-hypothesis test under Track C's budget and kill condition.
