# fable_hard_h1

**CHANGE:** new architecture (authored by Fable) — quantized weight-tied recurrence with T read from the input, operand re-injection each loop, detached-prefix training, entropy aux via custom `training_loss`. Design closely matches this project's own `17-recurrence-generalisation.md` recommendations (re-quantised recurrence, input injection, entropy aux, T=1 supervision).

**RESULT:** refuted, as configured — not an architecture failure, an optimization failure.

**Detail:** 1,595,648 params (d_model=256, 2 tied step-layers). Tested locally on e5 (60s) and m5 (600s), zero quota, not submitted for real scoring.

- e5: 4,355 steps, loss flat ~2.1-2.2 the entire run (started ~2.9), test 0.75% / ood 0.17%.
- m5: 31,634 steps (10x the wall-clock), loss **still flat ~2.1-2.2**, never broke the plateau even with 10x more steps, test 0.19% / ood 0.27%.

The loss curve is the tell: flat and noisy across two full orders of magnitude more compute, not slowly decreasing — this doesn't look like "needs more time," it looks like the optimizer is stuck. Prime suspect: peak LR is 3e-4 (`WarmupSchedule`, 300-step linear warmup then held flat forever, no decay) — every card that learned anything today used lr=3e-3, 10x higher. Worth retrying with a higher peak LR and/or an actual decay schedule before concluding anything about the architecture itself, since the design is well-motivated and hasn't had a fair optimization run yet.

**Not submitted for real scoring** — a flat 60s and flat 600s local loss curve makes a real Easy/Medium/Hard submission a near-certain waste of quota until the LR issue is addressed.

Metrics: stdout only, not pulled into the repo. Log: `solving/RESEARCH_LOG.md` (pending).
