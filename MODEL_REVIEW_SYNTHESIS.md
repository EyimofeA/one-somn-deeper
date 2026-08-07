# Model-review synthesis

## Evidence versus opinion

**Project evidence.** The legal Fable T-cap + AdamW candidate is source-valid but failed current local revalidation: Easy e1 achieved 1.33% test / 5.00% OOD / 3.17% mean in 482 updates, and Medium m1 achieved 0.067% / 0.000% / 0.050% in 8,363 updates; neither certified T=1 (`runs/fable_tcap_adamw_easy_e1/result.json`, `runs/fable_tcap_adamw_medium_m1/result.json`). The final-label tied VDF controls similarly lack held-out depth certification (`review_packets/latent_state_second_opinion_2026-08-05/FAILED_EXPERIMENTS.md`). In controlled diagnostics, a learned Square→Reduce recurrent cell improves substantially when trained on VDF-generated trace states, but remains below exact and is not a legal competition submission (`solving/STATUS.md`).

**Model opinion.** The Claude/Fable material proposes tied token-register iteration, latent-state recurrence, and RNS channels (`claude code fable/FULL_TRANSCRIPT.md`). These are hypotheses, not independent evidence. The GPT Pro packet currently contains a prompt but not a response.

**External evidence.** The packet cites the competition rules and papers under `research_packet_2026-08-05/papers/`; those sources constrain legality and motivate recurrence/generalization, but do not validate an unrun architecture.

**Synthesis inference.** The convergent negative result across final-label token-register variants, including longer Medium training, makes another prompt-tail curriculum or small optimizer tweak low-value. The highest-value distinct test is a clean evolving latent recurrent workspace, because it changes the state-interface hypothesis while preserving legal tied computation and a direct control. This remains an inference pending human review, not a promoted result.

## Decision awaiting synthesis

The next Track C decision is whether to approve the clean latent workspace as the single bounded research run, or to rank the genuinely distinct RNS proposal above it after the missing external chats are imported. Required approval fields: registered hypothesis, frozen prediction, control, success criterion, kill condition, coding budget, and GPU budget. Until then, Track C launches no experiment.

## New Track C result

The first bounded latent-workspace test is complete. On 18 seen and 8 unseen small semiprime moduli, final-label-only global latent training reached unseen-N exact 17.29% at T=1 versus 9.35% for the per-position register control; T=8 was 14.49% versus 14.02%. Both fit seen moduli exactly. This is a narrow state-interface signal, not evidence of scalable VDF solving; the full artifact and curve are `diagnostics/artifacts/clean_latent_workspace_seed0/eval_report.json`. The next decision is whether to scale this representation to a harder held-out-N regime or test the highest-ranked distinct review proposal (RNS), subject to a new registered control and kill gate.
