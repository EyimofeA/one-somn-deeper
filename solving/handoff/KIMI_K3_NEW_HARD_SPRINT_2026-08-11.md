# Kimi K3 new-source sprint

You are the independent competition implementer for One Layer Deeper. Current
time is approximately 23:44 UTC on 2026-08-11. You have a hard deadline:
**a new Hard source must be accepted by 23:58 UTC**. Work directly in this
repository. You are authorized to edit in scope, validate, submit one Easy e1
screen, and submit Hard h1 under the conditions below. Do not ask questions.

## Independence rule

Before reading our hypotheses, experiment notes, or submission implementations,
spend a short private design pass deriving a left-field learned architecture
for `x -> x^2 mod N` at T=1, with a tied mechanism that can also execute T
repeated applications. Write the architecture thesis in your new experiment
NOTE. Then read only what is necessary to obey the contract and avoid known
duplicates: `AGENTS.md`, `PITFALLS.md`, `RESEARCH_PROTOCOL.md`, the official
rules/API/validation under `competition/`, `solving/STATUS.md`, and the SHA/job
ledger. You may inspect old code afterward only to reuse evaluator boilerplate,
not its architecture.

## Why this is hard

T=1 must learn `x^2 mod N` from sparse final labels and generalize to unseen x
and unseen N. Public e5 has only 27 training moduli and sparse x coverage. MLPs
and Transformers fit train exactly yet get roughly 4% unseen-N; structured
latent approaches plateau around 10–19%; our generic Neural GPU learned outer
square digits but failed central cross-term/carry columns. Every recent Hard
card has 0/768 at T=1 on both profiles. Exact sequence scoring gives no partial
credit, while hidden Hard may alter the recurrence family. A useful card needs
generic learned state transformation, not a handwritten squaring/modulo solver.

## Required implementation

1. Create a **new** experiment directory dated 2026-08-11 with `submission.py`,
   `NOTE.md`, and `config.json`. The architecture must be materially new in this
   repository, not a renamed or lightly reformatted historical card.
2. It must learn T=1 directly and include a principled tied mechanism for
   applying the learned state transition T times. Favor a bold computational
   abstraction over hyperparameter tweaks.
3. Legal constraints: final-label-only evaluator training; random learned
   weights; no `%` on task values, modular arithmetic oracle, factorization,
   phi, three-argument pow, dataset inspection/augmentation, hard-coded forward
   algorithm, trace targets, custom training loop/backward, CPU offload, or
   precomputed answers. One file, <=256 KiB, <=500M persistent state elements.
4. Register `CARD/CHANGE/PREDICT` in `solving/experiments/predictions.md` before
   running. Prediction must name the causal mechanism and falsifier.
5. Ensure its SHA-1 does not match any existing `submission.py`; record it.
6. Run syntax checks, forbidden-pattern audit, and `one-layer validate`.
7. Submit the exact source to hosted **Easy e1 with `--wait`**. Record job ID,
   score, profiles, completed steps, and errors.
8. If Easy import/validation/runtime succeeds (even if accuracy is weak), you
   may make one fast, causally justified edit only if the evidence clearly
   demands it and time permits. Revalidate and preserve the final SHA. If you do
   not want to edit, submit the screened exact source unchanged.
9. Submit the genuinely new validated source to **Hard h1 without `--wait` no
   later than 23:58 UTC**. Do not submit any historical SHA. Record Hard job ID
   and remaining quota. If Easy has an infrastructure/runtime error, repair it
   once; do not upload a known-broken file to Hard.
10. Do not wait for the Hard result. End with exact paths, SHA-1, Easy job/result,
    Hard job, architectural novelty, legality rationale, and predicted failure
    mode.

The principal explicitly authorizes these Easy and Hard submissions. The goal
is a novel defensible bet, not another duplicate Fable/canonical/ConvGLU card.
