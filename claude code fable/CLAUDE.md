# Hard-tier modular-recurrence competition — handoff

## Files here
- `PRIMARY_SOURCES.md` — full competition packet (rules, evaluator source, generator, results table). Ground truth for everything.
- `submission_v1.py` — first build: T-proportional tied-loop digital-register iterator. Contract-verified against a harness replica.
- `submission_v2.py` — v2 fixes: confidence-gated quantization hardening (alpha capped 0.90, was wallclock-gated and caused a collapse), SwiGLU mixer for multiplicative digit products. Also contract-verified.
- `smoke_test.py` — harness-replica test suite (source policy, collate/target_positions semantics, eval-purity/determinism, bf16 autocast, grad-clip train loop). Run this against any new submission.py before trusting it.

# Hard-tier modular-recurrence competition — handoff

## Files here
- `FULL_TRANSCRIPT.md` — complete reasoning trail from the strategy session. Read this
  first if anything below is unclear; it has the derivations, not just conclusions.
- `PRIMARY_SOURCES.md` — full competition packet (rules, evaluator source, generator,
  results table). Ground truth for the contract.
- `submission_v1.py` — first build: T-proportional tied-loop digital-register
  iterator. Contract-verified. Known flaw: wallclock-gated quantization hardening
  causes an irreversible collapse to the digit-marginal floor (~2.17 test loss).
  Superseded by v2 — kept for reference only, do not submit.
- `submission_v2.py` — fixes the collapse: confidence-gated hardening (EMA of
  register-digit confidence, alpha capped at 0.90 in training), SwiGLU mixer for
  multiplicative digit products. Contract-verified. Current best build, but **not
  yet shown to solve the actual problem** — see status below.
- `smoke_test.py` — harness-replica contract suite (source policy, collate/
  target_positions semantics, eval-purity/determinism, bf16 autocast, grad-clip
  loop). Run against ANY new submission.py before trusting or submitting it.

## Status (as of handoff)
Independent real-GPU runs (by another agent, referred to as "Fable" in the transcript
— this is just a model name, not a separate tool) confirmed: every architecture tried
so far converges to test loss ~2.17, which is the digit-marginal floor, not real
computation. T=1 probe (single squaring step, no depth composition) scores ~0.75%
exact-match — at floor. **No build has yet demonstrated learning an actual modular
squaring step for an unseen modulus.** v2's fixes address a training-stability bug
(the collapse), not this deeper representational gap.

## The active gate — read this before spending Hard-tier quota
Per the transcript's P2 plan: **do not expect submission_v2.py to score above the
current leaderboard best (0.40%) yet.** Before investing further Hard-quota effort,
run the local grokking ladder (fixed single N -> multi-N seen -> held-out N, all at
T=1) to establish whether the one-step map is learnable at all. Rung 3 clearing
~5% exact-match locally is the actual gate for expecting anything from v2 on Hard.
Firing Hard shots with v2 anyway is fine (free, monotone-best-kept per competition
rules) — just don't read a 0% score as new information without first checking the
local diagnostics below.

Diagnostics to run early (details in FULL_TRANSCRIPT.md, section "Diagnostics
standardized D1-D7"): per-position digit accuracy (finer than exact-match near floor),
loop-state decoding against known intermediates, and the T=1 epsilon-suite split by
seen-N vs held-out-N.

Pretraining embedded weights is currently excluded by assumption, not by rules text —
worth a one-line question to the organizers if it becomes the real bottleneck.

## Environment notes
- Local dev: CPU torch is enough for contract-testing (`pip install torch
  --break-system-packages`). No GPU in a plain container — actual training runs need
  real compute (H100/L40S per the packet's compute budget section).

