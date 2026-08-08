# Today — 2026-08-08

## Competition execution

The new Hard card changes one mechanism in the GPT-5 Pro exact-match/SAM
source: T=1 prompt rows receive normalized 8x loss weight. Architecture,
optimizer, recurrence, batch reuse, and inference are unchanged.

- Source: `solving/submissions/exact_match_optimizer/submission.py`
- SHA-1: `8c796bf39f3b0d2f90043b08430be26c23f0f180`
- Static validation: passed on evaluator `e32c2f9`; 1,600,513 model-state
  elements; banned-token audit clean.
- Local L40 e5: 0.4583% mean; seen-N T=1 5/512; OOD-N T=1 0/512.
- Hosted Easy e5: `cb98f944-9b21-4869-af5a-c924845ca89e`, 0.3750% mean;
  seen-N T=1 3/512; OOD-N T=1 0/512.
- Hosted Hard: `9e7404cb-b0c9-480a-aa64-8d90cc853d67`, queued; daily Hard
  quota remaining: 0.

The parent exact-match card's hosted e5 profile was 2/512 seen-N T=1 and
1/512 OOD-N T=1. The new card improves only the primary seen-N count and ties
the parent's total T=1 hits. This is deliberately a weak first-rung bet, not a
claimed solution.

## Research completed

External-review synthesis is in `MODEL_REVIEW_SYNTHESIS.md`. The T=1 state
tournament and refiner results are in
`diagnostics/artifacts/t1_tournament_2026-08-08/summary.md`. The matched
decimal/binary/limb comparison is in
`diagnostics/artifacts/t1_representation_2026-08-08/summary.md`.

Deterministic global latent versus discrete diffusion/refinement on the L40:

| Arm | Train steps/s | Unseen-N exact | Unseen-N ms/example |
|---|---:|---:|---:|
| Global latent | 191.77 | 16.36% | 0.00623 |
| Refiner K=4 | 122.04 | 8.18% | 0.00726 |
| Refiner K=8 | 122.04 | 8.88% | 0.00944 |

The tested refiner is slower and materially less accurate. Structured latent
state is a narrow positive over global state; simple binary/limb replacement
is negative. These results kill another prompt-register curriculum, this
masked-token refiner, and representation-only micro-tuning under the tested
conditions.

## Current frontier and next decision

T=1 transition identifiability remains the bottleneck. A model can fit seen
rows while failing to learn a modulus-conditioned local law. The next useful
card should combine the validated Square→Reduce phase decomposition with a
real LSD-aligned latent tape under final-label-only T=1 training, against the
current structured-tape control. It should not add traces or tune recurrence
depth. This card is proposed, not launched.

Recent hosted references: Easy e1 8.50% (`7ee881f6`), Medium m5 0.17%
(`60510147`), prior Hard 0.0600% with zero T=1 (`14ce2afb`).
