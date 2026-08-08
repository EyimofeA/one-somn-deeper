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
- Exact-source L40 replication: 0.6250% mean; seen-N T=1 4/512; OOD-N T=1
  1/512; no certified rung.
- Hosted Easy e5: `cb98f944-9b21-4869-af5a-c924845ca89e`, 0.3750% mean;
  seen-N T=1 3/512; OOD-N T=1 0/512.
- Hosted Hard: `9e7404cb-b0c9-480a-aa64-8d90cc853d67`, running; daily Hard
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

Two three-seed final-label controls further localize T=1 failure. Hiding `N`
during a generic square phase gives 11.76% held-out-x / 17.29% unseen-N median,
versus 13.03% / 17.29% when `N` is visible. Replacing the square phase with
learned pair categories, pair-to-column routing, a shared fold, and an LSD
carry scan is worse at 10.50% / 16.36%. All nine runs fit train 100%. The raw-
square inductive bias that works under direct square labels is not identified
through final modular labels.

## Current frontier and next decision

T=1 transition identifiability remains the bottleneck. Phase separation and
an LSD-aligned pair-fold tape are now refuted, so the next card must change
legal credit assignment/identifiability. Do not tune recurrence depth, swap
number representation again, or transfer either new diagnostic to hosted
evaluation.

Recent hosted references: Easy e1 8.50% (`7ee881f6`), Medium m5 0.17%
(`60510147`), prior Hard 0.0600% with zero T=1 (`14ce2afb`).

## Repository organization audit

The conceptual split is sound, but the physical tree has drifted:

- `RESEARCH_PROTOCOL.md` requires per-experiment `NOTE.md` and `config.json`,
  while `solving/experiments/LAYOUT.md` explicitly forbids them. Reconcile
  this contradiction first.
- `diagnostics/` is 1.7 GB and mixes reusable source with 1,076 raw artifacts,
  analysis exports, logs, and a virtual environment. Keep source/tests/configs
  there; move generated material under one ignored artifact root with a compact
  tracked index.
- Root-level session exports, metrics, and duplicate 177–178 MB research
  packets obscure the entry points even when ignored. Put recoverable exports
  outside the repository and retain one manifest/link.
- `solving/submissions/` is documented as active-only but mixes five historical
  symlinks, six new source directories, temporary source, and reports. Give
  each active candidate one named pointer and keep provenance in its card.
