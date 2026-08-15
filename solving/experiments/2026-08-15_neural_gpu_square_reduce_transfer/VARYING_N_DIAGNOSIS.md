# Why the model fails when N varies

## Evidence

The frozen hosted source scores 16.25%--41.13% on every fixed-modulus Easy
dataset and 0.54%--0.62% on all varying-modulus datasets. Raw curves and the
full table are in [`NOTE.md`](NOTE.md) and [`metrics/`](metrics/).

The direct binary squarer reaches 99.55% untouched audit, but the competition
model transfers only its topology. It does not load those weights, changes the
representation from binary bits to decimal token embeddings, lets the square
phase see N, and trains from modular final labels only. Therefore the hosted
results do not yet measure pretrained-square transfer.

## Failure decomposition

### Data

E1/E2/E6--E10 contain one constant N. A network may absorb that N into its
weights and learn a modulus-specific map. E3/E4/E5 sample N. E3/E4 supervise
only T=2; E5 supplies T=1/2/3. Because all three fail similarly, missing T=1
alone is not the explanation. E5's final train exact is 32% while held-out
accuracy is 0.54%, showing substantial example fitting without reusable N
conditioning.

### Architecture

The mutable four-row state receives N once at initialization and once at the
square/reduce boundary. The reducer then overwrites that state for 2W local
updates without immutable N reinjection. Fixed N makes forgetting harmless;
varying N makes it fatal. The square phase also sees N, allowing an entangled
fixed-N shortcut instead of an N-independent raw-square representation.

The reducer has neither an explicit learned comparator/stop channel nor the
immutable-N interface used by successful directly supervised reducer
diagnostics. A generic fixed-depth ConvGRU can memorize one small modular map,
but has no enforced reason to learn compare/subtract behavior across N.

### Optimizer

The submission's wall-clock Muon schedule differs from the successful pilot's
update-count schedule. This explains M6 collapse and may destabilize later Easy
updates, but it cannot explain the fixed-N/varying-N discontinuity by itself.
E3--E5 fail before demonstrating a transferable N-conditioned rule.

### Data-architecture interaction

Final labels do not identify the internal square/reduce boundary. With fixed N,
the easiest solution is an entangled map specialized to that N. With varying N,
that shortcut disappears, while the architecture still does not protect N or
force the first phase to remain N-independent. This is the leading mechanism.

## Ranked controlled fixes

1. Reinject immutable N at **every reducer microstep**, changing nothing else.
2. Make the square phase N-blind, changing nothing else.
3. Run H14: frozen pretrained, trainable pretrained, random square plus matched
   reducer, and fully entangled control. Use a competition-shaped decimal
   squarer for a fair transfer test; binary weights are not directly compatible.
4. Add a generic learned comparator/continue channel to the reducer without a
   coded subtraction schedule.
5. Apply the exact step-based Muon warmdown before interpreting Medium scale.

Width is not first: 128 channels already solve fixed-width 11-bit squaring and
learn substantial fixed-N recurrence. Increase width only after immutable-N
conditioning and optimizer stability are controlled.

## Measurement repair

[`../../../diagnostics/monitor_competition_by_t.py`](../../../diagnostics/monitor_competition_by_t.py)
now trains with the submission's real `token_training_loss`, saves peak/final
checkpoints, and evaluates aggregate, `depth_t_*`, and `depth_ood_n_t_*` splits.
Future local runs therefore report exact seen-N and unseen-N profiles for
T=1/2/4/8/16/32/64 instead of one aggregate score. These are locally
regenerated public-data diagnostics, not retrospective hosted metrics.
