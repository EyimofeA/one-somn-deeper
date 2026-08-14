# Transformer 2x2-digit multiplication representation factorial

Status: complete; fixed-width Transformer baseline refuted.

## Question and split

Can a simple 169,998-parameter encoder-decoder Transformer interpolate
multiplication over every operand in `0..99`, and how do digit direction and
leading-zero padding affect it?

The deterministic 80/20 split groups unordered operand pairs: if `37 x 82` is
test, `82 x 37` is also test. This prevents commutativity leakage. There are
7,994 train and 2,006 held-out ordered examples. No length extrapolation is
claimed.

## Result

| Direction | Padding | Train exact | Test exact | Length exact | Invalid |
|---|---|---:|---:|---:|---:|
| LSD | natural | **94.12%** | 26.92% | 97.76% | 0% |
| LSD | zero padded | 93.63% | 23.13% | 99.85% | 0% |
| MSD | natural | 85.30% | **31.66%** | 97.36% | 0% |
| MSD | zero padded | 77.86% | 19.94% | 100.00% | 0% |

The preregistered direction prediction was wrong: MSD-natural generalized best
despite fitting train less. Padding behaved as predicted: it repaired output
formatting but reduced numerical exactness.

## Failure localization

For the best MSD-natural arm:

- LSD/tens/hundreds/thousands digit accuracy: 98.65% / 38.21% / 58.79% / 91.17%.
- One-, two-, three-, and four-digit products: 100.00% / 80.00% / 50.52% / 17.97% exact.
- Zero-, one-, two-, and three-carry-column cases: 85.44% / 57.88% / 38.71% / 4.48% exact.
- Two-digit by two-digit operands: 20.33% exact.

The model learns formatting and the local least-significant product digit. The
middle digits, where cross-products and carries interact, are the failure. MSD's
lower train fit and higher test score may be regularization, not algorithmic
learning; this single seed cannot distinguish them.

Raw reports, checkpoints, logs, and source are verified at
`diagnostics/artifacts/prime-0e64a3962e874632adeb435a3b192ef4/transformer_2x2_factorial/`.
