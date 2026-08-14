# Reduction-only recurrent-depth sweep

Status: completed research diagnostic; not a competition candidate.

## Question

When the model receives the correct decimal digits of `x squared`, is poor
unseen-modulus reduction primarily caused by too few internal recurrent
updates?

## Control

One 261,962-parameter local recurrent tape was trained from final remainder
labels at eight tied updates. It received `(N, x squared)` rather than `x`, so
learned multiplication was removed from the path. Quotients, comparisons,
borrows, and intermediate states were never supervised. The identical frozen
checkpoint was evaluated at 1, 2, 4, 8, 16, and 32 updates.

## Result

Training stopped after three consecutive full-training checks above 99.9%:
800 updates in 8.75 seconds.

| Evaluation updates | Train exact | Held-out x exact | Unseen N exact |
|---:|---:|---:|---:|
| 1 | 12.98% | 6.30% | 10.51% |
| 2 | 20.35% | 7.98% | 14.72% |
| 4 | 43.89% | 10.92% | 16.12% |
| 8 | 100.00% | 11.34% | 18.69% |
| 16 | 54.57% | 8.82% | 17.99% |
| 32 | 23.98% | 8.82% | 16.36% |

The two leading zero positions reach 100% trivially because every remainder is
below 100. At the trained depth, unseen-N LSD and tens-digit accuracies are only
36.92% and 30.84%.

## Interpretation

More recurrent updates do not repair this checkpoint: extending eight updates
to sixteen or thirty-two damages even training exactness. The learned dynamics
are tuned to their training horizon rather than converging toward a stable
reduction algorithm. Because held-out-x accuracy is also low, the failure is
broader than unseen-modulus routing: the model largely memorized training
examples despite receiving the correct product.

Raw report, checkpoint, source snapshot, and logs are backed up at
`diagnostics/artifacts/prime-0e64a3962e874632adeb435a3b192ef4/reduction_only_depth_sweep_seed0/`.

## One-minute grokking control

A fresh matched seed-0 run removed early stopping and trained for exactly 60
seconds (5,162 updates). Generalization was measured every 400 updates but was
never used to stop or modify training.

| Time | Train exact | Held-out x exact | Unseen N exact |
|---:|---:|---:|---:|
| 5.11 s | 100.00% | 11.76% | 16.82% |
| 14.44 s | 100.00% | 12.18% | 18.93% |
| 28.35 s | 100.00% | 12.18% | **19.16%** |
| 46.91 s | 100.00% | 11.34% | 16.82% |
| final | 100.00% | 10.92% | 17.76% |

There is no grokking transition on this horizon. Accuracy fluctuates around the
same low plateau after interpolation, and the final checkpoint is worse than
the peak. This rules out a rapid delayed-generalization event, not grokking at
substantially longer horizons or under different regularization.

The second verified backup is
`diagnostics/artifacts/prime-0e64a3962e874632adeb435a3b192ef4/reduction_only_grokking_60s_seed0/`.
