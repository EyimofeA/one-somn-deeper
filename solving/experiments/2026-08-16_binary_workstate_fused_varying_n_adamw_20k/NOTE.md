# Full fused varying-N AdamW at 20k steps

## Question

Was the stable fused AdamW curve stopped too early at 10,000 steps?

Only the training budget changed: 10,000 to 20,000 steps, or 5.12M to 10.24M
sampled examples. The run restarted from the same seed with the same data split,
128-channel model, 44 tied updates, 9% dropout, constant AdamW `3e-4`, and
final residue-bit supervision.

## Results

| Run | Train | Unseen x, seen N | Seen x, unseen N | Unseen x and N |
|---|---:|---:|---:|---:|
| 10k AdamW | 8.75% | 7.60% | 6.94% | 7.92% |
| 20k AdamW | **13.88%** | **9.62%** | **10.62%** | **9.16%** |
| 10k exact-square AdamW | 11.86% | 14.56% | 10.84% | 15.00% |

The 20k run's final step was validation-best. Relative to the earlier 10k run,
validation improved by 2.02 percentage points and cleared the preregistered
1.5-point practical-gain boundary. It narrowly missed the 10% validation gate;
joint-unseen also missed at 9.16%, while the seen-x/unseen-N audit reached
10.62%.

## Reproducibility caveat

The two runs used the same seed, but compiled BF16 GPU execution was not
bitwise deterministic. At step 10,000 the longer rerun had 5.56% validation,
versus 7.60% in the original run, despite similar training exactness. The longer
run recovered to 7.72% at step 15,000 and 9.62% at step 20,000. Therefore the
2.02-point endpoint difference supports continued learnability, but it is not
a precise deterministic estimate of the marginal value of the second 10k
steps.

## Decision

The result is mixed. Ordinary compute still helps: the model is not fully
saturated, and train exact rose to 13.88%. But doubling compute did not deliver
the preregistered 10% validation and joint-unseen pass, and the curve shows
gradual noisy improvement rather than grokking.

The direct exact-square diagnostic remains substantially better on held-out x
and joint unseen N. Longer training alone is unlikely to close that
multiplication/reduction credit-assignment gap efficiently. Retain 20k as a
compute reference, then change either the architecture or the Muon recipe
rather than automatically doubling the budget again.

Figure:
[`../../figures/binary_workstate_fused_varying_n_adamw_20k_2026-08-16.png`](../../figures/binary_workstate_fused_varying_n_adamw_20k_2026-08-16.png).

Ignored verified backup:
`diagnostics/artifacts/prime-7072f85e48094888bcf3893db897ea54/binary-workstate-fused-varying-n-adamw-20k-2026-08-16/`.
