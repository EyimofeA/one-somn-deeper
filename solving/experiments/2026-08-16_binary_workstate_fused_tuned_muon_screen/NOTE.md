# Tuned Muon learning-rate screen

The original fused-model Muon arm used a peak matrix learning rate of 0.02,
almost no weight decay, and no warmup, and collapsed. This screen retains the
same flattened-convolution orthogonalization mechanism but tests more plausible
scales: momentum 0.95, decoupled weight decay 0.1, a 250-step linear warmup,
and peak matrix learning rates 0.001, 0.003, and 0.006. Biases and other
one-dimensional parameters remain on AdamW at 3e-4.

Each arm uses the unchanged width-128 full fused model, seed-74 data, 44 tied
updates, 9% dropout, batch 512, and 3,000 steps. Model selection sees only the
held-out-x/seen-N validation split; both unseen-N audits stay unopened during
the screen.

## Result

| Optimizer | Best step | Train exact | Validation exact | Seconds |
| --- | ---: | ---: | ---: | ---: |
| AdamW 3e-4 anchor | 3,000 | 2.46% | 2.98% | 204.9 |
| Muon 0.001 | 1,500 | 0.15% | 0.50% | 209.0 |
| Muon 0.003 | 3,000 | 4.68% | 4.50% | 208.7 |
| Muon 0.006 | 3,000 | **6.16%** | **6.24%** | 206.9 |

Muon 0.006 is the clear promotion winner. It more than doubles the AdamW
validation improvement above chance at the same step and wall-clock budget,
while 0.001 demonstrates that official-default scale is too conservative for
these flattened recurrent convolutions. The screen supports optimizer scaling
as a second bottleneck independent of channel capacity.

Figure:
[`../../figures/binary_workstate_fused_muon_screen_2026-08-16.png`](../../figures/binary_workstate_fused_muon_screen_2026-08-16.png).
