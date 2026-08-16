# Full fused varying-N T=1 with AdamW

## Question

Did the fused x,N processor fail because Muon collapsed, or because the full
modular-squaring transition is intrinsically harder for this architecture?

This is the matched AdamW control for the earlier fused arm. The single change
is the optimizer: constant AdamW `3e-4` over every parameter replaces Muon
warmdown plus scalar AdamW. The data split, source representation, 128-channel
processor, 44 tied updates, 9% dropout, seed, and 5.12M-example budget are
unchanged.

The model receives x and N bits and only final residue-bit labels. It never
receives square, quotient, comparison, subtraction, carry, or execution-trace
supervision.

## Results

| Arm | Train | Unseen x, seen N | Seen x, unseen N | Unseen x and N |
|---|---:|---:|---:|---:|
| Fused, Muon | 2.41% | 2.04% | 2.06% | 1.74% |
| Fused, AdamW | **8.75%** | **7.60%** | **6.94%** | **7.92%** |
| Exact-square input, AdamW | 11.86% | 14.56% | 10.84% | 15.00% |

The AdamW fused curve was stable and selected its final step. Validation rose
from 1.04% at step 2,000 to 4.72% at step 5,000 and 7.60% at step 10,000. The
older Muon fused arm selected step 1,000 and collapsed afterward.

## Decision

The prediction is confirmed: optimizer collapse was a large secondary failure.
AdamW produced transferable signal across both unseen x and unseen N and
cleared the preregistered 5% gate. It missed the 10% strong-lead gate, and the
full training set remained only 8.75% fitted.

The exact-square AdamW arm's 14.56% validation versus fused AdamW's 7.60%
quantifies the remaining multiplication/reduction credit-assignment burden
under a controlled processor and optimizer. Because train and validation
track closely, the current failure is underfitting rather than the fixed-N
lookup behavior seen in the preceding experiment.

This card is retained as the new full-machine local anchor. It is not ready for
a competition submission: it has not learned a strong T=1 transition and has
not been evaluated under scorer wall-clock constraints.

Figure:
[`../../figures/binary_workstate_fused_varying_n_adamw_2026-08-16.png`](../../figures/binary_workstate_fused_varying_n_adamw_2026-08-16.png).

Ignored verified backup:
`diagnostics/artifacts/prime-7072f85e48094888bcf3893db897ea54/binary-workstate-fused-varying-n-adamw-2026-08-16/`.
