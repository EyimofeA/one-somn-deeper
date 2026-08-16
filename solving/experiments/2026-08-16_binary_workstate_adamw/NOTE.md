# AdamW control for exact-square binary reduction

This card changed only the optimizer in the matched exact-square reduction arm:
flattened-convolution Muon plus scalar AdamW became AdamW over all parameters.
The binary processor, seed, deterministic x/N splits, 44 tied updates, 128
channels, 9% recurrent dropout, batch 512, validation selection, and 5.12M
example budget were unchanged.

| Optimizer | Train | Unseen x, seen N | Seen x, unseen N | Unseen x, unseen N |
| --- | ---: | ---: | ---: | ---: |
| Muon warmdown | 4.04% | 5.92% | 3.56% | 6.80% |
| AdamW | **11.86%** | **14.56%** | **10.84%** | **15.00%** |

AdamW removed the step-1,500 collapse and was still improving at the final
checkpoint. The prediction is partially confirmed: optimizer instability was
real, but AdamW missed the 25% unseen-N gate and did not fit training. The
supported diagnosis is stable underfitting plus a reduction-side architectural
or objective problem, not optimizer collapse alone.

Evidence (ignored verified backup):
`diagnostics/artifacts/prime-7072f85e48094888bcf3893db897ea54/binary-workstate-adamw-2026-08-16/`.
Figure: [`../../figures/binary_workstate_adamw_2026-08-16.png`](../../figures/binary_workstate_adamw_2026-08-16.png).
