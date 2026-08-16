# Full-budget tuned Muon promotion

This card promotes the width-128 screen winner—flattened-convolution Muon at
learning rate 0.006, momentum 0.95, weight decay 0.1, and a 250-step warmup—to
the full 10,000-step budget. All model, data, seed, dropout, batch, recurrence,
and final-residue-only supervision settings match the AdamW anchor. The two
unseen-N audits were opened once at the validation-selected checkpoint.

## Result

| Optimizer | Train exact | Unseen x, seen N | Seen x, unseen N | Joint unseen | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| AdamW 3e-4 | 8.75% | 7.60% | 6.94% | 7.92% | 677.8 |
| Tuned Muon 0.006 | **16.84%** | **18.14%** | **14.60%** | **18.40%** | 681.0 |

The optimizer hypothesis is confirmed. At essentially identical wall time,
Muon more than doubles validation exactness and produces large gains on both
unseen-N profiles. It reaches 7.46% validation at step 3,500, approximately the
AdamW final endpoint. A sharp transient at step 6,000 drops validation from
12.46% to 6.20%, but the trajectory recovers and the final step is best at
18.14%. Thus the scale is highly effective but still produces noisy recurrent
optimization; a preregistered late-decay test is a high-value follow-up.

Prediction and classification are in
[`../predictions.md`](../predictions.md). The byte-verified raw run is stored
under the ignored `diagnostics/artifacts/prime-7072f85e48094888bcf3893db897ea54/`
tree.
