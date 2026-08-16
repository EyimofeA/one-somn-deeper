# Full fused varying-N capacity test

This card changes only the hidden channel count from 128 to 256 in the full
binary work-state model. Data, seed, final-residue-only supervision, 44 tied
recurrent updates, 9% dropout, AdamW at a constant 3e-4, batch 512, and the
10,000-step budget remain fixed. The parameter count rises from 443,777 to
1,772,289.

Prediction and result classification are registered in
[`../predictions.md`](../predictions.md). The byte-verified GPU artifact is
stored under the ignored `diagnostics/artifacts/prime-7072f85e48094888bcf3893db897ea54/`
tree.

## Result

| Width | Train exact | Unseen x, seen N | Seen x, unseen N | Joint unseen | Seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 8.75% | 7.60% | 6.94% | 7.92% | 677.8 |
| 256 | **15.38%** | **13.74%** | **11.22%** | **13.56%** | 2,171.5 |

The capacity hypothesis is confirmed. Width 256 improves exact-match learning
at the same optimizer-step and example budget on all four splits. It reaches
roughly the width-128 20,000-step validation endpoint after only 6,000--7,500
steps. The trade-off is throughput: compilation plus training is about 3.2
times slower, and at a matched wall time the two validation curves are close.
Thus the experiment diagnoses insufficient recurrent capacity per step, but
does not by itself provide a cheaper competition-time model.

Figure:
[`../../figures/binary_workstate_fused_width256_2026-08-16.png`](../../figures/binary_workstate_fused_width256_2026-08-16.png).
