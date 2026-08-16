# Full fused capacity × optimizer combination

This factorial card combines exactly the two independently successful changes:
256 hidden channels and flattened-convolution Muon at learning rate 0.006,
momentum 0.95, weight decay 0.1, and a 250-step warmup. Data, seed, 44 tied
updates, 9% dropout, batch 512, final-residue-only supervision, and the
10,000-step budget remain fixed. No decay or other third change was added.

## Result

| Width + optimizer | Train exact | Unseen x, seen N | Seen x, unseen N | Joint unseen | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 + AdamW | 8.75% | 7.60% | 6.94% | 7.92% | 677.8 |
| 256 + AdamW | 15.38% | 13.74% | 11.22% | 13.56% | 2,171.5 |
| 128 + tuned Muon | 16.84% | 18.14% | 14.60% | 18.40% | **681.0** |
| 256 + tuned Muon | **22.09%** | **22.84%** | **18.38%** | **22.50%** | 2,055.9 |

The combination clears the preregistered train, validation, and both unseen-N
gates, so capacity and optimizer preconditioning are complementary. It misses
the 27% super-additivity threshold: part of their benefit overlaps. At matched
steps the combination is the best arm, reaching 9.52% validation by step 3,000
and 22.84% by step 10,000. At matched wall time, width-128 Muon remains the
strong practical winner because it reaches 18.14% in one third the runtime.

The close train/validation endpoint is evidence against a simple memorization
explanation, and the two unopened audits confirm transfer. It is not evidence
that modular squaring is solved: more than three quarters of exact transitions
still fail, and the first unseen-N audit is only 18.38%.

Prediction and classification are in
[`../predictions.md`](../predictions.md). Figure:
[`../../figures/binary_workstate_fused_capacity_optimizer_2026-08-16.png`](../../figures/binary_workstate_fused_capacity_optimizer_2026-08-16.png).
The ignored raw artifact was copied from Prime pod
`7072f85e48094888bcf3893db897ea54` and verified file-by-file with SHA-256.
