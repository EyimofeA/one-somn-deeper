# Binary work-state capacity and learning-rate schedules

## Question

Does a larger reduction processor learn faster, and can a learning-rate
schedule improve the stable AdamW control?

This is a reduction-only diagnostic. Every model receives the exact binary
bits of `x^2` and immutable binary bits of `N`; it must predict `x^2 mod N`
from final residue-bit labels. It does not test whether the model can learn
squaring from competition labels.

## Matched setup

- deterministic seed-74 train/validation/audit split;
- 44 tied recurrent updates and 9% recurrent dropout;
- batch 512, 10,000 steps, or 5.12 million examples;
- validation selects the checkpoint; both audit splits are opened once;
- AdamW is used for every arm.

The anchor has 128 channels, 443,777 parameters, and constant learning rate
`3e-4`. Three isolated changes were tested: 192 channels at the same learning
rate, 500-step warmup to `1e-3` plus cosine decay to `1e-4`, and the same
warmup plus inverse-square-root decay.

## Results

| Arm | Params | Seconds | Train exact | Validation: unseen x | Audit: unseen N | Audit: unseen x and N |
|---|---:|---:|---:|---:|---:|---:|
| 128, constant `3e-4` | 443,777 | 680.1 | 11.86% | 14.56% | 10.84% | 15.00% |
| 192, constant `3e-4` | 997,441 | 1,591.7 | **15.06%** | **18.10%** | **14.40%** | **18.30%** |
| 128, warmup + cosine | 443,777 | 672.8 | 9.86% | 11.96% | 8.30% | 12.20% |
| 128, warmup + inverse sqrt | 443,777 | 672.2 | 11.24% | 13.40% | 9.92% | 13.94% |

## Decision

- **Width 192: retain as a capacity lead, not a speed lead.** It improves all
  endpoint exact metrics at matched examples, but costs 2.34 times as much wall
  time. At approximately 680 seconds, the 128-channel anchor is at 14.56%
  validation while the wide model is only around 11-12% by interpolation.
- **Warmup plus cosine: revert.** Its strong first 500 steps do not persist;
  the aggressive decay leaves it below the constant-rate anchor.
- **Warmup plus inverse square root: revert.** It is better than cosine but
  remains below the constant-rate anchor on validation and both audits.

The 192-channel curve briefly fell from 17.00% validation at step 8,000 to
4.60% at step 8,500, then recovered to 17.70% at step 9,000. AdamW is much
more stable than the earlier Muon run, but the wide run still has a transient
instability worth monitoring.

## Mechanistic conclusion

Additional channels improve capacity or per-example learning, but they do not
increase training speed on the L40. The constant `3e-4` AdamW schedule is the
best tested 128-channel optimizer recipe. Because even the wide privileged
reduction arm fits only 15.06% of training examples and misses the 25%
unseen-`N` gate, the dominant unresolved problem remains learning a reusable
reduction computation, not simply choosing a schedule.

See
[`../../figures/binary_workstate_capacity_schedules_2026-08-16.png`](../../figures/binary_workstate_capacity_schedules_2026-08-16.png)
for step-matched and wall-clock-matched curves.
