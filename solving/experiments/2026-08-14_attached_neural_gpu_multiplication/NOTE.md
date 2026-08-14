# Attached Neural GPU adapted to multiplication

Status: promoted as the strongest generic fixed-width multiplication model.

## Adaptation

The supplied `neural_gpu_square.py` was adapted from one-input squaring to
two-input multiplication without adding arithmetic routing:

- both operands are padded to seven LSD-first bits;
- operand A is embedded into workspace row 0 and B into row 1;
- the workspace is `[batch, 128 channels, 4 rows, 14 positions]`;
- one shared 3x3 ConvGRU cell runs for fourteen updates;
- a 1x1 readout decodes fourteen product bits from row 0;
- training uses final-product BCE only, AdamW, 20,000 updates;
- no partial products, carry labels, intermediate traces, or schoolbook routing.

The numeric split is identical to prior baselines: a commutativity-safe 80/20
split of all `0..99 x 0..99` pairs.

## Result

| Model | Train exact | Held-out exact |
|---|---:|---:|
| Transformer MSD-natural | 85.30% | 31.66% |
| Minimal 64-wide Neural GPU | 37.92% | 17.55% |
| Paper Neural GPU, binary short screen | 7.99% | 7.68% |
| **Attached architecture adaptation** | **100.00%** | **64.86%** |

The adaptation first exceeded the Transformer at update 4,000 (32.95%), reached
59.12% at 8,000, interpolated train at 15,000, and peaked at 65.00% test at
19,000. Final per-bit accuracy was 95.67%.

| Held-out group | Exact |
|---|---:|
| one-digit products | 100.00% |
| two-digit products | 97.78% |
| three-digit products | 89.58% |
| four-digit products | 50.61% |
| zero carry columns | 96.20% |
| one carry column | 83.88% |
| two carry columns | 65.83% |
| three carry columns | 50.90% |

Bits 0--2 were 100% exact; the weakest positions were bits 5--7 at
89.38%/76.42%/86.29%. The residual remains concentrated in the central product
region where multiple partial products and carries overlap.

## Interpretation

The result validates the attachment's main mechanism: a larger binary active
workspace with a single shared recurrent transition is easier to optimize than
the small paper reproduction or the prior one-lane tape. It is strong learned
fixed-width interpolation, not yet an algorithmic solution: held-out exact is
65%, only one seed was run, and length generalization is untested.

Verified report, final checkpoint, source, and log:
`diagnostics/artifacts/prime-0e64a3962e874632adeb435a3b192ef4/attached_neural_gpu_multiplication/`.
