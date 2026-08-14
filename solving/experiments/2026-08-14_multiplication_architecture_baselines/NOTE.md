# Fixed-width multiplication architecture baselines

Status: serial GRU and minimal Neural GPU complete; both refuted.

## Shared task

Every ordered multiplication pair in `0..99 x 0..99` was assigned through the
same commutativity-safe 80/20 unordered-pair split: 7,994 train and 2,006 test.
All models predict the exact four-digit product from final product labels only.
There is no length-generalization claim.

## Architectures and result

| Model | Mechanism | Parameters | Train exact | Test exact |
|---|---|---:|---:|---:|
| Serial VDF-style GRU | paired LSD columns, then two flush positions | 34,762 | 28.12% | 7.18% |
| Minimal Neural GPU | one 64-wide tape, eight tied local ConvGRU updates | 47,562 | 37.92% | 17.55% |
| Best Transformer control | MSD-natural encoder-decoder | 169,998 | 85.30% | 31.66% |

The serial GRU emits the LSD perfectly but receives only aligned operand pairs.
It cannot easily form the cross-position terms required for the middle product
columns. Its held-out tens/hundreds accuracies were 17.75%/37.99%, and
three-carry exactness was 5.24%.

The minimal Neural GPU allows digit information to move between neighboring
positions and nearly triples serial-GRU exactness. Nevertheless, its held-out
tens digit remained 23.23%, two-digit by two-digit exactness was 12.67%, and
three-carry exactness was 8.95%. Local communication is helpful but does not by
itself create organized partial-product and carry workspaces.

## Decision

Neither generic recurrent architecture clears the Transformer, much less the
90% fixed-width gate. Do not interpret the earlier small-regime square lookup
as solved multiplication. The next architectural question, if selected, must
change state organization rather than merely add duration: separate learned
lanes/workspaces for partial products and carry, while keeping routing generic.

Verified raw artifacts:

- `diagnostics/artifacts/prime-0e64a3962e874632adeb435a3b192ef4/serial_gru_2x2_multiplication/`
- `diagnostics/artifacts/prime-0e64a3962e874632adeb435a3b192ef4/simple_neural_gpu_2x2_multiplication/`
