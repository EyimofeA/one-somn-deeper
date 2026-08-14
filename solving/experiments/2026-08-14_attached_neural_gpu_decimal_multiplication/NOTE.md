# Decimal adaptation of the attached Neural GPU

Status: representation control complete; decimal refuted.

## Controlled change

The promoted binary model was changed to decimal symbols while preserving the
128-channel, four-row workspace, one shared 3x3 ConvGRU, fourteen recurrent
updates, split, seed, AdamW settings, and 20,000-update budget. Two LSD-first
decimal digits from each operand occupy rows 0 and 1. A ten-class readout emits
four LSD-first product digits. Training still uses only the final product.

## Result

| Representation | Peak train exact | Peak test exact | Final train exact | Final test exact |
|---|---:|---:|---:|---:|
| Binary | 100.00% | 65.00% | 100.00% | 64.86% |
| Decimal | 99.82% | 26.92% | 98.15% | 25.27% |

Decimal test exact reached 25.92% at 4,000 updates, then remained near 25--27%
while train accuracy rose toward 100%. This is memorization without late
grokking in the matched budget.

| Held-out diagnostic | Accuracy |
|---|---:|
| all output digits | 73.39% |
| ones digit | 98.80% |
| tens digit | 33.80% |
| hundreds digit | 65.35% |
| thousands digit | 95.61% |
| one-digit products | 97.37% |
| four-digit products | 19.66% |
| zero carry columns | 70.25% |
| three carry columns | 13.81% |

## Interpretation

The architecture can optimize decimal multiplication, so decimal failure is
not caused by broken training. The easy edge positions are learned; the tens
position, which requires the cross terms `a0*b1 + a1*b0 + carry`, fails. Seven
binary positions force reusable local interactions and reduce every output to
a binary decision. Two decimal positions instead expose a compact ten-symbol
lookup surface, permitting near-perfect training fit without the same reusable
procedure.

This is one fixed-width seed, not a general claim that decimal neural arithmetic
cannot work. It shows that this particular final-loss-only shared workspace
strongly benefits from binary representation.

Verified report, checkpoint, source, and log:
`diagnostics/artifacts/prime-0e64a3962e874632adeb435a3b192ef4/attached_neural_gpu_decimal_multiplication/`.
