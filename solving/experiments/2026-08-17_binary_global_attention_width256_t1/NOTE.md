# Binary global-attention width 256 T=1

## One change

Increase the rejected global-attention transition from width 128 to width 256.
Everything else remains fixed, including the deterministic data and 5.12M
example budget.

## Prediction and gate

Train exact should exceed 12% and validation exact should exceed 10%. Reject
below 8% validation. Keep alive for optimizer work only above 20% validation.

## Result

Rejected.

- Parameters: 794,369
- Best step: 9,500
- Train exact: 4.861%
- Validation unseen-x / seen-N exact: 3.060%
- Audit seen-x / unseen-N exact: 4.020%
- Audit unseen-x / unseen-N exact: 2.560%
- Wall time: 309.2 seconds

Quadrupling parameter count did not improve the width-128 validation result
(3.48%). The attention branch is closed: neither global communication nor
capacity was the missing mechanism.
