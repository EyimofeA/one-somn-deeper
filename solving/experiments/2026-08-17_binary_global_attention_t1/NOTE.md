# Binary global-attention T=1 transition

## Question

Does modulus-wide communication, rather than raw width, explain the fused
binary ConvGRU's low T=1 ceiling?

## Controlled change

The deterministic varying-N rows, seed, splits, target, checkpoint policy, and
optimizer family match the binary work-state diagnostic. The local 3x3
ConvGRU is replaced by one weight-tied global self-attention cell over three
tapes: immutable x bits, immutable N bits, and a mutable binary workspace.

The model receives no intermediate arithmetic labels. Only final residue bits
are supervised.

## Predicted decision

Promote only above 25% unseen-x/seen-N validation exact with both untouched
unseen-N audits above 20%. Reject below 20% validation exact.

## Result

Rejected at the fixed 5.12M-example budget.

- Parameters: 200,577
- Train exact: 4.472%
- Validation unseen-x / seen-N exact: 3.480%
- Audit seen-x / unseen-N exact: 3.880%
- Audit unseen-x / unseen-N exact: 3.240%
- Wall time: 146.3 seconds including 24.1 seconds of compilation

The result is far below the local ConvGRU reference (22.84% validation,
18.38% and 22.50% unseen-N audits). Global communication alone is therefore
not the bottleneck. The attention model remained underfit, so the next isolated
check is whether its 200k-parameter capacity is simply too small.
