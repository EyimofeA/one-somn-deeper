# Local-kernel 22-update T=1 transition

This is the direct throughput control missing from the earlier 44-to-11 jump.
It preserves the width-256, 3x3 ConvGRU exactly and removes only half of the
recurrent microsteps.

Promote above 20% validation exact. Reject below 15%.

## Result

Rejected as the new anchor.

- Parameters: 1,772,289
- Best checkpoint: step 7,500
- Train exact: 19.459%
- Validation unseen-x / seen-N exact: 14.240%
- Audit seen-x / unseen-N exact: 10.440%
- Audit unseen-x / unseen-N exact: 13.640%
- Wall time: 1,158.1 seconds, including a 61.7-second cold compile

The arm approximately matched the 44-step model at equal wall time through the
middle of training, but plateaued around 14% and failed all promotion gates.
One traversal of the 22-position workspace is not enough. Continue to the
preregistered 33-step interpolation; do not spend a longer budget on 22 steps.
