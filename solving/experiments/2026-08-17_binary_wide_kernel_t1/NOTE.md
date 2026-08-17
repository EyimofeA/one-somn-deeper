# Wide-kernel binary T=1 transition

The 3x3 ConvGRU was the strongest local transition but needs 44 microsteps.
This card expands every gate kernel to 3x7 and uses 11 updates, so each output
can communicate across the 22-bit workspace while preserving convolutional
alignment, gates, immutable x/N lanes, and final-only supervision.

Promote above 20% validation exact. Reject below 10%.

## Result

Rejected.

- Parameters: 4,131,585
- Best checkpoint: step 4,500
- Train exact at selected checkpoint: 43.858%
- Validation unseen-x / seen-N exact: 5.100%
- Audit seen-x / unseen-N exact: 4.840%
- Audit unseen-x / unseen-N exact: 4.980%
- Final-step train probe: 62.700%
- Wall time: 1,031.6 seconds

The 3x7 cell processed the fixed example budget about twice as fast as the
44-update 3x3 reference, but used that capacity to identify seen x values. Its
held-out-x score plateaued near 5% from step 1,500 onward while train exact rose
above 60%. There was no delayed grokking rise. Wide transport is a shortcut,
not an algorithmic improvement.
