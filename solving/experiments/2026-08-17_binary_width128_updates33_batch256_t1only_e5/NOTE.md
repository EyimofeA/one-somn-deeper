# Full-window T=1 gradients, batch 256

This changes only the wall-clock curriculum fraction from 0.50 to 1.00. The
model receives the same E5 rows, but custom loss ignores non-T=1 examples for
the full 60-second training window. It is still final-label-only learning.

The card targets the first certification rung. Promote if both T=1 profiles are
nonzero and one reaches at least 2/512, with more than 450 updates.

## Result

The run completed 731 updates, versus 505 for the half-window batch-256 arm.
Final train loss fell to `0.547514`. Overall Easy mean exact was `0.04167%`
(`1/1200` seen-N test, `0/600` OOD-N test). The seen-N T=1 profile was
`2/512 = 0.390625%`; the OOD-N T=1 profile was `0/512`.

This **fails promotion** because both T=1 profiles were required to be
nonzero. Concentrating the clock on T=1 clearly improves update throughput and
loss, but does not by itself produce a modulus-general transition. Therefore
mixed-depth gradient competition was a secondary optimization problem, not the
root cause of the T=1 generalization failure.
