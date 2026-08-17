# Dedicated scratch-lane fast messages

Every fourth local clock writes a gated shifted copy of work lane 2 into the
otherwise-unused lane 3. The gates begin at zero, preserving the local anchor;
the normal 3x3 cell can consume the separate message on subsequent clocks.

## Result

The selected step was 7,500: train exact `14.224%`, validation `17.16%`,
seen-x/unseen-N audit `13.52%`, and unseen-x/unseen-N audit `17.70%`.

The scratch lane accelerated early learning but missed the registered 18%
validation gate and did not improve both audits over the local anchor. Final
gate values and quotient slicing determine whether the mechanism moved at all.

The gates opened to `[0.0835, 0.1166, 0.1848]`, but raw `q=32..63` was exactly
`0%` and `q>=64` was `0.27%`. Thus the scratch bus improves early optimization
without extending reduction depth. **Reject.** Future fast transport must be
coupled to an explicit compare/subtract state transition or learned addressing,
not merely shifted activation injection.
