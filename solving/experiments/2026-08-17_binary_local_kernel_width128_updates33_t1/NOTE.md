# Width-128 / 33-update T=1 control

The official E5 transfer showed that width 256 receives only 129 updates in 60
seconds. This control changes only width to 128 while preserving the locally
promoted 33-step computation.

Promote above 16% validation with both unseen-N audits above 13% and wall time
below 800 seconds.

## Result

Promoted as the clock-aware local transition.

- Parameters: 443,777
- Best checkpoint: step 9,500
- Train exact: 15.658%
- Validation unseen-x / seen-N exact: 16.420%
- Audit seen-x / unseen-N exact: 13.620%
- Audit unseen-x / unseen-N exact: 16.800%
- Wall time: 639.4 seconds, including a 95.8-second cold compile

All accuracy and time gates passed. Width 128 is weaker per example than width
256 but far stronger per wall-clock. Promote it for evaluator-interface tests.
