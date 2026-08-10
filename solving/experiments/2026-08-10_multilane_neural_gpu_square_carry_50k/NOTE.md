# Multi-lane Neural GPU: matched 50k carry-duration test

Status: complete; delayed-learning hypothesis refuted.

The 12,000-step terminal carry intervention was null, but the established Task
A Transformer carry curve rose from only 2.77% exact at step 12,000 to 64.51%
at step 50,000. This card runs a matched 50,000-step pair from scratch: the
answer-only Neural GPU and its 130-parameter carry-head variant. Data, split,
seed, six-lane local backbone, 16 tied microsteps, main loss, constant AdamW,
batch size, and evaluation cadence are frozen within the pair.

If the Neural GPU has the same delayed carry-learning mechanism, the carry run
reaches at least 50% unseen-x exact and leads answer-only by at least 20 points.
Kill the route if carry remains below 20% or leads by less than 10 points. If
both improve similarly, duration rather than carry semantics is causal. No
further duration extension follows a kill.

## Result

At 50,000 updates, answer-only reached **53.5000% train exact** and **3.8500%
unseen-x exact**. Carry supervision reached **70.4750% train exact** and
**6.2500% unseen-x exact**. The 2.40-point unseen advantage is far below the
registered 10/20-point thresholds, and carry remains below the 20% kill gate.
Both kill conditions fired.

Carry supervision modestly regularized the grid but did not reproduce the old
Transformer's delayed transition. On unseen x, answer-only's middle digits
were 29.10%/15.60%; carry supervision reached 29.95%/21.80%. The benefit is
real but small and leaves the central computation overwhelmingly incorrect.
The route is closed: do not extend duration or tune the constant learning rate.
The next architecture must make carry causally available during computation,
not merely linearly decodable from the terminal state.

The paired run was copied from Prime pod
`a6eb7c97e54d4174a9b265674758a383` to the ignored local Prime artifact root.
The lifecycle helper verified 11 files and 896,705 bytes on both hosts,
including both checkpoints and source/config snapshots. The pod remains
active. Curves: [`carry_duration_comparison.png`](carry_duration_comparison.png).
