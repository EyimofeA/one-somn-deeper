# Multi-lane Neural GPU: raw-square capability

Status: complete; capability gate failed.

This is the first primitive gate for the generic competition-legal-shaped local
grid. The forward has an eight-position LSD-first tape, six writable lanes,
64-wide states, asymmetric left/self/right learned mixing, learned same-position
lane mixing, and one tied GRU update repeated 16 times. It contains no multiply,
carry, square, compare, subtract, modulus, or phase schedule.

The card trains from final raw-square labels on a frozen random split of all
four-digit-width inputs: 8,000 train x and 2,000 strictly unseen x. The N tape
is present but zero because reduction is intentionally out of scope. Arithmetic
appears only in synthetic label construction.

Capability passes at >=90% unseen-x exact and is strong at >=99%. Kill the
architecture for this primitive below 50% unseen-x exact or below 90% train
exact after 12,000 updates. One seed only; replication requires a new decision.

## Result

After 12,000 updates, seed 0 reached **12.9125% train exact** and **4.0000%
unseen-x exact**. Per-digit accuracy was 81.9000% train and 75.6125% unseen-x.
Both preregistered kill conditions fired. The near-matched early curves show
that the grid learned shared decimal correlations, but the widening late gap
and low exactness show that it did not discover the exact squaring algorithm.
There was no grokking-like transition.

A frozen-checkpoint position audit rules out a leading-zero explanation for
the 75.6125% digit score. LSD-to-MSD digit accuracies were **100.00%, 99.80%,
96.10%, 29.05%, 15.20%, 68.15%, 96.95%, and 99.65%**. The corresponding
target-zero rates were only 9.75%, 22.05%, 12.75%, 12.35%, 11.05%, 11.20%,
15.85%, and 30.40%. The grid learned the low- and high-edge identities of
decimal squaring, but failed in the central columns where multiple cross terms
and carries must be accumulated.

Do not add reduction, more microsteps, or optimizer tuning to this card. The
next research decision must change the supervision or computational primitive,
not extend this failed final-label-only generic grid.

The remote run was copied from Prime pod
`a6eb7c97e54d4174a9b265674758a383` to ignored local backup
`diagnostics/artifacts/prime-a6eb7c97e54d4174a9b265674758a383/runs/2026-08-10_multilane_neural_gpu_square`.
The lifecycle helper verified six files and 391,886 bytes on both hosts,
including a source/config snapshot. The
pod remains active. Learning curves are in
[`learning_curves.png`](learning_curves.png).
