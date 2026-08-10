# Canonical local-Conv scale trajectory

Status: completed; locality-underuse hypothesis refuted.

This is the exact true-0.1 source plus read-only scheduler telemetry every 100
optimizer steps: local scalar value/gradient and Conv/GLU weight norms. The
base wall-clock schedule steps first and training computation is unchanged.
No hosted submission is permitted from this instrumented source.

The scalar rose monotonically from 0.10297 after the first step to 1.77103 at
step 1600.  The GLU weight norm grew from 9.119 to 44.360 while gradients
remained nonzero.  Despite that active recruitment, public e5 T=1 was only
5/512 seen-N and 2/512 OOD-N (mean exact 0.8333%).  The branch therefore was
not starved by its 0.1 initialization: final-label optimization amplified it,
but the learned local features did not become a transferable arithmetic rule.
