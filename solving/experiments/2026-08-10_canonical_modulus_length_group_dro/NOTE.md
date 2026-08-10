# Canonical modulus-length group DRO

Status: completed; refuted at the immediate OOD-N kill.

The exact canonical model and 50/50 T=1 curriculum are retained.  The loss
computes ordinary final-label CE per row, groups rows by the number of observed
decimal digits in `N`, and takes a temperature-0.25 log-mean-exp over group
means.  Existing T=1 row weighting is applied before grouping.

This tests whether common/easier modulus scales drown out the largest training
moduli.  Gate: at least 8/512 on both T=1 profiles; immediate kill at OOD-N
at or below 1/512.  No hosted submission is permitted.

The run scored 0.7500% mean exact, but T=1 was only 2/512 seen-N and 1/512
OOD-N.  The immediate OOD-N kill fired.  Equalizing final-label pressure over
observed modulus lengths does not induce a modulus-general rule.
