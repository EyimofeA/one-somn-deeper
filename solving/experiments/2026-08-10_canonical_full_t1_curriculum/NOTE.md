# Canonical full T=1 curriculum

Status: completed; refuted by exact memorization and zero OOD-N T=1.

This card differs from the exact canonical register only by setting
`T1_ONLY_FRACTION` from 0.50 to 1.00.  The entire minute therefore trains the
same single transition used by the T=1 evaluation, with ordinary mean token CE.

Gate: at least 8/512 seen-N and 5/512 OOD-N T=1.  Stop after one local run if
either side fails.  No hosted submission is permitted.

Training loss fell to 0.0516, but mean held-out exact was only 0.1667% and
evaluation loss exceeded 8.5.  T=1 was 4/512 seen-N and 0/512 OOD-N.  Giving
every update to T=1 therefore intensifies memorization instead of identifying
the modulus-general transition.
