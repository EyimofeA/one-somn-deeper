# Easy serial recurrent candidate

`submission.py` is a clean, self-contained Easy candidate. It changes the
current stale Easy anchor by introducing a learned LSD-first scan and a
T-conditioned tied state transition. It has no externally formed square or
reduction state; final supervision is the evaluator-provided output only.

The public Easy evaluator does not expose intermediate labels, so no trace loss
is included. Local smoke validation is required before an e1 submission.
