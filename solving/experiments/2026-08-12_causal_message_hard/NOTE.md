# CMC Hard: causally consumed local messages

Anchor: the runnable GPT-5 Pro gated arithmetic tape (`00ec9e92...`), which
previously scored 0.13% on Easy. One mechanism changes: its generic local
convolutional residual becomes a low-dimensional learned left/right message
stream. Neighbor messages enter the recurrent content update before later
states and digits are formed. The final-label loss, optimizer, tape, recurrence
budget, attention path, and decoder remain unchanged.

Prediction registered before submission: the explicit causal channel may make
carry-like communication easier to reuse, but the report says the prerequisite
diagnostic has not been run. Therefore the honest Hard prediction is no
certified rung and 0--1 exact hits per 768-example T=1 profile; a nonzero T=1
profile would be weak evidence worth reproducing, while 0/768 on both refutes
direct transfer at this budget. This is an owner-requested exploratory Hard
submission, not a promoted research result.

Submitted to Hard as job `580f78bc-de32-4495-a1ca-c34726331d3a`. Uploaded
source SHA-1: `2b1d03547e064639cc914c9cbe6f529c8aec24a2`. The service accepted it and
reported zero Hard attempts remaining for the UTC day; metrics are pending.
