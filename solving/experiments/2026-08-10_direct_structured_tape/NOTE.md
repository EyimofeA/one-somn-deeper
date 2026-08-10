# Direct structured LSD tape

Status: completed; both 60-second and 300-second gates refuted.

This is the closest final-label-only competition port of the structured state
topology that reached 17.06% unseen-N in the small diagnostic.  Continuous
LSD-aligned slots are initialized from paired x/N digits, exchange one generic
left/self/right message, update through a shared per-position GRU conditioned
on same-position N, and decode digits with a shared head.  The state is reused
across requested macrosteps.

There is no arithmetic operation, trace, intermediate target, or fixed reducer
schedule.  Gate: over 1% mean exact and at least 10/512 on both T=1 profiles.
Kill at OOD-N <=2/512 or final train loss above 1.0.  Local run only.

At 60 seconds the card completed 1,953 updates with final train loss 1.0198,
mean exact 0.4167%, and T=1 at 2/512 seen-N plus 1/512 OOD-N.  Both the OOD
kill and the underfit boundary fired.  A 300-second same-data run will determine
whether the topology merely needs optimization or converges to memorization.

At 300 seconds it completed 9,640 updates but final train loss plateaued at
0.8536 rather than the predicted <0.2.  Held-out losses exploded to 7.394 test
and 9.253 OOD; T=1 was 2/512 seen-N and 0/512 OOD-N.  Extra optimization does
not reveal a latent general rule and instead increases specialization.
