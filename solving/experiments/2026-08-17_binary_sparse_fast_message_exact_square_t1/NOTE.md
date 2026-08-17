# Local reducer plus sparse fast messages

The ordinary local ConvGRU remains active every clock. Every fourth clock adds
a fixed residual message from bit offsets 2, 4, and 8 in a cycle. This tests
whether sparse long-range communication can extend the quotient frontier
without replacing the local arithmetic pathway or adding parameters.

## Result

The selected step was 8,500: train exact `6.597%`, validation `8.32%`,
seen-x/unseen-N audit `5.54%`, and unseen-x/unseen-N audit `8.84%`.

**Reject strongly.** The fixed messages more than halved validation relative
to the ordinary local reducer and never recovered. Parameter-free long-range
mixing corrupts the local state; fast communication must be gated or written
to a separate lane.
