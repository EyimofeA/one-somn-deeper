# Cyclic dilated transport, exact-square reducer

The same tied ConvGRU kernels are applied at horizontal dilations 1, 2, 4, and
8 on successive clocks. This keeps the parameter count fixed while testing
whether logarithmic bit transport can replace brute recurrent depth.

## Result

The selected step was 10,000: train exact `12.314%`, validation `14.74%`,
seen-x/unseen-N audit `11.02%`, and unseen-x/unseen-N audit `15.00%`.

**Reject.** Replacing every local neighborhood with dilation cycling reduced
validation below the ordinary 33-clock reducer's `17.10%`. Long-range reach
alone is insufficient; the reducer needs uninterrupted local computation.
The next card keeps the ordinary local cell on every clock and injects a
parameter-free long-range message only every fourth clock.
