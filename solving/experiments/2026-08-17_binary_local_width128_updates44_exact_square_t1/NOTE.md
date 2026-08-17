# Exact-square reducer with 44 local updates

This changes only recurrent updates from 33 to 44 in the matched exact-square
isolation. It tests whether the reduction cliff at quotient 32 is caused by
insufficient local transport time.

## Result

The selected step was 10,000: train exact `16.474%`, validation `19.92%`,
seen-x/unseen-N audit `16.04%`, and unseen-x/unseen-N audit `20.22%`.

The raw quotient diagnostic is decisive. The 33-update reducer was nearly
exact through `q=8..15`, `62.45%` at `q=16..31`, and `0.27%` at `q=32..63`.
With 44 updates it is nearly exact through `q=16..31` and reaches `9.86%` at
`q=32..63`. The extra 11 clocks moved the reliable frontier by one quotient
bit. This supports a bit-serial shifted-reduction mechanism whose generality is
limited by recurrent clock depth, not parameter count alone.

The aggregate validation prediction narrowly missed 20% (`19.92%`), while the
mechanistic `q=32..63 > 5%` gate passed. Retain as evidence for the recurrence
horizon, not as a competition-ready model: its wall time was `837s` versus
`549s` for 33 updates.
