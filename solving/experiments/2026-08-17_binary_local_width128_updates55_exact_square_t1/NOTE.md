# Exact-square reducer with 55 local updates

This changes only recurrent updates from 44 to 55. It tests the observed law
that 11 additional local clocks move the reliable reduction frontier by one
quotient bit.

## Result

The selected step was 10,000: train exact `17.953%`, validation `21.52%`,
seen-x/unseen-N audit `17.46%`, and unseen-x/unseen-N audit `22.06%`.

Raw quotient accuracy is exactly `100%` through `q=16..31`, `32.33%` for
`q=32..63`, and `0.22%` for `q>=64`. The direction confirms that more clocks
extend reduction, but the registered `q=32..63 > 50%` and validation `>23%`
gates failed. Runtime rose to `1065s` for the same examples.

**Conclusion:** brute recurrent depth is causal but too expensive. Keep this as
the clock-law endpoint and move to an architecture that transports high-bit
information faster per learned update.
