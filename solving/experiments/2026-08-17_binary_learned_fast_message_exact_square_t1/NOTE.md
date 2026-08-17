# Zero-initialized learned fast-message gates

This replaces destructive fixed long-range message scales with three learned
scalars initialized at zero. The initial forward pass is the known local
reducer, and optimization can open only the distances it finds useful.

## Result

The selected step was 8,000: train exact `14.186%`, validation `17.08%`,
seen-x/unseen-N audit `13.34%`, and unseen-x/unseen-N audit `17.80%`.

This recovers the ordinary local reducer (`17.10%`, `13.50%`, `17.82%`) but
does not improve it. Promotion gates failed. Gate magnitudes and quotient
slices are inspected separately to distinguish "correctly stayed shut" from
"opened into a non-algorithmic shortcut."

The gates did open to `[0.0878, 0.0294, 0.0392]` for distances `[2,4,8]`, but
raw `q=32..63` moved only from `0.27%` to `0.55%`. The mixed messages influence
optimization without extending the algorithmic frontier. A final card routes
them into the unused scratch lane instead of adding them to all state.
