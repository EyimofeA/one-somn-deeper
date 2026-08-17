# T=1 quotient-depth diagnostic

This is a read-only interpretation pass over selected checkpoints. It groups
held-out predictions by `floor(x^2 / N)`, the number of modulus-sized chunks
that exact reduction must remove. It also reports whether predicted integers
are valid residues (`prediction < N`) and whether the network copied `x` or
collapsed to zero.

## Prediction

If the local machine has learned an iterative reduction mechanism, exact
accuracy should degrade gradually rather than collapse immediately after
`q=0` or `q=1`, and most predictions should satisfy `0 <= prediction < N`.
Concentration in shallow quotient buckets would identify a shortcut and make
longer recurrence alone a weak next move.

## Result

The unchanged anchor's seen-N validation exact by **centered** quotient was:

- `q=0`: `99.21%`
- `q=1`: `94.50%`
- `q=2..3`: `87.73%`
- `q=4..7`: `59.75%`
- `q=8..15`: `13.54%`
- `q=16..31`: `0.41%`
- `q=32..63`: `0.15%`
- `q>=64`: `0%`

The two unseen-N audits reproduce the same cliff. More than 98% of predictions
are valid residues, and copy/zero collapse is negligible. Thus the model is
not merely emitting malformed integers. It has learned the symmetry
`x ≡ -(N-x) (mod N)` and nearly exact shallow centered cases, but it does not
execute the full-range transition. The raw-quotient recovery at `q>=64`
disappears under centering, confirming the follow-up prediction.

The delayed-decay checkpoint changes bucket details but not this mechanism.
Optimizer polish is therefore secondary to extending arithmetic depth.

### Exact-square isolation

When the 22-bit square is supplied directly, seen-N validation accuracy is
roughly `47-58%` for centered quotient buckets through `q=8..15`, falls to
`34.22%` at `q=16..31`, then collapses to `0.30%` at `q=32..63` and `0.19%`
at `q>=64`. Reduction therefore has its own sharp horizon near the fifth
quotient bit. The fused model additionally struggles to create large squares,
but removing squaring does not remove this reduction boundary.

Increasing clocks shifts that boundary: 44 updates make `q=16..31` nearly
exact and reach `9.86%` at `q=32..63`; 55 updates keep `q<32` exact and reach
`32.33%` at `q=32..63`. Simple cyclic dilation, fixed mixed messages, learned
mixed messages, and learned scratch-lane messages do not reproduce this
frontier shift. The strongest mechanistic conclusion is therefore that the
local ConvGRU has learned a real bit-serial reducer, but each additional
quotient bit consumes substantial recurrent time. Faster communication alone
does not implement the missing conditional compare/subtract operation.

## Follow-up prediction

The raw quotient slice unexpectedly recovers to about 10% for `q>=64` after
collapsing at `q=16..63`. We predict this is because examples with `x` close to
`N` have a small equivalent signed representative `N-x`. Regrouping by
`floor(min(x, N-x)^2 / N)` should remove the recovery and expose a mostly
monotone depth curve.
