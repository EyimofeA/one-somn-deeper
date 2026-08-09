# Shifted reducer synthesis

## Answer first

The sandbox now contains a learned sublinear modular-reduction mechanism, but
not a competition solution. A learned digit-serial comparator/subtractor can
reduce states with quotients through 99,999,999 on unseen four-digit semiprimes
with at least 99.9023% exactness using 81 fixed opportunities rather than O(q)
unit steps. The experiment directly supervised intermediate arithmetic and its
fixed decimal-shift schedule is not Rule-7 safe.

Evidence: [`../../solving/experiments/2026-08-09_shifted_long_division_reducer/NOTE.md`](../../solving/experiments/2026-08-09_shifted_long_division_reducer/NOTE.md).

## What changed in our model of the problem

Before this gate, public-scale reduction looked computationally infeasible:
the only exact learned reducer took one subtraction per unit quotient. The new
result shows that the learned arithmetic primitive is expressive enough. Its
failure mode was state-support geometry, not parameter capacity: uniform
negative comparison rows almost never include `D-N+r` when `D=N*10^p`, yet
those are precisely the leading-zero states encountered by long division.
Mixing boundary and uniform support repaired both behaviors.

That leaves three distinct unsolved problems:

1. **Squaring:** the best directly supervised four-digit square cell in the
   log is 85.35% held-input exact, not sufficient for composition.
2. **Scheduling:** the successful high-to-low `10^p N` traversal is a
   hard-coded algorithm and cannot be copied into a submission.
3. **Credit assignment:** final modular labels let existing T=1 models fit all
   training rows while learning example-specific codes rather than reusable
   square/reduce states.

## What a credible T=1-solved submission minimally needs

### Evidence-backed requirements

- An LSD-first local state, because learned borrow/carry information is
  position-local and serial.
- Separate immutable `N` and mutable value lanes.
- A tied transition reused across both positions and computation time.
- Boundary-state coverage or an architectural invariant that makes comparison
  near `D` generalize from ordinary examples.
- Exact unseen-N T=1 evaluation before any recurrence-depth claim.

### Fresh legal architecture thesis

Use a generic two-dimensional recurrent grid. Columns are decimal positions;
rows are refinement time. Every cell receives its local mutable state, nearby
cells, the aligned immutable `x/N` embeddings, and a learned phase/pointer
state. The same learned cell updates every location and time. No Python branch
names a multiplication column, shifts `N`, compares values, or subtracts a
chosen divisor. Output digit logits are read from the final mutable row.

This topology can *represent* the discovered algorithm while remaining a
generic learned cellular computation. It does not guarantee SGD will discover
it. The next gate should therefore be final-label T=1 only, with random
initialization, and compare the 2-D grid to a parameter-matched 1-D tied tape.

### Alternative: prefix-register recurrence

Scan `x` from most-significant digit to least and maintain two latent
registers: the prefix and its modular square. This avoids ever materializing
the full raw square, keeping every intermediate near `N`. A handwritten update
would be forbidden, but a generic tied register cell can learn the update. It
is computationally attractive and should be the second architecture arm, not
silently embedded as a solver.

## Next registered experiment

The highest-information next test is a matched final-label T=1 comparison:

- **control:** current generic LSD tape;
- **arm A:** 2-D local grid with learned phase/pointer state;
- **arm B:** two-register prefix scan with the same parameter count and total
  learned-cell applications.

Use public E5 only, three fixed seeds, and the existing 512/512 seen-N/OOD-N
profiles. Promotion requires at least 95% exact on both profiles for every
seed; anything chance-scale remains a failed mechanism, regardless of full
training exactness. Do not submit either arm to Hard before that gate.
