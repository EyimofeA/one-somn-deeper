# Full VDF pipeline audit

Date: 2026-08-04. Sources: public competition generator, manifests, and
controlled reducer artifacts only. No hidden data is read.

## The end-to-end computation that a legal model must learn

For prompt `(N, x, T)`, the target is the result of applying the same learned
one-step map T times:

```text
s0 = x
s(t+1) = Reduce(Square(st, N), N)
y = sT
```

`Square` and `Reduce` must both be learned modules in the submitted forward.
Parsing prompt tokens and using T only as an iteration count is compatible with
the public recurrence rule; calculating `x²`, a quotient, factors, or a modular
answer outside learned state is not.

## What is established

The serial reducer diagnostic establishes a learned unit transition only after
the raw nonnegative value `u=qN+r` is already supplied. Its learned comparator
and LSD-first subtractor are strong on unseen four-digit moduli when trained on
full q=0..100 state support. This is evidence for reduction, not an end-to-end
VDF step.

The per-position hierarchical controller is also established:

```text
state -> learned per-position controller -> k in [0,15]
      -> frozen learned unit reducer applied k times -> next state
```

At unseen-N q=100 it has 100% selected-k and macro-transition exact, 99.51%
terminal exact, and 7.50 mean outer decisions. It still makes 102.05 mean
inner unit-reducer calls. It therefore compresses *control decisions*, not
arithmetic computation.

## Public-scale implication

For a one-step square, public inputs obey `1 <= x < N`; hence
`q=floor(x²/N) <= N-2`. A true unit reducer needs O(q) learned transitions for
each squaring. Medium m4 includes 22-bit N, so a single square can require up
to 4,194,302 reductions and an approximately 14-decimal-digit raw-square
state. Repeating the VDF T times compounds this cost over all intermediate
states.

The hierarchical controller does not change that asymptotic cost, because a
chosen k invokes k unit transitions. It cannot yet be integrated as a
Medium-scale accelerator.

## Current missing bridge

1. A legal learned `Square(state,N)` that generalizes on unseen x/N.
2. A compositional learned macro-reduction transition that produces `F^k`
   exactly in substantially less than k unit calls.
3. Composition of the resulting learned VDF step for prompt T.

The direct and action-conditioned one-call macro decoders are both refuted.
The latter receives the correct learned chunk action and still has 0% unseen-N
q=100 macro-transition exact. Further controller tuning is not the path: the
next branch must change the state-transition representation while retaining a
learned end-to-end operation.
