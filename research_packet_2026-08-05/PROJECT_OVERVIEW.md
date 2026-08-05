# Project overview

## Goal

Learn repeated modular squaring from `(N, x, T)`:

`x^(2^T) mod N`

The desired algorithmic form is a tied learned transition:

`s0 = x`, `s(t+1) = F_theta(s(t), N)`, then return `s(T)`.

T is an execution count, not an arithmetic value the model should learn.

## Main established evidence

- Parallel decimal token architectures did not generalize modular reduction.
- LSD-first serial GRU diagnostics learned held-out-modulus subtraction,
  comparator gating, canonical identity, and bounded recurrent reduction.
- A clean diagnostic Square -> Reduce recurrence achieved strong held-out-N
  results only when trained with algorithmically relevant intermediate traces.
- The competition-style final-label VDF candidate used a prompt-tail token
  register as its state. It did not learn a transferable transition.
- Final-label curricula and diagnostic trace supervision did not rescue that
  specific token-register architecture on held-out x in the tested envelope.

## Current unresolved question

Is the missing ingredient a genuine latent recurrent workspace, or do these
results instead indicate a bad transition cell, objective, data distribution,
or insufficient scaling?

## Constraints

Competition submissions cannot contain handwritten modular arithmetic, a
closed-form solver, answer lookup, external arithmetic in forward, custom
training loops, evaluator-data manipulation, API keys, or persistent hard-coded
solved weights. Diagnostic data construction may use arithmetic labels only
when clearly marked non-submission research.
