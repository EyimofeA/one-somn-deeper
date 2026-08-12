# Neural arithmetic models: what matters for One Layer Deeper

## The missing map

“Neural arithmetic” names several different ideas. They should not be treated
as interchangeable.

1. **Arithmetic layers** bias a network toward scalar `+`, `-`, `*`, or `/`.
2. **Algorithm learners** reuse a state transition across positions or time.
3. **Program learners** learn or receive a decomposition into subroutines.

Our task needs all the first two abilities and exact decimal state management.
Program traces are useful diagnostics but generally not legal submission
supervision.

## NAC and NALU

The [Neural Arithmetic Logic Unit](https://arxiv.org/abs/1808.00508) begins with
a Neural Accumulator (NAC). Its effective matrix is

\[
W=\tanh(\hat W)\odot\sigma(\hat M),\qquad y=Wx.
\]

This encourages weights near `-1`, `0`, or `1`, making addition and subtraction
more natural than an unconstrained dense layer. NALU combines an additive NAC
with a multiplicative log/exp branch and a learned gate.

Why it is attractive: if the learned operation really is `a+b` or `a*b`, its
behavior can extrapolate beyond the training range.

Why it is not our solution: logarithmic multiplication is awkward around zero
and negative values, gates and constrained weights can optimize poorly, and
`x² mod N` is not one smooth scalar operation. Modulus introduces moving,
discontinuous quotient boundaries.

## Later arithmetic units

The [Neural Arithmetic Units](https://arxiv.org/abs/2001.05016) paper gives a
cleaner evaluation methodology and shows that successful interpolation does
not imply arithmetic extrapolation. Its NAU makes additive selection more
explicit. The [Neural Multiplication Unit](https://arxiv.org/abs/2006.01681)
uses products of softly selected inputs, avoiding NALU’s log/exp path for
non-negative multiplication.

These units are best viewed as **small inductive-bias components**. In our task,
a generic bilinear interaction is probably safer:

\[
\Delta h_i=U[(Vh_i)\odot(Wc_i)].
\]

It cheaply represents second-order interactions such as digit cross-products
without hard-coding decimal multiplication. It must be compared with a
parameter/FLOP-matched affine or GLU cell.

## Algorithm learners

The [Neural GPU](https://arxiv.org/abs/1511.08228) repeatedly applies tied local
convolutions to a spatial state. This supports carry-like waves and length
generalization. Our raw-square Neural GPU nevertheless reached only 4% unseen
exact, with middle-digit collapse; a terminal carry target improved 3.85% to
6.25%, not an algorithmic transition. Evidence:
[`2026-08-10_multilane_neural_gpu_square/NOTE.md`](../../solving/experiments/2026-08-10_multilane_neural_gpu_square/NOTE.md)
and
[`2026-08-10_multilane_neural_gpu_square_carry_50k/NOTE.md`](../../solving/experiments/2026-08-10_multilane_neural_gpu_square_carry_50k/NOTE.md).

[Universal Transformers](https://arxiv.org/abs/1807.03819) similarly reuse a
block over computational depth, but global attention alone did not make our
matched Transformer learn modular arithmetic: it fit train exactly and stayed
near 4% unseen-N. Evidence:
[`2026-08-10_x2modn_direct_transformer/NOTE.md`](../../solving/experiments/2026-08-10_x2modn_direct_transformer/NOTE.md).

The [Neural Programmer-Interpreter](https://arxiv.org/abs/1511.06279) learns
reusable subprograms from execution traces. It is conceptually important—state,
control, and reusable procedures—but its trace-rich training is diagnostic,
not directly transferable to final-label competition training.

## Research-worthy architecture

Use an LSD-aligned recurrent tape with, at every decimal position:

- immutable embeddings for local `x` and `N` digits;
- mutable content state `h`;
- low-dimensional left/right messages `m`;
- a generic bilinear/NMU-like interaction for product formation;
- one tied update cell;
- position-specific decoding.

Do **not** name a channel carry, schedule schoolbook phases, or insert modulus.
First compare the bilinear cell to a matched affine cell on raw-square final
labels. If it clears the central-digit gate, compare causally consumed messages
with terminal-only messages. Only then train final-label `x² mod N`.

## Reading and implementation order

1. NALU paper: understand constrained additive and multiplicative biases.
2. NAU/NMU papers: study failure criteria and extrapolation evaluation.
3. Implement scalar `a+b` and `a*b` range-extrapolation notebooks on CPU.
4. Neural GPU: understand tied local state and receptive-field growth.
5. Implement a digit-aligned raw-square model with a bilinear cell and matched
   affine control—no reduction yet.
6. Add causal-message versus terminal-message controls and interventions.
7. Study [Neural Algorithmic Reasoning](https://arxiv.org/abs/2105.02761) for
   the broader distinction between fitting outputs and executing procedures.

Mastery means being able to explain why a unit that extrapolates multiplication
can still fail modular squaring: it may form products but lack column routing,
carry state, reduction, exact canonicalization, or legal final-label credit.

