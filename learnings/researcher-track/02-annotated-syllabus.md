# Annotated syllabus: learn from papers and prior experiments

This version is for learning through external sources. Read only with a written
question. For every source, produce the listed artifact; highlighting a PDF is
not completion.

## Module 0 — orient to this exact project

### Read

- [`../../learnings/concepts/01-the-problem.md`](../concepts/01-the-problem.md)
- [`../../solving/STATUS.md`](../../solving/STATUS.md)
- [`../../RESEARCH_PROTOCOL.md`](../../RESEARCH_PROTOCOL.md)
- [`../../competition/README.md`](../../competition/README.md)

### Extract

Write one page distinguishing the public mathematical task, hidden Hard
uncertainty, evaluator training contract, exact-rung scoring, and prohibited
shortcuts. Draw the input/output token sequence.

### Test

Explain why a trace-supervised reducer can be scientifically valuable but not
a legal submission.

## Module 1 — modular arithmetic and exact integer procedures

### Primary learning sources

- [MIT Mathematics for Computer Science, modular arithmetic](https://courses.csail.mit.edu/6.042/spring18/mcs.pdf), number theory chapters.
- [MIT 6.006 Integer Arithmetic lecture](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/resources/lecture-11-integer-arithmetic-karatsuba-multiplication/).
- [Handbook of Applied Cryptography, Chapter 14](https://cacr.uwaterloo.ca/hac/about/chap14.pdf), efficient arithmetic background.

### Extract

By hand, compute two T=4 trajectories. For one four-digit x, list every
schoolbook square column, carry, long-division comparison, subtraction, and
borrow. Label which variables final remainder supervision hides.

### Why it matters

Without the actual dependency graph, words like “arithmetic bias” are empty.
You must know whether an architecture is helping product formation, routing,
carry, quotient selection, subtraction, or rollout.

## Module 2 — supervised learning, generalization, and identifiability

### Sources

- [Deep Learning, Chapter 5](https://www.deeplearningbook.org/contents/ml.html), capacity and generalization.
- [Understanding Deep Learning Requires Rethinking Generalization](https://arxiv.org/abs/1611.03530), memorization controls.
- [Grokking](https://arxiv.org/abs/2201.02177), delayed generalization on algorithmic data.

### Repository evidence

- [`../experiments/2026-08-10_x2modn_direct_mlp/NOTE.md`](../../solving/experiments/2026-08-10_x2modn_direct_mlp/NOTE.md)
- [`../experiments/2026-08-10_x2modn_direct_transformer/NOTE.md`](../../solving/experiments/2026-08-10_x2modn_direct_transformer/NOTE.md)
- [`../experiments/2026-08-10_e5_support_audit/NOTE.md`](../../solving/experiments/2026-08-10_e5_support_audit/NOTE.md)

### Extract

Draw hypothetical train/test curves for memorization, successful algorithm
learning, delayed grokking, underfitting, and distribution shift. Then classify
our curves and state what new observation would change your classification.

## Module 3 — MLPs, attention, and representations

### Sources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762), attention and positional representations.
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/), executable teaching implementation.
- [Universal Transformers](https://arxiv.org/abs/1807.03819), tied computation over depth.

### Extract

Implement one pre-norm Transformer block from tensor operations. Annotate the
shape after embeddings, Q/K/V projection, attention, residual, MLP, and logits.
Prove what weight tying changes—and what it does not force the block to learn.

### Project question

Why did global access help fit while failing unseen N? List three shortcut
solutions attention can represent more cheaply than modular arithmetic.

## Module 4 — recurrent neural networks and state

### Sources

- [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf), original LSTM.
- [GRU introduction in Learning Phrase Representations](https://arxiv.org/abs/1406.1078).
- [Understanding the exploding gradient problem](https://proceedings.mlr.press/v28/pascanu13.html).
- [S4: Efficiently Modeling Long Sequences](https://arxiv.org/abs/2111.00396), structured state spaces.

### Extract

Implement a scalar RNN, GRU cell, and LSTM cell on CPU. Feed the same delayed
copy task. Plot hidden-state and gradient norms. Explain why gating aids memory
but does not assign “carry” semantics.

### Project question

For a radius-one recurrent tape, calculate the receptive field after K steps.
Then calculate how increasing K changes hosted optimizer-update throughput.

## Module 5 — neural arithmetic units

### Sources

- [Neural Arithmetic Logic Units](https://arxiv.org/abs/1808.00508).
- [Neural Arithmetic Units](https://arxiv.org/abs/2001.05016).
- [Neural Multiplication Unit](https://arxiv.org/abs/2006.01681).
- [`../readings/neural-arithmetic-models.md`](../readings/neural-arithmetic-models.md).

### Extract

Implement MLP/NAC/NAU on addition and MLP/NALU/NMU on multiplication. Train on
a small interval and test on disjoint, larger ranges, zeros, and negative
values. Report five seeds, not the best seed. Inspect learned effective weights.

### Project question

Explain why scalar multiplication extrapolation does not imply correct decimal
column routing or modular reduction. Design a generic bilinear cell and its
parameter-matched affine control.

## Module 6 — neural algorithm execution

### Sources

- [Neural GPUs Learn Algorithms](https://arxiv.org/abs/1511.08228).
- [Neural Programmer-Interpreters](https://arxiv.org/abs/1511.06279).
- [Neural Algorithmic Reasoning](https://arxiv.org/abs/2105.02761).
- [CLRS Algorithmic Reasoning Benchmark](https://arxiv.org/abs/2205.15659).

### Repository evidence

- [`../experiments/2026-08-10_multilane_neural_gpu_square/NOTE.md`](../../solving/experiments/2026-08-10_multilane_neural_gpu_square/NOTE.md)
- [`../experiments/2026-08-09_shifted_long_division_reducer/NOTE.md`](../../solving/experiments/2026-08-09_shifted_long_division_reducer/NOTE.md)
- [`../experiments/2026-08-10_x2modn_sanity_seed0/NOTE.md`](../../solving/experiments/2026-08-10_x2modn_sanity_seed0/NOTE.md)

### Extract

Create an evidence table with columns: supervision, state, learned primitive,
length/range extrapolation, legal status, and remaining failure. Distinguish
representing an algorithm from identifying it through final labels.

## Module 7 — objectives and optimization

### Sources

- [Adam](https://arxiv.org/abs/1412.6980) and
  [Decoupled Weight Decay/AdamW](https://arxiv.org/abs/1711.05101).
- [Sharpness-Aware Minimization](https://arxiv.org/abs/2010.01412).
- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599), probability quality versus accuracy.

### Repository evidence

- [`../experiments/2026-08-10_canonical_worst_digit_loss/NOTE.md`](../../solving/experiments/2026-08-10_canonical_worst_digit_loss/NOTE.md)
- [`../experiments/2026-08-10_canonical_modulus_length_group_dro/NOTE.md`](../../solving/experiments/2026-08-10_canonical_modulus_length_group_dro/NOTE.md)
- [`../experiments/2026-08-05_fable_muon_adamw_e1_optimizer_control/NOTE.md`](../../solving/experiments/2026-08-05_fable_muon_adamw_e1_optimizer_control/NOTE.md)

### Extract

Derive token cross-entropy. Implement sequence-balanced CE, smooth worst-digit
loss, and late-step final-answer loss on fake logits. For each, state the exact
credit-assignment hypothesis and a result that would refute it. Explain why an
optimizer comparison must hold model, batch, data order, and budget fixed.

## Module 8 — causality and mechanistic claims

### Sources

- [Amnesic Probing](https://arxiv.org/abs/2006.00979), behavioral counterfactuals.
- [Causal Scrubbing](https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing), intervention-based interpretability methodology.
- [`../readings/gpt5-pro-causal-message-verdict-2026-08-12.md`](../readings/gpt5-pro-causal-message-verdict-2026-08-12.md).

### Extract

Build an RNN with a perfectly probe-decodable hidden stream disconnected from
the output. Demonstrate that probe accuracy does not imply causal use. Then
design zero, delay, swap, and matched-channel interventions for arithmetic
messages.

## Module 9 — experimental design and statistics

### Sources

- [NIST Engineering Statistics Handbook: experimental design](https://www.itl.nist.gov/div898/handbook/pri/pri.htm).
- [Statistical Rethinking lectures and text resources](https://xcelab.net/rm/), for uncertainty and model criticism.

### Extract

For one proposed card, write the estimand, anchor, one change, primary metric,
secondary diagnostics, seed protocol, stopping rule, promotion rule, and
multiple-comparison risk. Explain why “best of twenty seeds” is not evidence.

## Module 10 — recurrence and autonomous rollout

### Sources

- [Scheduled Sampling](https://arxiv.org/abs/1506.03099), exposure bias.
- [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377), tied computation and fixed points.
- [`../concepts/17-recurrence-generalisation.md`](../concepts/17-recurrence-generalisation.md).

### Extract

Train a toy one-step transition under teacher forcing, then evaluate autonomous
rollout. Plot error versus depth and state drift. Compare scheduled sampling,
state noise, and contractive regularization as separate hypotheses.

Do this module only after the T=1 mastery and research gates.

## Module 11 — competition and compute operations

### Read

- [`../../PITFALLS.md`](../../PITFALLS.md)
- [`../../solving/experiments/OPS.md`](../../solving/experiments/OPS.md)
- [`../../solving/experiments/LAYOUT.md`](../../solving/experiments/LAYOUT.md)
- [`../../scripts/README.md`](../../scripts/README.md)

### Extract

Conduct a fake submission drill: prediction, source audit, SHA comparison,
validation, mock Easy result, interpretation, commit, and no-upload stopping
decision. Conduct a fake GPU lifecycle: identity verification, command/log
capture, backup manifest, byte-count verification, and safe termination plan.

## Module 12 — directing agents

### Read

- [`../../AGENTS.md`](../../AGENTS.md)
- [`../../RESEARCH_PROTOCOL.md`](../../RESEARCH_PROTOCOL.md)
- [`../../MODEL_REVIEW_SYNTHESIS.md`](../../MODEL_REVIEW_SYNTHESIS.md)

### Extract

Give a weak model one bounded implementation card. Before accepting code, make
it restate the causal question, fixed variables, forbidden actions, expected
tensor shapes, tests, and falsifier. Run a separate audit prompt against the
diff. Grade both with the practicum rubric.

## Completion standard

Reading is complete only when you can reconstruct the argument without the
paper, connect it to one repository result, name one limitation, and design a
falsifying experiment. The practicum—not page count—decides readiness.

