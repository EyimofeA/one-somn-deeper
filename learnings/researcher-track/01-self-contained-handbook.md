# One Layer Deeper: a self-contained researcher handbook

## 1. What the researcher actually owns

The coding agent can implement a model. The researcher must decide:

1. what ability is currently missing;
2. which observation supports that diagnosis;
3. what single change would distinguish competing explanations;
4. what outcome would falsify the idea;
5. whether the evidence is diagnostic, competition-legal, or hosted;
6. whether another run is worth its time, money, and quota.

If these decisions are delegated, faster code produces faster confusion.

## 2. The mathematics

For the public modular-squaring task, start with:

```text
s₀ = x mod N
sₜ₊₁ = sₜ² mod N
```

The requested public answer is `s at step T = x raised to 2^T, modulo N`.
Congruence explains why
we may reduce after every step:

```text
a ≡ b (mod N)  implies  a² ≡ b² (mod N)
```

Example: `N = 77`, `x = 38`. Since `38² = 1444 = 18×77 + 58`,
we get `s₁ = 58`. Then `58² = 3364 = 43×77 + 53`, so `s₂ = 53`.

Do not silently assume the hidden Hard task is repeated squaring. The public
rules warn that aspects of the recurrence may change. A Hard-relevant model
therefore needs a generic learned executor, not a handwritten squaring solver.

### What one decimal step requires

Writing a decimal number as `x = Σᵢ aᵢ10ⁱ`, raw square column `k` contains:

```text
uₖ = cₖ + Σ(i+j=k) aᵢaⱼ
```

The emitted digit is `dₖ = uₖ mod 10`. The next carry is the integer part of
`uₖ / 10`. Reduction then needs comparison, quotient
selection, subtraction, borrow propagation, and canonicalization into
the interval `[0, N)`. Middle columns are hardest because they combine the most products.

The target reveals only the remainder `r` in `x² = qN + r`. It hides the raw
square, quotient, carries, comparisons, and subtraction path. That missing
supervision is the core identifiability problem.

## 3. Data and what each split means

- **Training exactness:** can the model fit observed rows?
- **Held-out x, seen N:** can it predict new inputs for a modulus it knows?
- **Unseen N:** can it transfer a shared procedure to a new function `f_N`?
- **Longer width:** can it apply the procedure beyond trained sequence length?
- **Larger T:** can its one-step state transition roll out without drift?
- **Hosted Hard:** can it learn the hidden task inside the evaluator budget?

Unseen x and unseen N are not equivalent. With a fixed N, a model can partially
memorize one finite table. A new N moves every quotient boundary. The public e5
audit found sparse coverage across only 27 training moduli, making
modulus-specific shortcuts cheap. Evidence:
[`../experiments/2026-08-10_e5_support_audit/NOTE.md`](../../solving/experiments/2026-08-10_e5_support_audit/NOTE.md).

Exact sequence accuracy is also harsher than digit accuracy. If four digits
were independently 90% correct, whole-answer accuracy would be only
`0.9⁴ = 65.61%`. Arithmetic errors are correlated, but the example explains
why “good token accuracy” is not enough.

## 4. What common model families can and cannot do

### MLP

An MLP can approximate finite mappings and fit training tables. It has no
native notion of digit position, message passing, or a reusable algorithm.
Our matched MLP fit training perfectly and stayed near 4% unseen-N. Use MLPs as
memorization controls, not default arithmetic solvers.

### Transformer

Attention makes every token accessible and can learn routing. It does not force
the network to multiply, carry, reduce, or reuse one transition. Our matched
Transformer also fit perfectly and stayed near 4% unseen-N. A Transformer is a
strong encoder/control component, not proof of algorithm learning.

### RNN, GRU, and LSTM

These reuse a transition over time, so they can represent scans, carries, and
finite-state algorithms. Their bottlenecks are compressed state, long gradient
paths, exposure bias, and sequential throughput. GRU/LSTM gates help retain
state; they do not identify which state variable should represent carry.

### Universal Transformer and recurrent Transformer

These reuse an attention block across computational depth. They combine global
routing with tied computation, but fixed wall-clock budgets trade more internal
steps for fewer optimizer updates. They still need an appropriate state and
objective.

### Neural GPU and cellular models

They maintain spatial state and reuse local updates. Information moves a
bounded distance each microstep, which aligns with carry/borrow waves. Our
generic Neural GPU learned correlations but failed exact raw squaring,
especially centrally. More recurrence alone did not fix the primitive.

### NAC, NALU, NAU, NMU, and bilinear cells

These bias scalar units toward addition or multiplication. They may help form
digit cross-products, but they do not automatically route columns, propagate
carry, reduce modulo a changing N, or emit exact decimal strings. The safest
project use is a generic multiplicative interaction inside a structured,
tied recurrent cell, tested against a matched affine control. See
[`../readings/neural-arithmetic-models.md`](../readings/neural-arithmetic-models.md).

### Program learners

Neural Programmer-Interpreters and trace-trained executors show that reusable
subroutines can be learned with execution supervision. They establish
capability, but their trace labels may be illegal or unavailable in the hosted
competition. Never transfer a diagnostic result into a competition claim.

## 5. Representation is a computational contract

A representation should answer:

- Where is each immutable input digit stored?
- Where is mutable state stored?
- Which direction can information travel?
- How many microsteps give a full receptive field?
- How does output position i read state position i?
- Is state preserved between recurrent task steps?

Decimal is not inherently bad: exact local decimal algorithms exist. Binary or
limb tokenization changes widths and carry frequency, but simple swaps did not
solve our task. A pooled global vector destroys location; a tape preserves it.
Our rushed causal-message Hard candidate then mean-pooled its tape during
decoding, illustrating how an architecture can build useful structure and
discard it at the final interface.

## 6. Losses: what signal are you adding?

### Token cross-entropy

The basic legal loss is:

```text
L = -Σ(answer positions j) log p(correct token yⱼ)
```

It is differentiable and stable but rewards individual digits, not exact
answers or algorithmic state.

### Sequence-balanced token loss

Average per example before averaging the batch. This prevents longer answers
from contributing more simply because they contain more tokens.

### Worst-digit or smooth-max loss

Emphasize the weakest answer position. It can address edge-versus-middle
imbalance, but our hosted/local controls did not repair unseen-N transfer.

### Exact-match surrogates and margins

Increase pressure for every answer digit to be correct. They may improve
calibration but cannot reveal hidden carries or quotients. A stronger objective
on the wrong representation still selects memorizing solutions.

### Auxiliary arithmetic losses

Carry, raw square, quotient, comparison, and trace losses are powerful
diagnostics. They answer “can this architecture represent the procedure?” They
are not legal final-label evidence unless the evaluator provides those targets.

### Late-step final-answer supervision

Predict the same legal final answer at several late recurrent microsteps. This
shortens gradient paths without adding arithmetic labels. It should be tested
only after a recurrent core demonstrates capability, and against a final-only
control.

### Group DRO and reweighting

Optimize the worst modulus-size or difficulty group. Useful when average loss
hides a weak group; ineffective if every group is solved through a shortcut.

The rule is: a loss is a hypothesis about missing credit. Change it alone,
predict which curve or subgroup changes, and specify the falsifier.

## 7. Optimizers and schedules

### AdamW

Default first. It adapts per-parameter step sizes and separates weight decay.
Tune learning rate, batch size, warmup, and decay before exotic optimizers.

### SGD

Provides a useful simplicity/control baseline and sometimes a different
implicit bias, but may train slowly on deep recurrent systems.

### Muon

Can improve updates for matrix parameters, but our optimizer controls did not
show that optimizer choice was the central arithmetic mechanism. Treat it as a
throughput/optimization hypothesis, not an algorithm.

### SAM

Penalizes sharp neighborhoods through an additional perturbation step. It costs
extra forwards/backwards and did not create Hard T=1 transfer here. Flatness is
not equivalent to the correct program.

### Weight decay and grokking

Weight decay can favor simpler solutions and sometimes precedes grokking. A
longer run is justified only by a registered signal: continuing train fit,
falling validation loss, a complexity transition, or a known grokking regime.
Our worsening held-out loss and long negative controls argue against “just wait”
for tested models.

### Wall-clock schedules

Hosted budgets are time, not update, constrained. More model depth or recurrent
steps may reduce the number of optimizer updates dramatically. Always report
both seconds and completed updates. A theoretically stronger model that sees
one tenth as many batches may be worse in competition.

## 8. How to read training curves

- Train loss down, validation loss down: learning supported structure.
- Train loss down, validation loss up: memorization or distribution mismatch.
- Both flat: optimization, bug, or insufficient updates.
- Token accuracy high, exact low: localized or correlated digit failures.
- Seen-N improves, unseen-N flat: modulus shortcut.
- T=1 strong, larger T collapses: recurrence stability/exposure bias.
- Local strong, hosted weak: throughput/configuration/evaluator mismatch.

Never report only the final loss. Record train exact, held-out-x exact,
unseen-N exact, per-digit accuracy, errors by difficulty, updates, time, and
seed variability.

## 9. The experiment loop

1. **State the current bottleneck.** Example: product formation versus message
   consumption.
2. **Name competing explanations.** At least two.
3. **Choose one changed variable.** Everything else is an anchor.
4. **Preregister.** Change, predicted curve, falsifier, budget, kill gate.
5. **Run CPU smoke tests.** Shapes, gradients, tiny overfit, deterministic eval.
6. **Run seed 0.** Stop at the registered gate.
7. **Replicate only a promoted result.** Seeds test robustness, not rescue.
8. **Intervene.** Ablate the claimed mechanism, with matched damage control.
9. **Classify evidence.** Diagnostic, legal-shaped, local mirror, or hosted.
10. **Write the result before starting another card.** Negative results close
    branches only at the tested scope.

The next scientific sequence should be: bilinear versus affine raw-square
cell; then causal versus terminal messages; then final-label T=1; only then
recurrence.

## 10. Code anatomy

A competition submission exposes:

- `build_model(ModelSpec)`: create random trainable state within limits;
- `forward(input_ids, attention_mask)`: produce token logits and optional legal
  auxiliary outputs;
- `token_training_loss(TokenLossBatch)`: evaluator-called differentiable loss;
- `build_optimizer(model, OptimizerSpec)`: optimizer and optional schedule;
- `SUBMISSION`: batch sizes and registered callbacks.

Start audits at tensor shapes. Write them beside every state:

```text
tokens:   [batch, sequence]
hidden:   [batch, position, channels]
messages: [batch, position, direction, message_channels]
logits:   [batch, sequence, vocabulary]
```

Then trace one example forward, one gradient backward, and one optimizer step.
Check that a claimed channel actually changes downstream logits. Parameter
existence is not parameter use.

## 11. Recurrence—later, not now

Suppose one-step error probability is (e). Later states are not independent:
one wrong state can place all following inputs off distribution. Teacher-forced
one-step accuracy therefore overestimates autonomous rollout.

After T=1 is near exact, test:

1. teacher-forced one-step transition;
2. autonomous T=2;
3. T=4 and T=8;
4. state drift and confidence over steps;
5. training-depth versus evaluation-depth extrapolation;
6. whether the same weights implement every step;
7. generic recurrence families if Hard transfer is the goal.

Possible tools include scheduled sampling, state noise, contractive/stable
transitions, tied weights, recurrent state normalization, and late-answer
losses. Each addresses a different failure and must be isolated.

## 12. Competition legality and provenance

Forbidden shortcuts include hard-coded solvers, task-value modulus operations,
dataset inspection, arithmetic traces in a submission, participant-controlled
backward, pre-solved weights, CPU offload of model state, and metric exploits.
Read the live rules before every upload.

A SHA-1 is only a source fingerprint. It answers “which exact file ran?” It is
not a model concept. Before quota: validate source, scan forbidden operations,
compare the fingerprint with prior uploads, freeze config, record prediction,
and obtain owner approval. Never infer novelty from a new filename.

## 13. Managing agents without surrendering research

Give an agent a bounded role: implementer, auditor, literature reviewer, or
experiment executor. Do not ask “solve the competition.” Supply:

```text
QUESTION: the one causal question
ANCHOR: exact source/config/metrics
CHANGE: one allowed variable
KEEP FIXED: representation, optimizer, data, budget, seed protocol
PREDICTION: expected curves and subgroup effect
FALSIFIER: numerical stop condition
OUTPUTS: code, tests, config, logs, plot, NOTE
FORBIDDEN: rule violations and unapproved runs/submissions
```

Require the agent to first restate the causal contrast and list confounders.
Reject code until it passes. A weaker model can execute well when the contract
is narrow, evidence is explicit, and promotion is mechanical.

For parallel agents, assign non-overlapping artifacts and one owner per file.
Ask one agent to implement and another to audit only when the review is truly
independent. If a requested model is unavailable, record that fact; do not
silently substitute another model.

## 14. Compute operations

Before GPU rental, require CPU smoke, frozen config, time budget, kill gate,
artifact paths, and backup plan. During a run, monitor loss, exactness, updates,
throughput, GPU utilization, and cost. Afterward, copy source, logs, reports,
plots, and checkpoints; verify file counts and bytes; only then terminate the
identified pod. Hosted evaluation and rented GPU are separate systems.

## 15. What our evidence currently supports

- Generic MLP/Transformer capacity fits but does not discover unseen-N
  arithmetic.
- Explicit state topology improves transfer modestly, not qualitatively.
- Trace-supervised arithmetic primitives are learnable.
- Simple representation, optimizer, duration, workspace, and local-convolution
  changes did not solve legal T=1.
- Central-digit failure does not yet distinguish product accumulation from
  carry consumption.
- The next clean question is a matched bilinear product-forming cell; then a
  matched causal-message intervention.
- No new competition or GPU expenditure is scientifically justified until the
  corresponding research gate passes.

## 16. The researcher's daily checklist

At session start, write: what shipped, the current bottleneck, and at most three
actions. At session close, mark each done or give the exact blocker. If reading
or planning has replaced a chosen deliverable, name that explicitly. If a run
cannot change tomorrow's decision, do not run it.
