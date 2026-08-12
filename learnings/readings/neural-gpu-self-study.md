# Neural GPUs: a self-study course for arithmetic research

## What you should be able to do afterward

By the end, you should be able to:

- explain a Neural GPU without saying only “an RNN with convolutions”;
- calculate its tensor shapes, receptive field, and compute cost;
- implement a small version from scratch;
- distinguish representational capacity from learnability;
- tell whether a model stores, transmits, and causally uses arithmetic state;
- design controls that isolate product formation, carry transport, and
  recurrent stability;
- audit an agent-generated Neural GPU implementation;
- explain why our Neural GPU failed and what evidence would reopen the branch.

The primary paper is [Neural GPUs Learn Algorithms](https://arxiv.org/abs/1511.08228)
by Łukasz Kaiser and Ilya Sutskever. Read this guide first, then the paper.

## 1. Begin with the problem an ordinary RNN has

Suppose the input is a decimal string with `L` digit positions. An ordinary
RNN compresses the prefix into one vector:

```text
digit₀ → hidden₀ → digit₁ → hidden₁ → ... → hidden_L
```

That can implement a left-to-right scan, but exact arithmetic may need several
pieces of information at different positions simultaneously. Compressing all
of them into one vector can be destructive. It is also sequential: position
`i+1` waits for position `i`.

A Neural GPU instead maintains a spatial memory:

```text
position:    0       1       2       3       ...
state:      h₀      h₁      h₂      h₃
```

Every position updates in parallel by reading itself and nearby positions.
The same update rule is reused across space and computational time.

## 2. The core tensor

For a one-dimensional teaching version, use:

```text
state shape = [batch, positions, channels]
```

Example:

```text
batch = 32
positions = 16
channels = 64
state shape = [32, 16, 64]
```

The original Neural GPU uses a two-dimensional memory, conceptually:

```text
state shape = [batch, rows, positions, channels]
```

Rows provide scratch capacity. Positions preserve alignment. Channels hold a
learned continuous representation. A “lane” in this repository plays roughly
the role of a row or separate local workspace.

Never accept a comment claiming “six rows” without inspecting the actual tensor
shape. If six embeddings are summed into one state, the model does not possess
six independently writable rows.

## 3. Local convolution means learned communication

A radius-one convolution updates position `i` using positions `i-1`, `i`, and
`i+1`:

```text
new_h[i] = learned_function(h[i-1], h[i], h[i+1])
```

It does not mean the model knows “send carry left.” It means the architecture
permits neighboring communication and training must discover its meaning.

With radius one, information can travel at most one position per microstep:

```text
after 1 step: distance 1
after 2 steps: distance 2
after K steps: distance K
```

Therefore a length-16 tape needs at least 15 local steps for one edge to affect
the opposite edge. More steps increase communication range but also increase
compute per training batch.

## 4. The gated convolutional update

The paper’s Convolutional Gated Recurrent Unit is a spatial analogue of a GRU.
A renderer-safe conceptual form is:

```text
update_gate = sigmoid(convolution_update(state))
reset_gate  = sigmoid(convolution_reset(state))
candidate   = tanh(convolution_candidate(reset_gate × state))

new_state = update_gate × state
          + (1 - update_gate) × candidate
```

The update gate chooses how much old memory to preserve. The reset gate chooses
how much old state participates in constructing the candidate. All convolution
kernels are learned.

Why gates matter: repeatedly applying an unconstrained update can erase or
explode state. A gate can preserve a value over many microsteps. But it does not
tell the model which value should be a carry, borrow, product, or phase flag.

## 5. Weight tying is the algorithmic bias

Let `F` be the learned local transition. A Neural GPU computes:

```text
h¹ = F(h⁰)
h² = F(h¹)
h³ = F(h²)
...
hᴷ = F(hᴷ⁻¹)
```

The same `F` is reused. That is closer to an algorithmic loop than a stack of
unrelated layers:

```text
h¹ = F₁(h⁰)
h² = F₂(h¹)
...
```

However, tying only says “reuse the operation.” It does not ensure the learned
operation is correct, stable, discrete, or meaningful outside training lengths.

## 6. Why the name contains GPU

The name is historical and architectural: the spatial computation resembles a
parallel cellular machine and maps efficiently to GPUs. It is not a hardware
driver and it does not require a special kind of GPU. A small Neural GPU works
on CPU; large training is simply faster on accelerator hardware.

## 7. A minimal PyTorch skeleton

This is pedagogical code, not a competition submission:

```python
import torch
from torch import nn


class ConvGRUCell1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gates = nn.Conv1d(channels, 2 * channels, 3, padding=1)
        self.candidate = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, state):
        # state: [batch, positions, channels]
        x = state.transpose(1, 2)  # [batch, channels, positions]
        update, reset = self.gates(x).chunk(2, dim=1)
        update = torch.sigmoid(update)
        reset = torch.sigmoid(reset)
        candidate = torch.tanh(self.candidate(reset * x))
        new_x = update * x + (1.0 - update) * candidate
        return new_x.transpose(1, 2)


class TinyNeuralGPU(nn.Module):
    def __init__(self, vocab_size: int, channels: int, steps: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, channels)
        self.cell = ConvGRUCell1D(channels)
        self.head = nn.Linear(channels, vocab_size)
        self.steps = steps

    def forward(self, token_ids):
        state = self.embedding(token_ids)
        for _ in range(self.steps):
            state = self.cell(state)
        return self.head(state)
```

Trace the shapes yourself. Then verify that `self.cell` is one object reused K
times—not K separately constructed cells.

## 8. First CPU labs

### Lab A: message propagation without learning

Create a tape containing one marked value. Hand-set a convolution so the mark
moves exactly one position right per step. Print every state.

Mastery question: why can a radius-one model not move the mark five positions
in three steps?

### Lab B: delayed copy

Train the model to copy a marked input digit to a position several cells away.
Train on distances 1–4 and test on 5–8. Compare tied and untied transitions.

Measure exact accuracy and gradient norm by microstep.

### Lab C: binary addition

Use least-significant-bit-first strings. Train on short additions and test on
longer strings. Plot accuracy by output position and by length.

Do not conclude “algorithm learned” from average digit accuracy. Require exact
strings and inspect carry-chain length.

### Lab D: decimal raw square

Use the existing 8,000/2,000 unseen-x split. Decode each output position from
its corresponding spatial position. Plot least-significant to most-significant
digit accuracy. Compare an ordinary affine/GLU update with a generic bilinear
product-forming update.

This is the first research-relevant test.

## 9. Curriculum learning and parameter sharing relaxation

The original paper used curriculum learning: begin with shorter/easier examples
and increase difficulty after the model reaches a performance threshold. This
can create a smoother path to an algorithm but can also hide fragility.

The paper also introduced relaxation of parameter sharing: transitions may
begin less tightly shared and are encouraged toward sharing. The intuition is
that fully tied optimization can be difficult early, while the final model
should behave like a reusable algorithm.

For our project, either technique is a separate causal variable. Do not combine
curriculum, sharing relaxation, new loss, new representation, and a new cell in
one run. A curriculum result must also be evaluated on examples outside every
training length and difficulty bucket.

## 10. What our Neural GPU actually did

Our raw-square model used:

```text
8 LSD-first positions
6 writable lanes
64-wide state
learned left/self/right mixing
same-position lane mixing
one tied GRU update
16 microsteps
8,000 training x values
2,000 strictly unseen x values
```

It achieved:

```text
train exact:       12.9125%
unseen-x exact:     4.0000%
unseen digit acc:  75.6125%
```

The unseen digit profile from least to most significant was:

```text
100.00, 99.80, 96.10, 29.05, 15.20, 68.15, 96.95, 99.65 percent
```

This shape matters. Edge columns contain fewer digit products. Middle columns
contain many cross-products plus carry interactions. The model learned shared
edge correlations but not the exact central computation.

What this establishes: the tested generic grid did not learn raw squaring.

What it does not establish: all Neural GPUs are incapable of arithmetic, or
carry is uniquely responsible. Product formation, product routing,
accumulation, state preservation, and carry transport are still confounded.

Evidence: [raw-square experiment](../../solving/experiments/2026-08-10_multilane_neural_gpu_square/NOTE.md).

## 11. Why the terminal carry head did not solve it

At 50,000 updates:

```text
answer-only unseen exact:       3.85%
carry-supervised unseen exact:  6.25%
```

A carry probe on the terminal state asks whether carry is recoverable there. It
does not force an early message to change the later state that forms a digit.
A hidden stream can correlate with carry while the output uses a different
path.

The next valid carry experiment therefore compares:

```text
terminal control: message reaches final decoder only
causal arm:       message enters future recurrent content updates
```

Then zero, delay, or swap the messages. If behavior does not change more than a
matched random-channel ablation, the model was not causally using them.

Evidence: [50k carry comparison](../../solving/experiments/2026-08-10_multilane_neural_gpu_square_carry_50k/NOTE.md).

## 12. Common implementation traps

### Trap 1: destroying spatial state at decoding

Mean-pooling all positions before decoding erases which state belongs to which
digit. Decode output position `i` from spatial position `i` unless the research
question explicitly tests a global decoder.

### Trap 2: a bypass solves the task

Global attention or a large pooled encoder may let logits ignore the local
machine. Ablate the recurrent state and the bypass separately.

### Trap 3: fake lanes

Adding several lane embeddings into one tensor is not the same as maintaining
separately writable lanes. Verify the state shape.

### Trap 4: changing train and evaluation depth silently

Running 16 microsteps during training and 64 at evaluation tests extrapolation,
but it can also destabilize a transition never optimized at that horizon.
Report the full train-depth/eval-depth grid.

### Trap 5: confusing microsteps with task T

`K` microsteps are internal computation used to produce one task transition.
Task `T` counts recurrence applications requested by the benchmark. They are
different axes.

### Trap 6: measuring only token accuracy

Exact arithmetic requires the entire output. Always include exact accuracy,
position profiles, and errors by carry/product complexity.

### Trap 7: more microsteps under a fixed clock

Twice as many microsteps may yield roughly half as many optimizer updates.
Record examples per second, optimizer steps, and internal cell applications.

## 13. How to audit agent-written Neural GPU code

Ask these questions in order:

1. What is the exact state shape?
2. Which axes are independently writable?
3. Is the transition genuinely tied?
4. What is the convolution radius?
5. What is the receptive field after K steps?
6. Where do immutable inputs enter?
7. Where can mutable state persist?
8. How does output position `i` read state?
9. Is there a global bypass?
10. Does task T control execution or merely condition an embedding?
11. Which loss reaches which microsteps?
12. Does an intervention prove the claimed state is used?
13. What changes relative to the anchor?
14. What CPU tests run before training?
15. What numerical result kills the branch?

If the agent cannot answer with tensor shapes and a causal contrast, do not let
it train the model.

## 14. Research roadmap for this project

### Stage 1: product formation

Compare a generic bilinear/NMU-like cell with a parameter- and FLOP-matched
affine/GLU cell on raw-square final labels. Keep tape, recurrence, decoder,
optimizer, data order, and budget fixed.

Promote only if exact unseen accuracy and central digits improve substantially.

### Stage 2: message consumption

If product formation passes, compare terminal-only and causally consumed
messages. Use matched intervention controls.

### Stage 3: legal final-label modular squaring

Remove arithmetic auxiliary targets without changing the core. Both arms must
fit training before unseen-N differences are interpretable.

### Stage 4: recurrence

Only after near-exact T=1, test autonomous T=2, T=4, and T=8, state drift,
perturbation recovery, and train-depth/eval-depth extrapolation.

## 15. Final mastery test

Without referring to this document, explain:

1. why locality is helpful but insufficient;
2. why weight tying is an inductive bias, not proof of an algorithm;
3. why the middle-digit profile implicates more than carry alone;
4. why a terminal probe does not demonstrate causal use;
5. why more microsteps can hurt hosted learning;
6. how K differs from task T;
7. what one experiment should run before adding modular reduction;
8. what result would make you stop Neural GPU work.

Then implement the tiny cell, run the propagation lab on CPU, and show the
gradient path from a later output to an earlier message writer. That is the
minimum standard for directing an agent on the next Neural GPU card.

