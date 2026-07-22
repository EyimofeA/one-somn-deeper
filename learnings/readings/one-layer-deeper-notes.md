# One Layer Deeper — Lecture Notes

Written in Simplified Technical English. Short sentences. One idea per sentence.
Consistent terms. Active voice.

**Terms used in these notes**

| Term | Meaning |
| --- | --- |
| block | One weight-tied unit of computation. You apply it many times. |
| step | One application of the true recurrence. Example: one squaring. |
| state | The tensor that the block reads and writes. |
| depth | The number of times you apply the block. |
| T | The number of steps the problem asks for. |
| K | The number of times you apply the block during training. |
| f | The true one-step map. |
| B | Your learned block. |
| exact | Every output token is correct. |

---

## Part 1 — What a raw transformer computes

### 1.1 A layer does two things

A transformer layer has two operations. There is nothing else.

1. **Attention** moves data between positions. It is a soft gather. Each position
   reads a weighted average of the other positions.
2. **The MLP** changes data at one position. It applies the same function at
   every position.

Attention is the only operation that crosses positions. The MLP is the only
operation that is non-linear in a useful way.

### 1.2 Depth is fixed

A model with L layers applies L layers to every input.

An easy input gets L layers. A hard input gets L layers. The amount of
computation does not depend on the input. This is the property the competition
attacks.

### 1.3 The circuit view

Think of the model as a circuit.

```
            positions  ------------------------->  (width = sequence length)
          +---+---+---+---+---+---+---+---+---+
 layer 1  | . | . | . | . | . | . | . | . | . |
          +---+---+---+---+---+---+---+---+---+
 layer 2  | . | . | . | . | . | . | . | . | . |     depth = L
          +---+---+---+---+---+---+---+---+---+     (fixed, small)
   ...    | . | . | . | . | . | . | . | . | . |
          +---+---+---+---+---+---+---+---+---+
 layer L  | . | . | . | . | . | . | . | . | . |
          +---+---+---+---+---+---+---+---+---+
```

The circuit is **wide and shallow**. Width grows with the input. Depth does not.

### 1.4 The formal limit

Merrill and Sabharwal (2023) proved a result about this shape.

> A transformer with logarithmic precision is in uniform TC⁰.

TC⁰ is the class of problems solvable by constant-depth circuits with threshold
gates. "Constant" means the depth does not grow when the input grows.

So a fixed-depth transformer solves only TC⁰ problems, for all input sizes.

### 1.5 Why this decides the competition

Barrington (1986) proved a second result.

> The word problem over the group S₅ is NC¹-complete.

The word problem is this. You get a list of permutations. You compute their
product. That is **iterated composition**.

TC⁰ is contained in NC¹. Most researchers believe the containment is strict.

Put the two results together.

> **Result 1.** Iterated function composition is believed to be outside the reach
> of a fixed-depth transformer. Width does not replace depth. The name of the
> competition is literal.

This is the first thing to understand. It is not a training problem. It is not a
capacity problem. A fixed-depth model does not have the shape.

---

## Part 2 — The three ways to add depth

There are only three. Two are closed to you.

### 2.1 Stack more layers

Add layers until depth is large enough. This fails for three reasons.

1. Depth is still fixed. It does not grow with T.
2. Parameters grow with depth. The ceiling is 500,000,000 elements.
3. A deeper stack has no reason to extrapolate to a larger T.

**Closed.**

### 2.2 Write a scratchpad

Generate tokens one at a time. Each generated token adds one more pass through
the network. This is chain-of-thought. It is the standard method to add serial
depth to a transformer.

The evaluator fixes the output positions. Your model returns logits for a fixed
set of positions. It does not generate tokens.

**Closed by the rules.**

### 2.3 Apply one block many times

Keep one block. Apply it K times. This is the Universal Transformer.

1. Parameters do not grow with depth.
2. Depth becomes a runtime choice.
3. Depth can depend on T.

**Open. It is the only route that is open.**

> **Result 2.** The rules force latent recurrence. Recurrence is not one option
> among several. It is the only option that survives the constraints.

---

## Part 3 — The task, stated exactly

### 3.1 The recurrence

```
x₀ = x mod N
xₜ = xₜ₋₁²  mod N
y  = x_T
```

N is a semiprime. The factors p and q are secret. The model gets N, x and T as
decimal digits. The model returns y as decimal digits.

### 3.2 The map

Define `f_N(u) = u² mod N`. This is a function from Z_N to Z_N.

The answer is `f_N` applied T times to x.

### 3.3 The two sub-problems

The task splits cleanly. Solve both or score zero.

| Sub-problem | What it needs |
| --- | --- |
| **A. Represent** | Compute `f_N` exactly for one step. |
| **B. Compose** | Apply the step T times, exactly, with T larger at test time. |

Your current work attacks B. It does not attack A. Part 7 explains why A is
harder than it looks.

---

## Part 4 — The error mathematics

This part is the centre of the notes. Read it twice.

### 4.1 Analog recurrence cannot extrapolate

Let `z` be the state. Let `B` be the learned block. Let `f` be the true step.

Let `ε` be the one-step error. Let `L` be the Lipschitz constant of B in the
region of interest. L measures how much B expands a small difference.

The error after k steps obeys this recursion.

```
e_k  ≤  L · e_{k-1}  +  ε
```

Solve it.

```
e_k  ≤  ε · (L^k − 1) / (L − 1)        for L ≠ 1
e_k  ≤  k · ε                          for L = 1
```

Three cases exist. Examine each.

```
 L > 1     e_k grows like L^k.        Error explodes. Extrapolation fails.
 L = 1     e_k grows like k·ε.        Error grows slowly. Still fails at T=200.
 L < 1     e_k ≤ ε / (1 − L).         Error is bounded.
                                      But B contracts.
                                      Different states merge.
                                      The model loses x.
```

> **Result 3.** No value of L works. A purely analog recurrence cannot be exact
> at large T. This is a property of the arithmetic, not of any particular
> architecture or optimizer.

Note the correction to a common claim. The failure is not only "exponential
compounding". At L = 1 the growth is linear and the model still fails, because
the scoring is exact-match. Linear growth in a system with zero tolerance is
still a failure.

### 4.2 Quantization removes the error completely

Now add a quantizer.

Let `D` be the set of valid states. In your case D is the set of one-hot digit
strings. Let `δ` be the minimum distance between two members of D.

Define the step as `S = q ∘ B`, where q snaps to the nearest member of D.

**Claim.** If `‖B(d) − f(d)‖ < δ/2` for every `d` in D, then `S(d) = f(d)`
exactly.

**Proof.** B(d) lands inside the ball of radius δ/2 around f(d). By the
definition of δ, no other member of D lies in that ball. So q returns f(d). ∎

Then `S` applied T times equals `f` applied T times, for **every** T.

```
        analog                          quantized
        ------                          ---------

   f(d) *                          f(d) *---.
        |  drift                        |    ) noise margin δ/2
   B(d) o  ε                       B(d) o---'
        |                                |
   next * accumulates              q(B(d)) = f(d)   error = 0
```

> **Result 4.** Exactness is a **threshold** property. Below the margin δ/2 the
> error is exactly zero. Above it the error is total. There is no middle region.

This is the same mathematics as the noise margin of a logic gate. It is the same
mathematics as the threshold theorem in fault-tolerant computing. A digital
computer runs for 10¹⁵ steps without drift for exactly this reason.

### 4.3 The condition that training does not test

Look again at the condition.

> `‖B(d) − f(d)‖ < δ/2` for **every** d in D.

The condition quantifies over **every reachable state**. It does not quantify
over the training distribution.

Training visits only the states that appear within K steps of a clean start.
That is a small subset of D.

```
   D  = all reachable digit strings
   ┌────────────────────────────────────────────┐
   │                                            │
   │   ┌──────┐                                 │
   │   │ seen │   states within K steps         │
   │   │      │   of a clean start              │
   │   └──────┘                                 │
   │                                            │
   │        the model has no constraint here    │
   │        and this is where T = 200 lives     │
   └────────────────────────────────────────────┘
```

> **Result 5.** The model fails at large T because it was never asked to be
> correct on the states it produces itself.

The fix follows directly. Train the block on its own outputs.

Method: sample n. Run the block n times with gradients turned off. Then run it
m more times with gradients on. Compute the loss at step n+m.

This is the **progressive loss** of Bansal et al. (2022). It enlarges the
training set toward D. It costs almost nothing, because the detached steps hold
no activations.

---

## Part 5 — What each rule constrains, and why

| Rule | Direct implication |
| --- | --- |
| Model state ≤ 500M elements | No lookup table over Z_N when N is wide. You must build an algorithm. |
| Evaluator owns the loop and the backward pass | No custom training loop. The `auxiliary` return is your only free channel. |
| Wall-clock budget, not a step budget | A deeper forward pass buys fewer updates. The optimizer is a first-class design variable. |
| Fixed output positions | No scratchpad. No chain-of-thought. See 2.2. |
| Hard may change the recurrence | Any method specific to squaring scores zero on Hard. |
| Exact accuracy, all tokens | An approximate method scores zero. |
| Held-out depth split | T-extrapolation is scored directly. |

Combine the last three rows.

> You must learn an **unknown** step map from data, apply it an **unseen** number
> of times, and be **exactly** right.

Only an error-correcting recurrence satisfies all three at once. Section 4.2 is
the only known mechanism. This is why the design space is narrow.

---

## Part 6 — The paths, ranked

### Path A — Fixed-depth transformer

Learn the input-output map directly. Add width until it fits.

- Fails by Result 1 and Result 3.
- Memorizes the training set. Scores near zero on held-out depth.
- **This is what the leaderboard is doing.** The top score is 0.40 percent.
- **Reject.**

### Path B — Algebraic solver

Use the structure of (Z/N)*. In the character basis, squaring multiplies every
phase by 2^T. The answer is a closed form. It is O(1) in T.

- Exact and fast on Easy.
- Transfers to nothing when Hard changes the recurrence.
- Prohibited by rule 10 as a task-specific solver.
- **Reject.** You already did.

### Path C — Analog weight-tied recurrence

The Universal Transformer, applied K times.

- Fails by Result 3.
- **This is where your code is now.**

### Path D — Quantized recurrence, progressive loss, input injection

- Quantize between steps. This gives Result 4.
- Use progressive loss. This gives Result 5.
- Re-inject N and x every step. This stops drift off-task.
- Extrapolates in T for free once the margin condition holds.
- Recurrence-agnostic. It transfers to Hard.
- **Recommended. Build this next.**

### Path E — Two-level recurrence

Outer loop over t = 1 to T. Inner loop over digit limbs.

- Necessary if N has many digits. Part 7 explains why.
- Both loops share weights on their own axis.
- **Build this if Path D stalls, or immediately if N is wide.**

### Path F — Pointer doubling

Represent the map f as an object, not as a value. Then compose it with itself.

```
 g₀ = f
 g₁ = g₀ ∘ g₀    = f²
 g₂ = g₁ ∘ g₁    = f⁴
 g_k = f^(2^k)
```

You reach `f^T` in log₂(T) compositions, not T.

- This is a genuine log-depth algorithm for iterated composition.
- It is recurrence-agnostic. It works for cubing, for affine maps, for anything.
- It extrapolates perfectly. T = 100 to T = 200 costs one extra composition.
- **Cost:** you must hold f as a table over Z_N inside the forward pass. This is
  feasible only if Z_N is small. Check the digit count of N before you spend
  time here.
- **Measure first. Then decide.**

### Path G — Cycle structure

The orbit of x under f enters a cycle. Write μ for the tail length and λ for the
cycle length.

```
      x₀ → x₁ → x₂ → x₃ → x₄ → x₅
                      ↑          ↓
                      x₈ ← x₇ ← x₆

      tail μ = 3        cycle λ = 5
```

Then `x_T = x_{μ + ((T − μ) mod λ)}`.

- For small N, μ + λ is usually small.
- A model with an exact step reaches this for free.
- Path D subsumes Path G. Once quantized, the state enters the cycle by itself
  and stays there. Extrapolation in T becomes automatic.
- **Do not build this. Measure it.** Plot the distribution of μ + λ on your local
  datasets tonight. If the values are small, Path D gives you T-extrapolation
  with no extra machinery.

---

## Part 7 — Why one step is harder than it looks

You must compute `u² mod N` where u and N are digit strings.

Break it into three operations.

**1. Partial products.** For n digits there are n² digit pairs. Each pair is
local. Attention can gather them. A transformer handles this well.

**2. Carry propagation.** Carries chain along the digits. This looks serial. It
is not. The carry monoid is associative, so carry propagation is a **parallel
prefix scan**. Attention performs prefix scans. This is why addition
length-generalizes when the position encoding is right. See McLeish et al.
(2024), the Abacus embedding paper: 20 digits at training, 100 digits at test.

**3. Reduction modulo N.** This is division. Division needs comparison and
conditional subtraction, repeated. There is no known constant-depth method for
general N.

> **Result 6.** One step is itself a recurrence. A fixed-depth block cannot
> represent it beyond about three digits.

This is the missing half of your design. Your model has no representable
algorithmic solution available to it. So memorization is the only fixed point
that training can reach. You are regularizing against memorization while the
alternative stays unreachable.

The formal tool here is **RASP-L** (Zhou et al., 2023). The conjecture is
simple.

> A transformer length-generalizes on a task if and only if the task has a short,
> loop-free, position-independent RASP program.

Use it as a design check. Write the RASP-L program for your step **before** you
train. If you cannot write it, the model cannot learn it, and you save 3,600
H100 seconds.

---

## Part 8 — The plan, in order

Each item changes one thing. Each item has a prediction. Write the prediction
before the run.

1. **Measure μ + λ on local data.** No GPU needed. This decides whether Path F
   and Path G are live. One evening.
2. **Count the digits of N in h1 and m5.** This decides whether Path E is
   mandatory. Ten minutes.
3. **Progressive loss.** Implement inside `forward`. Return the consistency term
   through `auxiliary`. Prediction: the T-extrapolation curve flattens.
4. **Straight-through quantization between steps.** Prediction: exact-match
   becomes a step function in T, not a decay.
5. **Input injection each loop.** Prediction: small gain, large gain when
   combined with 4.
6. **Init scale down.** See Omnigrok (Liu et al., 2022). Grokking is controlled
   by the weight norm at initialization. Scale init by α < 1. Prediction: the
   memorize-to-generalize transition moves earlier than 64,000 steps.
7. **Weight decay 0.1 → 1.0 → 3.0.** One constant. Run it, but after 6.
8. **Muon optimizer.** Two reasons. It is faster in wall-clock. And the
   Newton-Schulz orthogonalization holds the update singular values near 1,
   which controls L in Section 4.1. For a block you apply 200 times, control of
   the Jacobian spectrum is the whole problem.
9. **Two-level recurrence.** Only if 3 to 8 stall.

Do not add parameters. Capacity is a solved problem in your project. A model at
d = 2048 sits in the lazy regime and memorizes. See Kumar et al. (2024) on the
lazy-to-rich transition.

---

## Part 9 — The measurement protocol

One command. One fixed output path. Every experiment produces it.

```
Train on T ∈ {1, 2, 3}
Evaluate at T = 4, 5, 8, 16, 32, 64
Plot exact-match against T
```

Read the curve like this.

```
 exact
 match
   1.0 |*****************      learned the step (Result 4 holds)
       |
       |****                   learned a few steps then drifts
       |    \___
       |         \_____
   0.0 |*   \__________        memorized depth
       +----------------------
         1  2  3 | 4  8  16  T
                 |
              training limit
```

This curve is your scoreboard. The Hard leaderboard is not. Fifteen of the
seventeen entries sit between 0.02 and 0.05 percent. The gap between rank 11 and
rank 3 is one or two examples. That is noise, not signal.

---

## Part 10 — Reading order

Read in this order. Each entry says what you get from it.

**Tier 1 — this week**

1. **Bansal et al. (2022), "End-to-end Algorithm Synthesis with Recurrent
   Networks: Logical Extrapolation Without Overthinking."** Gives you the recall
   connection and the progressive loss. Implement from this paper.
2. **Schwarzschild et al. (2021), "Can You Learn an Algorithm?"** Gives you the
   experimental method for depth extrapolation. Read for Part 9.
3. **Zhou et al. (2023), "What Algorithms Can Transformers Learn?"** Gives you
   RASP-L. Read for Part 7.
4. **Geiping et al. (2025), "Scaling up Test-Time Compute with Latent
   Reasoning."** The modern scaled version of your architecture. Gives you
   truncated backpropagation and random iteration counts.
5. **Liu et al. (2022), "Omnigrok."** Gives you the init-scale knob.

**Tier 2 — architecture**

6. Dehghani et al. (2018), Universal Transformers. You have read this.
7. Bai, Kolter, Koltun (2019), Deep Equilibrium Models. Gives you O(1)-memory
   gradients through unbounded depth.
8. Fung et al. (2021), Jacobian-Free Backpropagation. The practical version of 7.
9. Merrill and Sabharwal (2023), "The Parallelism Tradeoff." The proof behind
   Result 1.
10. Barrington (1986). Read the statement, not the proof. Twenty minutes.

**Tier 3 — arithmetic**

11. McLeish et al. (2024), Abacus embeddings. Position encodings for digits.
12. Nanda et al. (2023), "Progress Measures for Grokking." Reverse-engineers the
    grokked circuit for modular addition. Tells you what to probe for.
13. Lee et al. (2023), "Teaching Arithmetic to Small Transformers."

**Tier 4 — context, one evening**

14. Rivest, Shamir, Wagner (1996), Time-lock Puzzles. Tells you what the
    sequentiality assumption claims. It claims something about
    cryptographic-size N. It claims nothing about benchmark-size N. That gap is
    Path F and Path G.

---

## Summary in seven lines

1. A fixed-depth transformer cannot iterate. This is a theorem, not a tuning
   problem.
2. The rules close the scratchpad. Latent recurrence is the only route.
3. An analog recurrence cannot be exact at large T, for any Lipschitz constant.
4. A quantized recurrence is exactly correct once the one-step error is below
   the noise margin.
5. That condition must hold on every reachable state, not on the training
   states. Progressive loss is how you get there.
6. One step is itself a recurrence, because reduction modulo N is division.
7. Measure the T-extrapolation curve. Ignore the leaderboard.
