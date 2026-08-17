# Next T=1 reducer: from clock-limited recurrence to a learned microprogram

Date: 2026-08-17

This is a forward-looking design, not a result. It follows the measured
quotient frontier in
[`experiments/2026-08-17_binary_t1_quotient_diagnostic/`](experiments/2026-08-17_binary_t1_quotient_diagnostic/).

## What is now known

The width-128 local ConvGRU is not a generic failed neural network. It has
learned a real but incomplete reduction procedure.

| Exact-square reducer | Validation | raw q=16..31 | raw q=32..63 | Runtime |
|---|---:|---:|---:|---:|
| 33 clocks | 17.10% | 62.45% | 0.27% | 549s |
| 44 clocks | 19.92% | 99.63% | 9.86% | 837s |
| 55 clocks | 21.52% | 100.00% | 32.33% | 1065s |

The sharp frontier moves when and only when recurrent clocks increase. Width,
global attention, wide kernels, dilation, raw shifted messages, initialization,
and learning-rate schedules do not reproduce that movement. The missing object
is not merely communication. It is a reusable conditional state transition
that compares an aligned value with `N`, decides whether to update, and writes
the new remainder without destroying local carry/borrow information.

The validation distribution makes this a hard ceiling, not a tail issue:

| raw quotient | Share of validation | Cumulative share |
|---|---:|---:|
| 0..15 | 13.62% | 13.62% |
| 16..31 | 5.38% | 19.00% |
| 32..63 | 7.30% | 26.30% |
| 64..127 | 9.28% | 35.58% |
| 128..255 | 14.14% | 49.72% |
| 256..511 | 18.12% | 67.84% |
| 512..1023 | 19.40% | 87.24% |
| 1024..2047 | 12.76% | 100.00% |

Thus even perfect accuracy through `q=63` caps this split at 26.30%. A model
must handle quotient magnitude near 2,000 to solve the full 11-bit task.

The same 5,000 rows look radically smaller under an MSB-prefix reduction
view. Across 22 source-bit stages, the exact classical state would require a
mean **4.21**, median **4**, and maximum **10** conditional reductions; 99% of
rows need at most 8. The first wrap occurs only at stages 12--22. These numbers
are diagnostic motivation only and are not supplied to the model. They explain
why streaming can escape an O(q) basin without increasing model width.

Support is also adequate for a real transition test. The 100,000 training
paths cover 201,140 of 238,428 possible `(N, bounded residue, next bit)` states
for the 90 seen moduli (84.36%; every modulus is 80.44--88.37% covered).
Among held-out-`x`/seen-`N` validation paths, 97.28% of local transitions have
appeared in training and 70.92% of complete rows use only seen transitions.
These classical states are used only for this audit. A seen-N failure would
therefore implicate credit assignment/representation more than local support;
unseen N remains genuine algorithmic extrapolation.

## Strongest mechanistic hypothesis: repeated subtraction

The observed boundary is much more consistent with *learned repeated
subtraction* than with binary long division. Adding 11 clocks moves the useful
quotient range outward, while the maximum operand width is unchanged. A true
bit-serial long-division-like policy should need time proportional mainly to
the 22-bit workspace, not to the numerical value of the quotient.

This remains an inference: final-output diagnostics cannot reveal the exact
hidden program. But it changes the research target. The goal is not merely to
make information travel faster. It is to make a doubling/alignment-style
policy easier to discover than subtracting one `N` per recurrent cycle.

## Mathematical target, without implementing it in the forward pass

A conventional binary reducer repeatedly considers an aligned copy of `N`,
decides whether the current value is at least that copy, and conditionally
updates the value. We may use that only to define what capabilities a learned
cell needs. Competition code may not contain a hard-coded division/remainder
algorithm, fixed solved weights, task-value modulus, or dataset inspection.

The model therefore needs to *learn* four capabilities:

1. represent a tentative remainder and immutable modulus;
2. communicate high-bit comparison evidence across positions;
3. preserve a decision while local differences/borrows propagate;
4. conditionally commit or reject a candidate update.

The existing cell has (1), partially learns (2--4), but spends too many clocks
rediscovering their coordination for each new quotient bit.

## Prior result that this design must not accidentally repeat

The August 9 shifted-long-division diagnostic already learned an almost exact
decimal comparator/subtractor on unseen moduli and reduced quotients through
99,999,999 with at least 99.9023% exactness at every tested scale. It used
direct intermediate supervision and a fixed high-to-low shift traversal.
Therefore it established *capability*, but not legal discovery from final
labels. Rebuilding another supervised subtractor is not research progress.

Tonight adds the missing final-label evidence: the generic local ConvGRU does
discover a reducer, but apparently settles on repeated subtraction. The live
problem is now narrowly defined as learning the scheduler/alignment policy
without putting that policy in Python.

## Recommended architecture

The highest-value next diagnostic is now a **streaming exact-square reducer**,
before adding a global pointer. It changes the computational geometry instead
of trying to accelerate the repeated-subtraction basin.

```text
22 exact-square bits, MSB first
           │ one new bit per tied step
           ▼
  bounded recurrent residue tape  <──── immutable N tape
           │
           ▼ after 22 steps
      residue-bit logits
```

This is research-only because the exact square is supplied externally. The
cell receives no prefix residues, quotient, comparison, subtraction, carry,
or execution trace; only the final residue label trains it. A successful
result would establish that final-label credit can identify reduction when the
architecture keeps every intermediate magnitude bounded. It would *not* prove
that the competition model can generate the square tape.

This differs from rejected H13. H13 streamed the original 11 bits of `x` and
asked the same latent state to invent squaring and reduction simultaneously.
The new card streams the 22 bits of an already isolated exact product and asks
only whether the reduction transition can be learned.

The fallback, if streaming itself is not identifiable, is the learned pointer
machine below:

```text
immutable source lane ───────────────┐
immutable modulus lane ──────────────┼──> tied local ConvGRU ──> work lane
                                     │            ▲               │
work lane ──> learned projections ──>│ message    │               │
             + sparse content router └── scratch lane <───────────┘
                                              │
                                    state-driven controller
                                    (hold / gather / commit)
```

### 1. Keep the proven local path

Retain the tied 3x3 ConvGRU, immutable source/modulus lanes, boundaries,
binary representation, Muon, and recurrent dropout. The 33/44/55-clock result
shows this path can implement correct shallow reduction and extend with time.
Replacing it with global attention or dilation destroys useful locality.

### 2. Transform messages before moving them

Tonight's fast-message variants shifted raw hidden activations. That assumes
the same channels mean the right thing at every source and destination. The
opened gates with no frontier gain show that assumption is false.

The next bus should use learned zero-initialized projections:

```python
message_d = project_d(work_state)
scratch = scratch + gate_d * route(message_d, distance=d)
```

`project_d` is a learned 1x1 channel transform. `route` may initially use fixed
relative offsets 2/4/8 only in the research sandbox. The competition version
should prefer learned content/relative-position attention so the operation is
not a hard-coded division schedule. Zero gates preserve the local anchor at
initialization.

### 3. Separate communication from commitment

The scratch lane stores comparison/borrow evidence. A learned controller reads
pooled scratch/work features and emits gates for three *learned* experts:

- continue local gathering;
- hold the current work value;
- commit a learned candidate update.

Those labels describe intended roles, not coded arithmetic. Every expert is
randomly initialized and learned from final residue labels. The controller is
state-driven, not a Python phase counter, so the model—not the forward pass—
chooses its microprogram.

### 4. Preserve outer recurrence

After one transition, discretize residue bits with the existing straight-
through boundary and feed only that exact state to the next outer T step. The
inner microprogram is shared within one transition; the whole transition is
then shared across T. Do not revisit outer recurrence until the held-out-N T=1
gate passes.

## Preregistered experiment ladder

Use the existing deterministic 100k/5k/5k/5k T=1 rows. Do not create a new
dataset and do not tune on either unseen-N audit.

### A. Streaming exact-square reducer — recommended first

Use the existing deterministic rows and exact-square source. Reveal one source
bit MSB-first per tied recurrent step. Keep `N` immutable and decode only after
all 22 bits. Use the same final residue BCE, Muon, recurrent dropout, train/
validation/audit policy, and approximately matched examples.

Prediction: validation above 50% and both unseen-N audits above 40%; the
ambitious success gate is 90/85%. Accuracy should be approximately flat across
raw quotient buckets. Reject if validation remains below 30% or if accuracy
still forms a moving quotient frontier.

Before any submission translation, obtain a narrow legality ruling on whether
revealing one existing input bit per generic tied update is ordinary recurrent
input routing or a forbidden fixed algorithm. The reducer diagnostic itself is
not submission code because it receives the external square.

### B. Projected scratch bus — cheap falsifier, not the final bet

One change from the failed scratch-message card: apply a learned 1x1 projection
before each shifted message. Keep 33 clocks and three zero-initialized gates.

Prediction: validation above 18%; raw q=32..63 above 10%; q<16 above 95%.
Reject if the gates open but q=32..63 stays below 5%.

This test separates the two hypotheses. If transformed transport creates a
large jump in the quotient frontier, communication was the limiting resource.
If it only advances proportionally with clocks, the cell is still executing
the repeated-subtraction policy and the next experiment must change the
learned state transition rather than add reach.

### C. Learned content-addressed pointer — fallback candidate

Keep every local clock. Add one scratch query representing the current focus.
Every fourth clock it attends over projected work/modulus keys with learned
relative-position bias, writes the attended evidence into scratch, and emits a
new query. The attention result may influence the learned local update, but it
may not directly overwrite public output bits. No Python phase index tells the
query which bit to visit and no fixed traversal order is encoded.

Prediction: higher upside than A but worse throughput. A successful run should
jump across quotient powers of two at 33 clocks. Promote only if it beats A's
quotient frontier at matched wall time, not merely matched examples, and if
attention locations depend on `(source, N)` rather than following one constant
position pattern.

### D. State-driven expert controller

Add learned hold/gather/commit experts only after A or B moves the frontier.
Initialize the commit residual at zero so the anchor remains recoverable.

Prediction: the controller should reduce late oscillation and make frontier
gains stable. If communication has not already moved q=32..63, this is likely
to learn a cosmetic phase schedule and should not be run.

### E. Coarse quotient bottleneck — low-priority independent alternative

A global branch predicts a small discrete latent code from `(source, N)`. A
shared learned local decoder uses that code, source, and modulus to produce the
residue. Neither the latent code nor any internal state receives quotient
labels. The bottleneck is useful only if counterfactual code interventions
change the output in ordered quotient-like bands and transfer to unseen N.

This borrows the factorization idea from neural arithmetic modules without
using a hard-coded reciprocal or exact multiply/subtract path. It has stronger
end-to-end identifiability than a free scratch grid, but also a serious failure
mode: the code can become a generic memorization index. A capacity-matched
continuous-code control is mandatory. Prior supervised action-conditioned
macro decoders already failed to generate exact multi-unit updates, so this is
lower priority than streaming and must not reuse that closed formulation.

### F. Return to fused x only after the reducer gate

Minimum reducer gate:

- validation >=30%;
- both unseen-N audits >=25%;
- raw q=32..63 >=50%;
- measurable nonzero accuracy in finer q=64..127;
- no collapse of q<16.

Then replace exact square with `x` and compare fused learning. If fused remains
far below the reducer, restore a learned squaring phase or a joint curriculum;
do not claim the direct model has learned squaring from shallow-q success.

The preferred fusion is not a free square tape followed by the reducer. Use a
streaming two-work-lane state over the original `x` bits: one writable lane can
represent the consumed prefix, the other the bounded transformed state, and
the same whole-position learned scan updates both. Only the final public state
is decoded. These are intended roles, not supervised variables or coded
updates. This differs from failed H13 because each source-bit stage has a
complete serial view of the work and modulus positions rather than two local
message hops.

Run this fused translation only if the exact-square streaming diagnostic clears
its unseen-N gate. If it fails while the diagnostic succeeds, the unresolved
problem is square/transition identifiability. If both succeed, discretize the
public state and test evaluator-owned outer recurrence; do not change all three
levels simultaneously.

## Measurement changes

Every future T=1 run should report:

- aggregate exact and per-bit accuracy;
- exact by raw quotient for reducer-only input;
- exact by centered quotient for fused input;
- valid-residue rate and zero/identity shortcut rates;
- train/validation curves by examples and wall time;
- learned bus gates/controller usage;
- the two unseen-N audits opened once after validation selection.

Aggregate exact alone hid the mechanism. The quotient frontier is now the
primary diagnostic.

The decisive diagnostic is *frontier movement per recurrent clock*. A useful
architectural change should jump across quotient powers of two at fixed clock
count. A linear shift is evidence that the same slow microprogram survived.

## Literature boundary

- [Neural GPUs Learn Algorithms](https://arxiv.org/abs/1511.08228) supports
  tied local recurrent computation, recurrent dropout, and difficult
  optimization across long unrolls.
- [Extensions and Limitations of the Neural GPU](https://arxiv.org/abs/1611.00736)
  shows that curriculum and scale can broaden learned arithmetic, while
  atypical inputs still expose brittle rules.
- [Neural Programmer-Interpreters](https://arxiv.org/abs/1511.06279) motivates
  scratchpads and reusable learned programs, but its rich execution-trace
  supervision is unavailable here.
- [Making Neural Programming Architectures Generalize via Recursion](https://arxiv.org/abs/1704.06611)
  motivates reducing the domain of each learned subproblem. Our version must
  learn the phases from final labels and remain within the competition ban on
  hard-coded forward algorithms.
- [Learning Division with Neural Arithmetic Logic Modules](https://arxiv.org/abs/2110.05177)
  confirms that even continuous scalar division is unusually sensitive to
  input range and optimization. Its reciprocal units are not exact modular
  reducers; the transferable idea is a constrained latent arithmetic
  bottleneck, not importing a division oracle.
- [Algorithm Development in Neural Networks: Insights from the Streaming
  Parity Task](https://arxiv.org/abs/2507.09897) shows that a tied streaming RNN
  can undergo a delayed transition to a finite-state algorithm after enough
  training experience. That supports monitoring beyond first interpolation,
  but parity has a tiny fixed state space; it does not predict that
  variable-modulus reduction will grok.

The practical synthesis is: preserve local arithmetic, add a transformed
scratch communication path, and make phase/commit decisions learned and
state-dependent. Raw reach, extra width, and more clocks alone are insufficient.
