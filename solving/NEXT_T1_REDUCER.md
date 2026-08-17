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

### A. Projected scratch bus — cheap falsifier, not the final bet

One change from the failed scratch-message card: apply a learned 1x1 projection
before each shifted message. Keep 33 clocks and three zero-initialized gates.

Prediction: validation above 18%; raw q=32..63 above 10%; q<16 above 95%.
Reject if the gates open but q=32..63 stays below 5%.

This test separates the two hypotheses. If transformed transport creates a
large jump in the quotient frontier, communication was the limiting resource.
If it only advances proportionally with clocks, the cell is still executing
the repeated-subtraction policy and the next experiment must change the
learned state transition rather than add reach.

### B. Learned content-addressed pointer — primary candidate

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

### C. State-driven expert controller

Add learned hold/gather/commit experts only after A or B moves the frontier.
Initialize the commit residual at zero so the anchor remains recoverable.

Prediction: the controller should reduce late oscillation and make frontier
gains stable. If communication has not already moved q=32..63, this is likely
to learn a cosmetic phase schedule and should not be run.

### D. Coarse quotient bottleneck — independent alternative

A global branch predicts a small discrete latent code from `(source, N)`. A
shared learned local decoder uses that code, source, and modulus to produce the
residue. Neither the latent code nor any internal state receives quotient
labels. The bottleneck is useful only if counterfactual code interventions
change the output in ordered quotient-like bands and transfer to unseen N.

This borrows the factorization idea from neural arithmetic modules without
using a hard-coded reciprocal or exact multiply/subtract path. It has stronger
end-to-end identifiability than a free scratch grid, but also a serious failure
mode: the code can become a generic memorization index. A capacity-matched
continuous-code control is mandatory.

### E. Return to fused x only after the reducer gate

Minimum reducer gate:

- validation >=30%;
- both unseen-N audits >=25%;
- raw q=32..63 >=50%;
- measurable nonzero accuracy in finer q=64..127;
- no collapse of q<16.

Then replace exact square with `x` and compare fused learning. If fused remains
far below the reducer, restore a learned squaring phase or a joint curriculum;
do not claim the direct model has learned squaring from shallow-q success.

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

The practical synthesis is: preserve local arithmetic, add a transformed
scratch communication path, and make phase/commit decisions learned and
state-dependent. Raw reach, extra width, and more clocks alone are insufficient.
