# Streaming exact-square reducer

Status: preregistered design; not run.

CPU implementation smoke only: the 24-channel scan fallback reached 100%
exact on all 12 padded rows from toy moduli 5 and 7 by 200 updates, proving the
forward and gradient path can fit. On 13 rows from unseen toy moduli 3, 4, and
6 it scored 4/13 (30.77%), exactly equal to always predicting zero. This is not
an algorithmic result—the support is intentionally tiny—but it is a warning
that train interpolation is not a promotion signal.

A second CPU smoke used 72 complete rows from six 4-bit training moduli and 32
rows from four unseen moduli. The scan reached 100% train, but unseen-N exact
peaked at 14/32 (43.75%) at step 100—only one row above the 13/32 unreduced
square baseline—and fell to 9/32 after interpolation. This still does not test
the registered 90-modulus regime. It reinforces validation checkpointing and
rules out treating tiny-table train fit as evidence.

## Question

Can final residue labels identify a reusable variable-`N` reduction transition
when the source magnitude is consumed one bit at a time and the recurrent work
state stays bounded, rather than presenting the full 22-bit square at once?

This card responds to the measured all-at-once exact-square frontier: 33, 44,
and 55 ConvGRU clocks move accuracy progressively through raw quotient bands,
consistent with learned repeated subtraction. A streaming representation can
express reduction in a fixed number of source-bit stages independent of the
numeric quotient.

On the unchanged 5,000 validation rows, a diagnostic MSB-prefix calculation
uses a mean 4.21, median 4, and maximum 10 conditional reductions over the 22
stages; 99% of rows use at most 8. The first wrap occurs at stages 12--22. By
contrast, the raw quotient reaches 1,975. These internal classical states are
not training data or model inputs; they only quantify the intended reduction
in computational depth.

The 100,000 training paths cover 201,140/238,428 (84.36%) of all possible
classical `(N, bounded residue, next bit)` transitions for the 90 seen moduli.
The held-out-`x`/seen-`N` validation paths have 97.28% transition overlap with
training, and 3,546/5,000 rows contain no locally novel transition. This is a
diagnostic support audit only. It makes seen-N validation a fair test of
transition learning while preserving unseen-N as the extrapolation gate.

## Scope and legality

This is a research-only reducer diagnostic. Python supplies the exact 22-bit
`x*x` tape, just as in the existing exact-square isolation. The model receives
no quotient, prefix remainder, comparison, subtraction, carry, or other trace
target. Training uses only the final 11 residue bits.

The exact-square input makes this ineligible as a competition submission.
Before translating the mechanism into a submission, ask the organizers only:
"May a model reveal one already-present input bit per application of a generic
tied recurrent cell, when no arithmetic action or intermediate label is
hard-coded?"

## Unchanged data and selection

- seed 74;
- 11-bit `x` and semiprime `N`;
- 100,000 training rows;
- 5,000 unseen-`x`/seen-`N` validation rows for checkpoint selection;
- 5,000 seen-`x`/unseen-`N` and 5,000 joint-unseen audit rows, opened once;
- batch 512, BF16/TF32, recurrent dropout 0.09, tuned Muon;
- final residue BCE only.

Do not generate a new split. Reuse the exact row constructors from the matched
binary T=1 suite.

Launch command on an L40S:

```bash
python train.py \
  --mode exact_square \
  --out runs/main \
  --channels 128 \
  --updates 2 \
  --steps 10000 \
  --batch-size 512 \
  --dropout 0.09 \
  --optimizer tuned_muon \
  --muon-learning-rate 0.006 \
  --muon-weight-decay 0.1 \
  --muon-warmup-steps 250 \
  --eval-every 500 \
  --compile
```

## Model

State shape is `[batch, channels=128, lanes=4, positions=11]`.

- lane 0: immutable 11-bit modulus embedding;
- lane 1: writable residue/work state, initialized to zero;
- lane 2: writable scratch state;
- lane 3: the current source-bit token, injected at a boundary position;
- learned left/right boundary markers;
- one tied 3x3 ConvGRU cell;
- one shared 1x1 readout from the work lane after all source bits.

Process the exact-square bits from most significant to least significant. For
each of the 22 source bits, inject that bit into lane 3 and apply the same cell
twice. There are therefore 44 cell updates, matching the previous H13 compute
shape and the existing 44-clock exact-square reducer's nominal update count.
No step has a distinct parameter set or coded arithmetic role.

Conceptual pseudocode:

```python
state = zero_work_and_scratch()
for source_bit in exact_square_bits_msb_first:
    for _ in range(2):
        visible = inject_immutable_N_and_current_bit(state, N, source_bit)
        state = shared_convgru(visible)
return shared_bit_readout(state.work)
```

The code shown above describes tensor routing, not a claim that the cell will
learn the desired update. The cell must discover every arithmetic operation
from final labels.

## Controls

1. Existing all-at-once exact-square width-128/44-clock checkpoint:
   19.92% validation, 16.04% and 20.22% audits.
2. If the main run clears 30% validation, run a matched LSD-first source-order
   control. This tests whether the gain is specifically consistent with a
   prefix-style recurrence rather than merely staged input masking.
3. Do not tune width, optimizer, clocks-per-bit, or source order before the
   first validation result.

If the local main arm fails below 30%, run one preregistered architectural
fallback: [`train_scan.py`](train_scan.py) with `--updates 1`. It replaces the
two local ConvGRU microsteps with one tied bidirectional GRU scan over the 11
residue/modulus positions per source bit. This is motivated by the project's
directly supervised serial comparator/subtractor success. It still uses only
final labels and does not receive a coded arithmetic action. Predict above 50%
validation; reject below 30%. Do not run both arms concurrently.

Launch the scan fallback without `--compile` first because the cuDNN GRU path
may graph-break:

```bash
python train_scan.py \
  --mode exact_square \
  --out runs/scan \
  --channels 128 \
  --updates 1 \
  --steps 10000 \
  --batch-size 512 \
  --dropout 0.09 \
  --optimizer tuned_muon \
  --muon-learning-rate 0.006 \
  --muon-weight-decay 0.1 \
  --muon-warmup-steps 250 \
  --eval-every 500
```

## Prediction and decision rule

Prediction before run:

- main validation above 50%;
- both unseen-`N` audits above 40%;
- ambitious success gate: validation at least 90% and both audits at least 85%;
- exactness roughly flat across raw quotient buckets rather than a clock-sized
  frontier;
- slower early learning but a much higher endpoint than all-at-once/44.

On the fixed 5,000-row validation split, trivial exact baselines are 0.44% for
zero, 0.68% for identity, and 4.44% for the unreduced low 11 square bits. The
30% kill gate is therefore well above shortcut scale and above the 21.52%
all-at-once/55-clock result.

Reject this representation if selected validation is below 30% or if exactness
still declines sharply with raw quotient. A 30--50% result is promising but
underfit; only then test three cell updates per source bit. A gain confined to
seen `N` is memorization, not promotion.

If train exact reaches at least 98% while validation is still below 30% at
10,000 steps, continue the same checkpoint for one preregistered additional
5.12M examples. A delayed validation rise counts as grokking only while train
remains near 100%; do not call ordinary joint train/validation improvement
grokking. If train is still underfit, do not spend the continuation on a
grokking hypothesis.

## Required diagnostics

- train, validation, both audits, per-bit exact, and wall time;
- exact by raw quotient using finer buckets through 1024--2047;
- exact by modulus bit length;
- decode work-state bits after every consumed source bit;
- linear probes for the mathematical prefix remainder, fitted after training;
- causal reset/patch of work and scratch lanes at source-bit stages 6, 11, 16,
  and 21;
- compare attention-free state trajectories for same prefix/different suffix
  and same suffix/different prefix examples.

Prefix probes are correlational. State resets and activation patches are the
causal evidence. Neither may tune the selected checkpoint.

## Runtime estimate

The all-at-once 44-clock run took 837 seconds for 10,000 steps. This card has
the same number of ConvGRU applications on a grid half as long, plus cheap bit
injection, so budget 8--14 minutes for the main run and 5 minutes for final
diagnostics on an L40S. Stop after the main result if its validation kill fires.
