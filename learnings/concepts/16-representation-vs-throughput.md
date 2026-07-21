# Representation vs throughput (Easy, 2026-07-21)

> **READ THE LAST SECTION FIRST — "e1 is not a valid ranking signal".** Most of the
> comparisons in this note resolve ~0.4 pp on top of a constant predictor. The method
> lessons stand; the e1 scores do not mean what they appear to.

Session by Claude Code. Twelve scored Easy runs against `depth_d32_k4_ut` as anchor.
Cards live in `solving/submissions/claude_*`. Every claim below is a scored number,
not a smoke result.

## The one thing that worked: place value

`claude_pv_k4_ut` — add two embeddings derived from `input_ids` alone:

- **field** — is this position in the N span, X span, or T span?
- **place** — 1 + distance from the least-significant digit *of its own span*, so the
  units digit of every operand shares `place == 1` whatever the operand's length.

Single change vs the anchor; d=32, 4 heads, shared block ×4, same optimizer/schedule/batch.
Added state ~2.2K elements (~0.0004% of the 500M ceiling).

**Scored on e1, 3 replicates each (the protocol in `solving/experiments/NOISE.md`):**

| card | rep1 | rep2 | rep3 | mean |
|---|---|---|---|---|
| anchor `depth_d32_k4_ut` | 4.67 | 4.67 | 4.67 | **4.67%** |
| C1 `claude_pv_k4_ut` | 5.83 | 5.83 | 5.83 | **5.83%** |

**+1.16 pp, with every replicate identical.** This is the one result in this note that
is fully established. Splits: C1 test 2.70 / ood 9.00 vs anchor 1.30 / 8.00.

On e3 and e5 the picture did **not** hold up — see the noise section. C1's headline e3
number (1.31% vs the anchor's 0.63%) looked like a 2.1× win and **is not real**.

Why it should help: the task is square-and-reduce mod N on digits. With only absolute
position the block must rediscover "index 7 is the units digit of X" from scratch under
a 60s clock, and that mapping *moves* whenever operand lengths move.

## Four things that did not work

All are single changes on top of C1, all scored, all worse.

| card | change | e1 test/ood | e1 mean |
|---|---|---|---|
| `claude_pv_ansplace` | + output place value (distance from sequence end) | 2.00 / 2.00 | **2.00%** |
| `claude_pv_d128` | width 32 → 128, heads 4 → 8 | 2.00 / 2.00 | **2.00%** |
| `claude_pv_tcoupled` | loop count = T+1 instead of fixed 4 | 2.00 / 2.00 | **2.00%** |
| `claude_pv_noabspos` | remove absolute position entirely | 0.70 / 7.00 | **3.83%** |
| `claude_pv_evalk4` | + train-K2/eval-K4 | 3.30 / 1.00 | **2.17%** |
| `claude_evalk4_zeroinit` | zero-init depth embedding | 2.70 / 2.00 | **2.33%** |

C1 itself is 2.70 / 9.00 = 5.83%. The last two get their own sections below. Read the
first four as separate lessons:

1. **Width is not the constraint — the clock is.** d=128 (210K params) ran 394 steps vs
   d=32's 377 and scored a third as well. There are four orders of magnitude of unused
   parameter budget and it is not the bottleneck. Stop reaching for it.
2. **(field, place) is a complete positional code but not a sufficient one.** Every role
   is uniquely identified by (field, place) — provably, no collisions — yet deleting
   absolute position dropped test 3.30 → 0.70. Attention needs a smooth metric for
   relative offsets that a categorical embedding does not provide. Place is *additive*
   value, not a replacement.
3. **T-coupled recurrence failed at the thing it was designed for.** Loop count = T+1
   made OOD *worse* (9.00 → 2.00). Plausible reason: a fixed-K model applies the same 4
   loops to an unseen T=4 row as to training, so it is never off-distribution; coupling
   the count puts OOD rows at a loop depth never trained. Adaptive depth needs the depth
   itself to be trained across a range, not merely indexed.
4. **The residual stream is crowded at d=32.** ansplace added a *correct* signal
   (verified: 60/60 rows reconstruct the answer from (digit, answer_place)) and still
   lost. Five embeddings summed into 32 dimensions is past capacity.

## The noise band — read this before trusting any single run

**The band is not uniform across datasets. Measure it per dataset.**

The manifests pin `seeds: [74]`, so replicates of one file differ only through GPU
nondeterminism and a timing-dependent step count. That turns out to matter enormously on
one dataset and not at all on another.

**e1 — reproducible.** Three replicates each of two different cards returned *identical*
scores (4.67 ×3 and 5.83 ×3). One run per card on e1 is therefore trustworthy, and a
1.16 pp gap is far outside anything the dataset can manufacture.

**e3 — not reproducible.** The same file, `claude_pv_k4_ut`, submitted twice:

| run | score | test / ood | steps |
|---|---|---|---|
| first | **1.31%** | 1.30 / 1.40 | 1505 |
| replicate | **0.69%** | 0.50 / 0.90 | 2041 |

Same code, ~±0.6 pp apart — roughly a factor of two. The anchor scores 0.63% on e3, i.e.
**inside that band**, so C1's apparent e3 win is not supported by evidence.

Why the two datasets differ: e1 fits ~380 optimiser steps into its 60s while e3 fits
1500–2200. The e3 runs sit far enough into training for a differing step count to move the
result, and the step count itself varies with machine load. e1 runs die early and land in
the same place every time.

This also retracts a tempting conclusion. `claude_pv_fast` computes (field, place) in ~8
kernels instead of ~15 with **provably bit-identical output** (verified equal on
train/test/ood at three padding widths) and scored 0.75% on e3 against C1's 1.31%. That
looked like "extra throughput bought overfitting" — but 0.75 and 0.69 and 1.31 are all one
band. **Nothing about throughput is established on e3.**

Practical rules:

- e1 is **reproducible** — one run per card is enough there. But reproducible is not the
  same as meaningful: see "e1 is not a valid ranking signal" below. e1 measures how well a
  model fits the output prior, and it measures that very consistently.
- **Never promote on a single e3 or e5 run.** Treat any e3 gap under ~0.7 pp as a tie.
- e3/e5 are the *honest* Easy datasets (no answer-space collapse) but are low-signal and
  high-variance, so they need replicates to say anything at all.
- Net: **rank on Medium.** Easy is for correctness checks and for measuring the harness.

## The two e1 wins do not stack — they anti-stack

`claude_pv_evalk4` combines the two best e1 changes: place value (C1) and train-K2/eval-K4.
Both parents reach ood 9.00 and differ mainly on test, so stacking looked free.

| card | test | ood | mean |
|---|---|---|---|
| C1 place value (train4/eval4) | 2.70 | **9.00** | 5.83% |
| `depth_d32_k2_ut_evalk4` (train2/eval4) | 4.70 | **9.00** | 6.83% |
| **combined** | **3.30** | **1.00** | **2.17%** |

Place value did exactly its job — test rose 2.70 → 3.30, the best test of the three. But
**OOD fell 9.00 → 1.00** and sank the card.

Reading: *train/eval depth mismatch is only free while the representation is weak.* The
plain UT model learns something diffuse enough to survive being run at a depth it never
trained at. Once the block is given a sharp positional code it commits to a specific
loop count — a 2-loop algorithm keyed to place values — and running 4 loops at eval
corrupts it.

Consequence for the shortlist: **eval-only extra depth and strong input structure are
competing strategies, not complementary ones.** Pick one. And since OOD is half the Easy
score and the whole point of Hard, a change that trades 8 pp of OOD for 0.6 pp of test is
a bad trade even when the arithmetic happens to work out.

## Reusable technique: deriving structure from `input_ids`

Legal (no dataset access, no trapdoor) — markers N/X/T are public token ids 2/3/4,
digits are 7..16, PAD is 0.

- `field = ((ids >= 2) & (ids <= 4)).cumsum(1).clamp(0, 3)` — markers appear in order
  N < X < T, so one cumsum yields 1/2/3 (the sum of three cumsums is the cumsum of the sum).
- `place` — spans are contiguous, so a digit's span ends one position before the next
  *boundary* (next marker, or the first pad). One reverse-cummin gives every position its
  next boundary; place is then arithmetic. No Python loop, no per-field masked reduction.

See `claude_pv_fast/submission.py`. Bit-exact against the naive form, ~2× faster.

## Where the answer actually lives

`collate_squaring_mod` (`competition/data/squaring_mod.py:92`):

    target_positions[row, :target_len] = arange(input_len - target_len, input_len)

The answer is read off the **last len(answer) positions of the input** — positions that
physically hold T-span tokens. So the final real position is always the answer's units
digit, and distance-from-end *is* the output place value.

Note this format (`separate_input_output=True`, the `bidirectional` data roots) puts **no
answer in `input_ids`** — the sequence is just `[N, digits, X, digits, T, digits]`. The
other tokenizer path in the same file appends `ANS` + answer digits into `input_ids`;
that path is not what these manifests use. Do not derive fields by clamping the marker
cumsum at 3 without checking which format you are on — under the causal path that silently
folds the answer region into the T span.

Exploiting distance-from-end directly (`ansplace`) *lost*, so knowing where the answer
lives has not yet been converted into a win. It stays on the backlog for a wider model
where the residual stream can afford it.

## Depth embeddings are loop IDs, not learned features

`depth_d32_k2_ut_evalk4` (e1 best, 6.83%) declares `nn.Embedding(K_MAX=4, D_MODEL)` but
trains with `k_loops = 2`, so slots **k=2, k=3 never receive a gradient** — yet eval runs
4 loops, adding default **N(0,1)** vectors into a d=32 stream on loops 3–4. That looked
like a bug worth cleaning up, so `claude_evalk4_zeroinit` zero-inits the depth embedding
(one change, nothing else touched).

It collapsed:

| card | test | ood | mean | train acc @60s |
|---|---|---|---|---|
| `depth_d32_k2_ut_evalk4` | 4.70 | 9.00 | **6.83%** | — |
| `claude_evalk4_zeroinit` | 2.70 | 2.00 | **2.33%** | **9.0%** (highest seen) |

The zero-init card **trains best and generalises worst.** The untrained random vectors
were doing real work: their only job is to make loop k *distinguishable* from loop j.
Zero them and every loop receives the same (null) depth signal, the shared block cannot
tell iterations apart, and it degenerates toward a plain repeated map that memorises.

**Distinctness matters; being trained does not.** Do not "clean up" untrained depth slots.

## What to try next

- **N-conditioning done digit-locally** — cross-attention from each X digit to N's digits,
  aligned by place id, rather than a pooled FiLM vector. FiLM was rejected on e1/e5, but
  that tested the wrong form: mod-N reduction is digit-local, not a smooth affine
  modulation of a pooled summary. This is the highest-EV untested axis.
- **Label-free aux via the `auxiliary` return** — `forward` returns `(logits, auxiliary)`
  and `auxiliary` is handed straight to `training_loss`, so ponder cost, per-loop
  consistency ‖h_k − h_{k−1}‖ and entropy are all reachable today. Untried. (Label-*aware*
  intermediate supervision is structurally blocked — see `11-ideas-backlog.md`.)
- **Anything that raises OOD.** Every card here is stuck at ood ≈ 9.00 on e1 or falls off
  a cliff; nothing has pushed past it. The test split is where place value helps and OOD
  is where the ceiling is.
- Do **not** revisit: wider models, removing absolute position, T-coupled loop counts,
  output place value at d=32, zero-init depth embeddings. All scored, all lost.

## e1 is not a valid ranking signal — the answer space collapses

Computed locally from `data.squaring_mod.trapdoor_squaring_mod` over every x in [1, N),
no quota spent. For e1 (N = 323 = 17 × 19):

| T | distinct answers | majority-class baseline |
|---|---|---|
| 1 | 89 | 1.24% |
| 2 | 49 | 2.48% |
| 3 | 29 | 4.97% |
| **4 — the ood split** | **19** | **9.94%** |
| 5, 8, … | 19 | 9.94% |

The image of x ↦ x^(2^T) mod 323 **saturates at 19 values for every T ≥ 4**. λ(323) =
lcm(16, 18) = 144, and the squaring map's image shrinks until it hits a fixed point, so
the ood split has only ~19 possible answers and the single most common one covers ~10%
of it.

Consequences, and they are severe:

- **Every card's ood ≈ 9.00% is at or below the 9.94% you get from always emitting the
  most common answer.** No card has ever beaten trivial on e1's ood split.
- The e1 combined trivial baseline is ≈ **6.42%** (2.90% test, 9.94% ood). The best card
  ever recorded here is **6.83%**. The whole leaderboard climb from ~1% to 6.83% is
  mostly models learning the output prior.
- `claude_pv_k4_ut` (5.83%) and `depth_d32_k4_ut` (4.67%) are both **below trivial**.
- It explains the attractors seen all session: unrelated architectures landing on exactly
  2.00/2.00, or exactly 4.70/9.00, are converging on the same prior-fitting solutions.
  `claude_pv_d128_k8` scored 4.70% test on a 3.1% train fit — scoring above its own
  training accuracy, which only makes sense if it is not computing the answer at all.

**This does not invalidate the method lessons** (replicate before promoting; e1
reproducible / e3 not; depth codes must be distinct; representation and eval-depth
tricks anti-stack). Those are about how the harness behaves. It does invalidate using e1
*score* to rank architectures.

### Why e3/e5 look "stuck" and e1 looks healthy — backwards

e3/e5 vary N, so no such collapse occurs and there is no strong prior to exploit. Their
0.3–1.4% scores are **honest** — they say the models cannot do the task. e1's 5–7% is
**inflated**. The correct reading of this repo's history is not "we got to 6.8% on e1 and
are stuck at 1% on e5"; it is "we have never done the task, and e1 was hiding it."

### Medium is not broken — use it

| dataset | distinct answers | majority baseline |
|---|---|---|
| e1, T=4 (ood) | 19 | 9.94% |
| **m1 (N = 10403 = 101 × 103), T = 4/8/16** | **1351** | **0.08%** |

m1's prior is worth 0.08%, so anything meaningfully above zero there is real computation.
**Rank on Medium, or on e3/e5, but not on e1.**

Before trusting any future dataset, compute its majority-class baseline first — it is
free, local, and takes seconds:

```python
import collections
from data.squaring_mod import trapdoor_squaring_mod
c = collections.Counter(trapdoor_squaring_mod(x, T, p, q) for x in range(1, p * q))
print(len(c), max(c.values()) / sum(c.values()))
```

## Majority-class baselines for every dataset (computed 2026-07-21)

Run this before trusting any score on a dataset. Free, local, seconds.

| dataset | N | distinct answers | majority baseline |
|---|---|---|---|
| **e1 ood (T≥4)** | 323 | **19** | **9.94%** ← broken |
| e1 test (T=1/2/3) | 323 | 89/49/29 | 2.90% |
| m1 (T=4/8/16) | 10403 = 101×103 | 1351 | **0.077%** |
| m2 (T=4) | 38021 = 193×197 | 649 | 0.168% |
| m2 (T=8/16) | 38021 | 199 | 0.673% |
| m3 (T=2) | 11/13/15-bit | 1588 | 0.694% |
| m4 (T=8) | 14/18/22-bit | 1938 | 0.333% |
| m5 (T=2/4/8) | 12/14/16-bit | 4349 | 0.262% |

Every Medium dataset is healthy — no collapse. **m1 has the weakest prior (0.077%) and
the widest T range (4/8/16), so it is the best Medium discriminator.** m2 partially
collapses at T≥8 (199 distinct), so prefer m1 over m2 for fixed-N work.

Use this table to sanity-check any score before celebrating it. A first datapoint:
an m5 run scored **0.089%**, which is *below* m5's 0.262% prior — i.e. worse than
guessing the most common answer.

## The LR schedule is broken on Medium (affects every card in the repo)

`_build_scheduler`, copied verbatim into every submission here, estimates the training
horizon as:

    t_max = max(100, int(spec.training_time_seconds * 8))

i.e. it assumes **~8 optimiser steps per second**. Measured reality:

| tier | assumed t_max | steps actually completed | steps/sec |
|---|---|---|---|
| Easy e1 (60s) | 480 | ~380 | ~6 |
| **Medium m1 (600s)** | **4,800** | **44,993** | **~75** |

The assumption holds on Easy (79% of the schedule consumed — fine) and is off by
**9.4×** on Medium. `CosineAnnealingLR` is *periodic* past `T_max`, so on Medium the
LR finishes annealing around step 4,560 and then **sawtooths between 0 and the base
3e-3 for the remaining ~40,000 steps** (measured mean 1.5e-3). The optimiser is
repeatedly kicked out of every minimum it finds.

This matches the observed Medium failure exactly. Both `claude_pv_fast` on m1 and the
other agent's card on m5:

- loss falls from ~11.5 to ~2.2 within a few hundred steps, then sits flat for 45,000+
- train exact accuracy oscillates 0.0–1.6% with no trend
- final scores land on the majority-class prior (0.083% vs m1's 0.077%)

**ln(10) = 2.303.** A plateau at 2.13–2.20 means the model is barely better than
uniform over digits — it learned the marginal digit distribution and nothing else.

Do **not** conclude from those runs that the architecture cannot represent the task.
That has not been tested yet; the optimiser never got a stable descent phase.

### Fix: anchor the schedule to wall clock, not a guessed step count

`claude_pv_fast_tsched` / `claude_pv_tadapt_tsched` replace the step-based schedule with
a `LambdaLR` whose progress is `elapsed_wall_time / spec.training_time_seconds`. It
anneals exactly once over the real budget no matter how many steps that turns out to be,
so it is correct at 60s, 600s and 3600s alike, and stays correct when a wider or deeper
card changes throughput. Verified: 199,186 steps inside a compressed window, LR
monotonically non-increasing through the second half, ending at the 1% floor.

This is the single highest-leverage change found today and it applies to **every card in
the repo**, including the other agent's. Any Medium or Hard result produced with the old
scheduler should be treated as uninformative about architecture.

## HARD H1 RESULT: it grokked the training set and generalised zero

`claude_hard_h1` (d=2048, 50.5M params, fixed K=4, wall-clock LR, std=0.02 init).
3600s, **190,017 steps**. Leaderboard score 0.03%; metrics report mean exact 0.0000%.

    step      1   loss 3.322   train exact   0.00%
    step  7,900   loss 2.179   train exact   0.00%   ┐
    step 47,900   loss 2.179   train exact   0.00%   │ ~60,000-step plateau
    step 55,900   loss 2.167   train exact   0.00%   ┘
    step 63,900   loss 1.974   train exact   0.00%   ← transition begins
    step 71,900   loss 1.191   train exact   9.40%
    step 87,900   loss 0.227   train exact  75.40%
    step 167,900  loss 0.0000  train exact 100.00%

Evaluation, all three H1 splits: **0.0000%**.

| split | exact | loss |
|---|---|---|
| test | 0.0000% | 15.836 |
| ood_t | 0.0000% | 16.170 |
| ood_n_t | 0.0000% | 16.387 |

Eval loss ~16 against a ln(10)=2.303 uniform baseline means the model is not merely
wrong, it is **confidently** wrong — it emits memorised answers with high confidence.

### What this settles

1. **Capacity and optimisation are solved.** Train loss reaches exactly 0.0000 and
   train exact-match 100%. The architecture can represent and fit this task perfectly.
   Everything that remains is inductive bias / generalisation.
2. **The Medium "plateau" was pre-grokking, not underfitting.** The transition begins
   at ~64,000 steps. The best Medium run completed **58,060**. Every Medium run in this
   repo — both agents — stopped a few thousand steps short of the phase transition and
   was read as a hopeless flatline. It was not.
3. This retracts the reasoning in the Hard card's own docstring ("the model is
   UNDERFITTING… the answer to underfitting is capacity"). The diagnosis of the symptom
   was right; the mechanism was wrong. More width was not what tipped it over — more
   *steps* were.
4. **H1 has three eval splits** (`test`, `ood_t`, `ood_n_t`), so Hard explicitly scores
   unseen T and unseen N+T. Generalisation is the whole benchmark.

### What to do next

The failure is now precisely located: the model has no pressure to learn a *reusable
single step*, so with enough steps it takes the lookup-table solution. Levers, in order
of expected value:

- **Weight decay.** We ran 0.1. The grokking literature finds decoupled weight decay is
  the main thing that moves a network off the memorising solution onto the generalising
  one. Try 1.0 / 3.0. Cheapest possible experiment, single constant.
- **Force iteration architecturally.** Make state space = output space (working value
  lives at the tail positions where the answer is read), apply ONE shared step T times,
  and **re-quantise the state toward one-hot digits between steps** so error cannot
  accumulate across iterations. A memoriser cannot exploit that; an algorithm can.
- **Input injection every loop** (re-feed x and N each iteration) — standard UT/DEQ
  practice, currently absent.
- **Label-free entropy aux** via the `auxiliary` return (reachable today, still untried):
  penalise entropy of intermediate digit distributions, i.e. train "stay discrete
  between steps" directly rather than hoping the architecture enforces it.
- **T=1 rows are direct single-step supervision** and nothing has exploited that. T=2/3
  supervise its composition. That is depth supervision, not a nuisance parameter.

Do NOT reach for more capacity. It is not the constraint.

### Process note

Run the grokking check LOCALLY. `competition/` generates data and runs the real training
loop offline with no quota. A 60,000-step plateau followed by a transition is invisible
inside a 600s clock but cheap to find on a local GPU overnight. Most of what this session
spent quota on was answerable for free.
