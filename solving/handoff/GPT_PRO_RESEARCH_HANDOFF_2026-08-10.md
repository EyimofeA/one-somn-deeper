# GPT Pro research handoff — 2026-08-10

Paste the prompt below into the existing GPT Pro conversation. It is also
self-contained enough for a fresh conversation.

---

You are taking over high-level research direction for the **One Layer Deeper**
competition project. Continue from our prior conversation if one exists, but
treat this prompt as a verified-state reset: preserve useful prior ideas, audit
them against the facts below, and explicitly identify any conflict. If this is
a fresh chat, begin from first principles.

Your role is **principal research theorist and teacher**, not an architecture
idea generator. Diagnose the bottleneck, derive a focused research program,
and teach me enough to judge it. Do not write or submit code yet. Do not rent
compute or recommend spending competition quota until an experiment has a
causal hypothesis, a discriminating control, a prediction, and a promotion or
kill threshold.

## Project and authoritative sources

- Research repository: https://github.com/EyimofeA/one-somn-deeper
- Upstream competition: https://github.com/tilde-research/one-layer-deeper
- Problem: https://onelayerdeeper.ai/problem
- In the research repo, read in this order if browsing GitHub is available:
  `AGENTS.md`, `PITFALLS.md`, `RESEARCH_PROTOCOL.md`, `solving/STATUS.md`,
  `HYPOTHESES.md`, the latest `learnings/sessions/`,
  `learnings/concepts/01-the-problem.md`, and the relevant experiment `NOTE.md`
  files cited by `solving/STATUS.md`.
- Repository evidence and evaluator metrics override this prose if they differ.
  Report the discrepancy instead of silently choosing one version.

## Competition from first principles

For public Easy/Medium, the input encodes decimal digits of `(N, x, T)` and the
target is the exact decimal representation of

`s_0 = x`, `s_{t+1} = s_t^2 mod N`, output `s_T`.

The model receives a token sequence like `N digits(N) X digits(x) T digits(T)`
and predicts all answer digits. One wrong digit makes the entire example
wrong. T=1 is therefore the one-step map `f_N(x) = x^2 mod N`.

Hard is hidden and explicitly warns that aspects of the recurrence may change;
we must not assume it is repeated squaring. Its exact family is unpublished.
Hard ranks by consecutively certified exact rungs `T = 1,2,4,8,16,32,64`:
first seen-modulus Max T, then unseen-modulus Max T, then first-unsolved-rung
accuracy. A single error prevents certification of that rung. A useful Hard
solution probably needs a generic learned serial-computation mechanism, not a
squaring-specific trick.

Submission constraints include one self-contained `submission.py` of at most
256 KiB, at most 500M persistent state elements, random learned weights, an
unbroken GPU/autograd path, evaluator-owned data/training/backward, and strict
wall clocks. Hard-coded forward algorithms, arithmetic solvers, factorization,
Euler-phi shortcuts, data inspection, data augmentation, custom training loops,
and participant-controlled backward/nested training are forbidden. Diagnostic
trace supervision may be used to understand mechanisms locally, but it is not
automatically legal evidence for a final-label competition submission.

Budgets: Easy 60 training seconds and 60 accepted attempts/day; Medium 600
seconds and 6/day; Hard 3,600 seconds and 1/day. Hosted evaluation uses H100s.
The recorded deadline is August 31, 2026 at 10:00 PM PT; recheck upstream
before deadline-sensitive action.
There is **no active rented GPU now**. We previously used one 48 GB Prime L40
at $0.86/hour and can rent similar compute later if justified; availability and
price must be rechecked. Do not use my Mac GPU.

## How this project experiments

We try to use a Karpathy-style loop rather than architecture roulette:

1. state one causal question and the smallest experiment that can answer it;
2. preregister the expected outcome, alternative outcome, promotion gate, and
   kill condition before looking at results;
3. change one mechanism at a time against a parameter/budget/data-matched
   control with fixed train, held-out-x, and unseen-N splits;
4. demand training-fit, exact-example accuracy, per-digit localization, and
   multiple seeds where the seed-0 gate survives;
5. use diagnostic trace supervision only to test whether a mechanism is
   learnable, then retest the decisive claim with legal final-label-only loss;
6. screen locally, then use Easy, then Medium, and reserve Hard for a validated
   source with explicit owner approval;
7. preserve source SHA-1, config, raw metrics, logs, checkpoints, plots, and an
   experiment `NOTE.md`, then record the conclusion and rejected follow-ups.

We care about experiments that distinguish explanations. A higher score without
a mechanism-changing control is not sufficient evidence.

## What we are trying to solve

The immediate scientific gate is not long recurrence. It is:

> Learn T=1 `x^2 mod N` from legal final labels and generalize exactly to both
> unseen `x` and unseen `N`.

Until this reaches a qualitatively strong regime, T>1 work confounds learning
the one-step operation with recurrent-state stability. Require exact seen-N and
unseen-N T=1 evidence before making recurrence or Hard claims.

The core question is why models that fit every training example still achieve
only roughly 4–19% unseen-N exact accuracy, and what legal computational
abstraction or objective makes the transition identifiable and learnable.

## Verified experimental findings

Keep **competition results**, **legal final-label research**, and **diagnostic
trace-supervised mechanisms** separate.

### 1. Direct final-label learners memorize

- A roughly 2.19M-parameter MLP reached 100% train exact in three seeds but only
  **3.79–3.92% unseen-N exact**.
- A roughly 1.79M-parameter four-layer Transformer also reached 100% train exact
  in three seeds but only **4.06–4.26% unseen-N exact**.
- Attention gave less than half a percentage point and did not change the
  learned mechanism. This is not primarily an optimization-to-training-fit
  failure.

### 2. Latent topology helps modestly, not decisively

In a matched T=1 tournament:

- answer/register baseline: **4.62% held-out-x / 8.64% unseen-N**;
- global latent: **10.50% / 16.36%**;
- structured LSD-aligned tape: **12.18% / 17.06%**.

A later matched representation comparison gave decimal **11.76% held-out-x /
18.69% unseen-N**, binary **2.10% / 4.21%**, and two 4-bit limbs **11.34% /
14.49%**. All but the binary arm fit training perfectly. More latent structure,
binary digits, and simple limbs do not solve transfer.

### 3. Trace-supervised arithmetic proves mechanisms can be learned

On controlled small-number diagnostics, learned Square-to-Reduce components
can generalize strongly when intermediate arithmetic states are supervised and
the reducer is trained on the actual square-generated state distribution. One
two-digit sanity reproduction reached **100% unseen-N raw-square exact,
95.79% unseen-N T=1, and 92.52% T=8**.

Training the same reducer on generic `qN+r` states versus real square-generated
traces previously changed held-out reduction from about **46.96% to 95.56%**
and VDF T=8 from **33.88% to 89.02%**. Distribution matching is real.

A shifted learned long-division diagnostic achieved at least **99.9023%**
remainder exactness through quotients up to 99,999,999 using fixed sublinear
opportunities. But it directly supervised arithmetic transitions and used a
fixed high-to-low decimal-shift schedule. It is mechanism evidence, not a legal
submission design.

### 4. The generic Neural GPU localized a concrete failure

We built an 89,902-parameter recurrent grid with eight LSD-first positions,
six 64-wide scratch lanes, learned left/self/right communication, and one tied
GRU-like cell repeated for 16 microsteps. It trained only on final raw-square
digits over 8,000 `x` and evaluated on 2,000 unseen `x`.

After 12,000 updates it reached **12.9125% train exact, 4.0000% unseen exact,
and 75.6125% unseen digit accuracy**. LSD-to-MSD positional accuracies were
approximately `100, 99.8, 96.1, 29.05, 15.2, 68.15, 96.95, 99.65%`.

The outer digits transferred while the central cross-term/carry columns
collapsed. This is evidence for a cross-product accumulation and temporal
credit-assignment failure, not merely leading-zero exploitation.

Historical Transformer work showed answer-only **0.75%**, carry-supervised
**71.4%**, carry-plus-diagonal **80.1%**, and a shuffled-carry control near 1%.
But attaching only a terminal carry head to the new Neural GPU moved matched
50k unseen exact from **3.85% to 6.25%**. Its final state could report carry
somewhat without the recurrent dynamics consuming carry at the required time.

### 5. Legal competition transfer failed

The legal six-lane Neural GPU card removed all carry targets, traces, solvers,
modulus operations, and arithmetic phases. It used final-label loss and four
tied local microsteps per requested T. Local Easy e5 scored **0.6667% mean**.
Hosted Easy job `ff081248-f600-40c6-a133-045783f76c68`, exact source SHA-1
`c436691686c76e406445484b64849ac06eac5cac`, scored **0.3333% mean**
(`0.5000%` test, `0.2000%` OOD), certified no rung, and completed only 532
updates. The nested recurrence was also wall-clock/dispatch inefficient.

The strongest known hosted Easy result in the repo is a different Fable
T-cap/AdamW source at **8.50% mean**, but it certified no T=1 rung. Its Medium
m1 result was **0.0333%** and no rung. Known Hard attempts remain effectively
zero: the canonical register source scored **0.0500%** with **0/768** at both
seen-N and unseen-N T=1; a T=1-weighted/SAM source scored **0.02333%**, also
0/768 on both profiles; Fable v2 scored **0.0467%** and certified nothing.
Audit the current repo before treating these as the complete submission list.

## Current interpretation, open to your critique

Evidence currently points away from “just use a larger Transformer,” optimizer
tuning, more scratch lanes, longer training, binary tokenization, diffusion-like
refinement, or broad T curricula. The bottleneck appears to combine:

1. severe non-identifiability/underspecification from sparse modular final
   labels;
2. failure to construct reusable intermediate arithmetic state;
3. failure to make carry/borrow information causally available when needed;
4. the need for a generic learned scheduler or phase state without hard-coding
   an arithmetic algorithm;
5. exactness and hosted wall-clock constraints.

The live architectural thesis is a generic tied recurrent computation with
immutable `N/x` lanes, mutable local state, and causally consumed carry-like
state. Candidate abstractions include a 2-D local grid with a learned
phase/pointer state or a prefix-register scan that keeps intermediates near N.
Neither is validated, and neither may silently encode a handwritten schedule.

## What I want from you

Produce one rigorous research memo with these sections:

1. **Teach the problem from first principles.** Explain modular squaring, why
   unseen `x` differs from unseen `N`, why T=1 is the decisive gate, why exact
   sequence accuracy is harsh, and why Hard is substantially harder. Use one
   small worked example.

2. **Audit our interpretation.** Separate verified evidence, inference, and
   speculation. Identify the three most important conclusions that really
   follow and the three places where we may be over-interpreting results.

3. **Explain why this is hard for neural networks.** Discuss function
   identifiability, sample sparsity, interpolation versus algorithm learning,
   representation, carry credit assignment, recurrence stability, and
   optimization. Explicitly decide whether the dominant problem is data,
   objective, architecture, optimization, or some interaction.

4. **Give at most three research directions.** Derive each from a diagnosed
   failure. For each provide: mechanism, why legal in principle, smallest
   discriminating experiment, matched control, predicted observation if the
   hypothesis is right, falsifier, compute estimate, and promotion/kill gate.
   Rank them. Do not give an unranked architecture list.

5. **Design the next experiment.** Choose exactly one experiment. Keep it T=1
   and final-label-only for the decisive test, with diagnostic auxiliaries only
   in separately labeled mechanistic controls. Specify dataset construction,
   train/held-out-x/unseen-N splits, seeds, model/control, loss, measurements,
   plots, wall-clock budget, and pre-registered decision rule. Explain how it
   distinguishes causal carry use from merely decodable carry.

6. **Competition strategy.** Explain what would have to be true before moving
   Easy -> Medium -> Hard, and how a method aimed at public repeated squaring
   might transfer to Hard's unknown nearby recurrence. State what not to spend
   quota on.

7. **My learning pathway.** Build me a staged curriculum so I can understand
   and challenge this project. Start at my current practical level but do not
   assume I understand the jargon. Cover modular arithmetic, digit algorithms
   and carry/borrow, neural algorithm learning and grokking, Transformers versus
   recurrent/state-space computation, Neural GPUs/cellular computation, latent
   state and credit assignment, systematic generalization, exact-match evals,
   and the competition rules. For every stage give:
   - the question I should be able to answer;
   - a concise explanation;
   - one high-quality reading or lecture;
   - one small pencil-and-paper or PyTorch exercise;
   - the relevant file/experiment in our repo;
   - a mastery check.
   End with a dependency-ordered two-week study plan requiring no rented GPU.

8. **Questions for us.** Ask only the missing questions whose answers could
   materially change your preferred next experiment.

Write for an intelligent technical user who wants first-principles clarity,
not jargon. Use equations and pseudocode where they help. Be blunt about
negative evidence. Do not claim that a diagnostic is competition-legal merely
because its neural components were learned. Do not claim recurrence progress
without exact unseen-N T=1 evidence.

---
