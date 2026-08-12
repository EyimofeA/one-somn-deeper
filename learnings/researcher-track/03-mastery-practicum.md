# Mastery practicum: become the project’s research director

This is the recommended version. It uses mastery gates rather than a calendar.
You advance by producing artifacts and defending decisions. A coding agent may
help only where the exercise explicitly permits it.

## Operating rule

No new GPU rental, model sweep, or competition submission until Gate 2. No
recurrence research until Gate 4. Already-running hosted jobs may finish and be
recorded, but their completion does not authorize another run.

## Your research notebook

Create one entry per exercise with:

```text
QUESTION
MY ANSWER BEFORE READING/RUNNING
EVIDENCE USED
WHAT I CHANGED MY MIND ABOUT
UNRESOLVED QUESTION
NEXT DECISION
```

An answer generated wholly by an agent does not count. You may ask an agent to
critique your answer after you write it.

## Gate 0 — mathematical and benchmark literacy

### Exercises

1. By hand, compute `s₀, s₁, s₂, s₃, s₄` for `(N=77, x=38)` and another case of
   your choice. Prove reduction after each square preserves the final result.
2. For one four-digit x, make a table of product pairs `(i,j)`, destination
   column `i+j`, raw column sum, emitted digit, and carry.
3. Reduce the square modulo a four-digit N by long division. Record comparison,
   quotient digit, subtraction, and borrow states.
4. Tokenize three example prompts and answers. Identify which logits are scored.
5. Explain held-out x, unseen N, longer width, larger T, and hidden Hard as five
   different generalization questions.
6. Calculate whole-answer accuracy `p^d` for three values of `p` and `d`. Explain
   why independence is only an illustration.

### Oral test

- Why is T=1 necessary but not sufficient?
- Why can fixed-N success be table interpolation?
- Which hidden intermediate variables does the final remainder erase?
- Why do middle square digits contain more evidence about algorithm learning?
- Why might a perfect Easy squaring circuit fail hidden Hard?

### Pass condition

Answer every question without code or notes and make no confusion between T,
recurrent microsteps K, sequence length, or number of digits.

## Gate 1 — code and gradient literacy

### Lab 1: read a submission

Open the baseline and annotate:

- every persistent parameter;
- every tensor shape;
- where positions enter;
- what information each output token can access;
- where loss enters;
- what one optimizer step changes;
- what is evaluator-owned versus participant-controlled.

Repeat on the causal-message Hard source. Find at least five mismatches between
its comments/intention and actual computation. Expected examples include fake
“rows,” global-attention bypass, tape mean-pooling, T as conditioning rather
than execution, and late-loss confounding.

### Lab 2: implement tiny components

Without copying project code, implement and unit-test:

1. an MLP;
2. one pre-norm attention block;
3. a vanilla RNN cell;
4. a GRU cell;
5. a tied recurrent block applied K times;
6. a NAC/NAU-like additive unit;
7. a bilinear multiplicative cell;
8. left/right message shifting on a digit tape.

For each, print parameter count, input/output shapes, and gradient norm after a
toy loss. Use CPU only.

### Lab 3: loss mechanics

Using fixed fake logits, implement:

- token CE;
- per-example sequence-balanced CE;
- smooth worst-digit loss;
- final-only recurrent loss;
- late-step final-answer loss;
- group reweighting.

For each loss, manually perturb one logit and predict the gradient change before
running autograd. State which project failure the loss could address and which
it cannot.

### Pass condition

Given an unfamiliar 200-line submission, trace one example forward and explain
which parameters receive gradient from a selected output digit. Produce a CPU
test proving whether a claimed message channel affects future logits.

## Gate 2 — experimental judgment and agent direction

### Exercise: diagnose old experiments

Select six repository cards:

- direct MLP;
- direct Transformer;
- raw-square Neural GPU;
- 50k carry auxiliary pair;
- shifted reducer;
- one hosted Hard run.

For each, fill:

| Field | Required answer |
|---|---|
| Question | The actual causal question |
| Anchor | Exact source/config/result compared |
| Change | One changed variable, or list confounds |
| Primary metric | Including split and denominator |
| Result | Number, not adjective |
| Establishes | Narrow supported conclusion |
| Does not establish | Tempting overclaim |
| Next decision | Stop, replicate, intervene, or promote |

### Exercise: write a complete card

Write the proposed bilinear-versus-affine raw-square card:

```text
CARD
QUESTION
ANCHOR
ONE CHANGE
KEEP FIXED
PREDICTED TRAIN CURVE
PREDICTED HELD-OUT CURVE
PREDICTED DIGIT-POSITION EFFECT
FALSIFIER
SEED-0 KILL GATE
REPLICATION GATE
BUDGET
REQUIRED ARTIFACTS
LEGAL STATUS
```

### Agent-management trial

Give a deliberately weaker agent this contract:

```text
Role: implementation engineer, not research director.

Question: Does a generic low-rank bilinear interaction improve raw-square
cross-product formation over a parameter-matched affine/GLU control?

Before coding, return:
1. your restatement of the causal contrast;
2. every variable that must remain fixed;
3. tensor shapes;
4. parameter/FLOP matching plan;
5. CPU tests;
6. actions you are not authorized to take.

After approval, implement only the two cells and tests. Do not train, rent GPU,
submit, alter data, add carry labels, change optimizer, or edit unrelated files.
```

Grade the response:

- 2 points: correct causal contrast;
- 2: finds confounders;
- 2: exact shapes and tests;
- 2: respects scope and rules;
- 2: smallest understandable diff.

Below 8/10, rewrite the contract rather than letting the model improvise.

### Pass condition

Defend the card in a ten-minute oral exam. When challenged with “why not a
larger Transformer?”, “why not more data?”, “why not train longer?”, and “why
not carry supervision?”, answer using existing evidence and the intended
estimand—not preference.

Passing Gate 2 permits a bounded CPU-only research implementation. It does not
yet permit GPU rental or competition quota.

## Gate 3 — run and audit a reproducible CPU experiment

### Required experiment

Run a tiny synthetic arithmetic task that finishes on CPU. The scientific
result is unimportant; the operational standard is the test.

Before running:

1. create the experiment directory;
2. write prediction and falsifier;
3. freeze data split and seed;
4. record environment and command;
5. run shape, finite-loss, backward, tiny-overfit, and deterministic-eval tests;
6. ensure plots can be generated from fake metrics.

During running, record:

- step, elapsed seconds, examples seen;
- train loss and exactness;
- held-out loss and exactness;
- gradient norm;
- parameter norm;
- throughput;
- seed and config.

After running:

1. generate fixed train/test loss and exactness plots;
2. compare with preregistration before explaining;
3. mark result confirmed, refuted, or unclear;
4. state one next decision;
5. write NOTE, config, source pointer, and checksums;
6. commit only the experiment artifacts allowed by the repository layout.

### Debugging decision tree

```text
Does one batch overfit?
├─ no → bug, insufficient capacity, optimizer, or loss issue
└─ yes
   Does train fit but held-out fail?
   ├─ yes → shortcut/identifiability/distribution issue
   └─ no
      Does exact lag token accuracy?
      ├─ yes → local/correlated output failure
      └─ no → inspect data, metric, and evaluation code
```

### Pass condition

Another person can reproduce the run from the card without asking you what you
meant. You can identify whether a deliberately injected error belongs to data,
model, loss, optimizer, metric, or orchestration.

Passing Gate 3 permits one preregistered seed-0 GPU diagnostic only after a
cost and lifecycle review.

## Gate 4 — neural arithmetic mechanism experiment

### Stage A: product formation

Run the bilinear/NMU-like cell against a matched affine/GLU cell on raw-square
final labels. Keep representation, recurrent depth, optimizer, batch, data
order, update budget, initialization scheme, and decoder fixed.

Required evidence:

- train and unseen exact curves;
- LSD-to-MSD accuracy;
- error by number of cross-products;
- parameter/FLOP/throughput comparison;
- at least three seeds only if seed 0 clears its gate.

Kill below 20% unseen exact or below ten points over control. Do not add modular
reduction to rescue a failed primitive.

### Stage B: causal message consumption

Only if Stage A passes, compare terminal-only and causal-message arms. Require
matched parameter count, FLOPs, data, recurrence, and decoder. Intervene with
zero, delay, swap, and equal-dimensional random-channel ablations.

The claim passes only if messages are decodable, intervention changes behavior,
damage exceeds matched ablation, and failures concentrate where the mechanism
predicts.

### Stage C: legal final-label T=1

Remove arithmetic auxiliaries. Train the unchanged core on final `x² mod N`
labels. Both control and causal models must reach comparable >=99.5% training
exactness before interpreting unseen-N differences.

### Pass condition

You can distinguish these outcomes:

1. product diagnostic fails → multiplicative cell did not solve formation;
2. product passes, causal diagnostic fails → message timing is not the missing
   primitive;
3. diagnostics pass, legal labels fail → identifiability/credit is dominant;
4. legal model improves but intervention does not erase gain → architecture may
   help, causal explanation rejected;
5. legal model passes intervention and unseen-N gates → promote to local mirror.

Only the fifth permits competition consideration. Only near-exact T=1 permits
recurrence research.

## Gate 5 — recurrence research

### Required conceptual defense

Explain teacher forcing, exposure bias, autonomous state distribution, error
compounding, tied versus untied transitions, computational depth, receptive
field, fixed-point behavior, and hidden Hard transfer.

### Experiment ladder

1. Frozen one-step transition on true states.
2. Autonomous T=2.
3. Autonomous T=4 and T=8.
4. Train-depth/eval-depth grid.
5. State perturbation recovery.
6. State drift, entropy, and confidence curves.
7. Two unrelated synthetic recurrence families for generic-executor evidence.

Test one stabilization at a time:

- state noise for recovery;
- scheduled sampling for exposure;
- spectral/contractive constraints for drift;
- normalization for scale;
- late final-answer loss for credit;
- adaptive halting for compute allocation.

Do not call longer-T improvement algorithmic extrapolation if it comes from
training on those exact depths.

## Gate 6 — hosted submission authority

### Pre-Easy checklist

- local source is legal and validated;
- exact local candidate passed its promotion gate;
- evaluator-mirror throughput is adequate;
- source fingerprint is new and compared with submission history;
- prediction is written;
- quota and active-job state are known;
- owner explicitly approves this exact source and tier.

### Easy-to-Medium checklist

- hosted Easy confirms the local direction;
- a second Easy dataset reduces luck risk;
- no mechanism is changed during promotion;
- local Medium-duration mirror supports throughput and fit.

### Medium-to-Hard checklist

- near-zero errors on large seen-N and unseen-N T=1 suites;
- at least 99.9% exact across repeated profiles;
- autonomous T=2/4/8 evidence;
- generic executor evidence if hidden recurrence matters;
- 3,600-second feasibility;
- source-level rule review;
- duplicate audit and owner approval.

Hard certification requires all examples in a rung, so “pretty good average
accuracy” is not a promotion criterion.

## GPU lifecycle exam

Before rental, write expected cost, maximum hours, kill condition, remote path,
log/checkpoint cadence, and backup destination. Verify provider identity and
SSH target before starting. Before termination:

1. stop the training process cleanly;
2. copy source, config, logs, reports, plots, and required checkpoints;
3. exclude environments and caches;
4. verify file count and byte total on both sides;
5. verify the exact provider instance identifier;
6. terminate only that instance;
7. confirm provider reports zero unintended active compute.

Failure to back up means the experiment is not finished. Failure to terminate
means the budget is not controlled.

## Weekly research-director review

Answer:

1. What is the single current bottleneck?
2. Which evidence makes it the bottleneck?
3. What competing explanation remains?
4. Which next experiment changes the decision?
5. What branch is now closed, at exactly what scope?
6. Did any result get promoted beyond its evidence class?
7. Did agent-generated prose become an uncited “finding”?
8. Are experiment folders complete and reproducible?
9. What compute/quota was spent, and why was it worth it?
10. What should we explicitly not do next week?

## Final project-director exam

You receive an unfamiliar agent proposal containing a new architecture, two
loss changes, a new optimizer, and a claim of arithmetic generalization from
one seed. In 30 minutes you must:

1. reconstruct the causal claim;
2. identify all confounds;
3. reduce it to one discriminating card;
4. specify controls, metrics, interventions, and gates;
5. classify legal versus diagnostic components;
6. estimate CPU/GPU/hosted cost;
7. write the agent execution contract;
8. decide run, revise, or reject.

Pass when your decision remains defensible after an adversarial review. At that
point you are managing the project rather than following its agents.

## Topics the original request could easily miss

The learning plan also needs:

- probability, calibration, and uncertainty across seeds;
- data leakage and split construction;
- causal controls and intervention design;
- numerical stability and mixed precision;
- profiling, throughput, memory, and wall-clock accounting;
- reproducibility, version control, source fingerprints, and artifact lineage;
- competition rules and research ethics;
- negative-result scoping;
- budgeting and stopping decisions;
- communication: separating evidence, interpretation, and speculation.

These are not administrative extras. They determine whether a model result is
real, reproducible, legal, and decision-useful.
