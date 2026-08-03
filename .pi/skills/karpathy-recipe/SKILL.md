---
name: karpathy-recipe
description: Apply Karpathy's "A Recipe for Training Neural Networks" (2019) to ML experiments. Maps his stages — become one with the data, overfit single batch, simple→complex, one change at a time, visualize everything — into actionable steps. Use when debugging training, designing experiments, diagnosing why a model isn't learning, or planning the next experiment.
---

# Karpathy Recipe for Training Neural Networks

Source: [karpathy.github.io/2019/04/25/recipe](https://karpathy.github.io/2019/04/25/recipe/)

## The ladder (apply in order — don't skip)

### 1. Become one with the data

Before touching the model, the agent must:

- Inspect data samples directly (shapes, ranges, edge cases)
- Check for label noise, imbalance, duplication
- Run the competition data generator and look at actual inputs/outputs
- Understand what the model is being asked to predict

**This project:** read `learnings/concepts/01-the-problem.md` and `learnings/concepts/02-where-the-data-is.md`. Look at generated data in `competition/data/`.

### 2. Skeleton + dumb baseline

- Build a minimal end-to-end train/eval loop
- Run a trivial baseline (constant prediction, random, or 1-layer linear)
- Establish the metric floor — what does "no skill" look like?

**This project:** the b0/b1/b2 baselines in `solving/experiments/`. Anchor: `depth_d32_k2_ut` for e1, `depth_d32_k4_ut` for e5.

### 3. Overfit a single batch

- Take one small batch, train to 100% exact match
- If it can't overfit, the architecture or optimizer is broken
- Only then add complexity

**This project constraint:** hosted runner doesn't return weights → approximate with `smoke_cpu` watching train-batch exact rise under 60s.

### 4. Simple → complex, one change at a time

- Start with the simplest model that could plausibly work
- Add one thing: width, depth, recurrence, attention, aux loss
- One experiment = one variable changed
- Ablate: if you add recurrence, also try without to isolate effect

### 5. Visualize everything

- Plot every scalar the runner returns: loss, exact match, perplexity
- Plot T-extrapolation curves (accuracy vs problem size)
- Dashboard from JSONL metrics — no silent complexity

### 6. Only then scale

- Width, params, steps — only after the small version works
- Plot score vs params/steps under fixed clock
- Don't scale a broken mechanism

## Where this project is (state as of learnings)

- Stages 1–4: done. Have baselines, overfit diagnostics, K-sweep, UT depth embeddings
- Current bottleneck: reusable digit-level recurrence (not memorization)
- `solving/STATUS.md` has the live funnel

## When applying this skill

1. Read `solving/STATUS.md` for current question and scoreboard
2. Read `solving/RESEARCH_LOG.md` for what's been tried
3. Classify the problem on the Karpathy ladder above
4. Propose the smallest change that tests one hypothesis
5. Require a prediction before any run (per `RESEARCH_PROTOCOL.md`)

## Non-negotiables

- Hypothesis before submit
- Plot every returned scalar
- Prefer Easy ablations over unverified Medium complexity
- No checkpoint fine-tune (API doesn't return weights)
- Reserve 10 Easy/day for fast iteration