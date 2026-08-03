---
name: submit
description: Prepare, validate, and submit experiment cards to the One Layer Deeper competition. Runs pre-submit checks (ban list grep, one-variable diff, prediction requirement), validates against competition rules, and executes submission. Use when the user says "submit", "ship this", "send to competition", or is about to run a scored evaluation.
---

# Submit an experiment

## Pre-submit checklist (run before anything else)

### 1. Ban list grep

```bash
grep -nE 'pow\(|%|sympy|gmpy|pow_mod|torch\.load' submission.py
```

If anything matches → **block the submit**. Report the line and the violation. Do not proceed.

### 2. One-variable check

Diff the candidate `submission.py` against the anchor. If more than one hyperparameter block changed → warn, ask for confirmation.

### 3. Prediction requirement

Check `solving/experiments/predictions.md` — does this card have a CARD/CHANGE/PREDICT entry? If not, **block**. Say: *"No prediction on file. Write your three lines first."*

### 4. File size

```bash
wc -c submission.py
```

Must be under 256 KiB. Block if over.

### 5. Model state limit

Confirm parameter count ≤ 500,000,000. If unknown, warn.

## Submission commands

After checks pass:

```bash
one-layer submit <path-to-submission.py> --tier <easy|medium|hard>
```

- **Easy**: ~60/day quota. Use for ablations and fast iteration.
- **Medium**: ~6/day quota. Use only when Easy results confirm the mechanism.
- **Hard**: Hosted-only. **Never auto-submit to Hard.** Always ask for explicit confirmation first.

## After submit

1. Record the result in `solving/RESEARCH_LOG.md`
2. Update `solving/STATUS.md` scoreboard
3. Write the RESULT line in `solving/experiments/predictions.md`
4. Update `solving/experiments/<card>/NOTE.md` with outcome

## Hard submit — explicit gate

Before any Hard submit, say this exact text and use `ask_user_question`:

> "You are about to submit to Hard tier. This costs quota and counts on the leaderboard. The current best is ~0.05% exact match. Are you sure?"

Only proceed if the user confirms. Never auto-submit to Hard.