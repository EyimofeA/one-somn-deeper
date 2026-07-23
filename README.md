# One Somn Deeper

Research sandbox for [One Layer Deeper](https://github.com/tilde-research/one-layer-deeper).

## Now

The competition model is still unsolved: best Hard result is **0.05%** exact.
The active bottleneck is narrower: can a learned model execute one reusable
digit-level recurrence rather than memorize examples?

The current diagnostic established that copying passes, held-out digit
products fail, and learned carry normalization is possible with a continuous
recurrent state (79.45% held-out exact match) but damaged by a prototype
bottleneck (soft 38.8%; hard 0.25%).

Read [`solving/STATUS.md`](solving/STATUS.md) for the current state, then
[`solving/RESEARCH_LOG.md`](solving/RESEARCH_LOG.md) for chronological facts.

## Reproduce or catch up

1. [`solving/STATUS.md`](solving/STATUS.md): current question, scoreboard, and next falsifiable test.
2. [`solving/RESEARCH_LOG.md`](solving/RESEARCH_LOG.md): all recorded runs in chronological order.
3. [`solving/research/`](solving/research): minimal canonical code for active diagnostics.
4. [`solving/experiments/OPS.md`](solving/experiments/OPS.md): A6000 and hosted-run commands.
5. `git log --follow -- <path>`: exact prior code or configuration; duplicated card snapshots are intentionally not retained.

## Layout

```
RESEARCH_PROTOCOL.md
HYPOTHESES.md
learnings/          concepts, sessions, readings, papers
solving/STATUS.md
solving/RESEARCH_LOG.md
solving/research/      canonical active mechanisms
solving/experiments/   OPS and non-code measured artifacts only
solving/submissions/   generated active upload artifacts only
scripts/            frozen extrapolation_curve.py
colab/
```

Upstream CLI (clone for local smoke): [tilde-research/one-layer-deeper](https://github.com/tilde-research/one-layer-deeper).
