# Experiment layout

**One experiment = one commit.** `solving/RESEARCH_LOG.md` is the durable
per-experiment record; the commit is the exact code and configuration record.

Full decision protocol: [`../../RESEARCH_PROTOCOL.md`](../../RESEARCH_PROTOCOL.md).

## Target tree

```
solving/experiments/
├── OPS.md
├── LAYOUT.md              # this file
├── predictions.md         # append-only CARD / CHANGE / PREDICT / RESULT
├── <YYYY-MM-DD>_<name>/   # measured artifacts only, when worth retaining
│   ├── curve.png          # from scripts/extrapolation_curve.py only
│   └── metrics.jsonl      # gitignored if large
└── archive/               # superseded ablations moved here later
```

`solving/research/` holds the canonical code for active mechanisms. Git
commits hold the exact source and configuration history. Do not copy a
per-card `NOTE.md`, `config.json`, `manifest.json`, or `submission.py` into
experiment directories. Put the prediction, one-variable change, result, and
interpretation in `solving/RESEARCH_LOG.md`; the associated commit is the
reproducible implementation record.

`solving/submissions/` = **active upload artifacts** only. All cards
(including failures) live as `solving/experiments/2026-07-21_<name>/` (and
later dates).

## Commit vs ignore

| Commit | Do not commit |
|--------|----------------|
| `RESEARCH_LOG.md`, `predictions.md`, `STATUS.md` | `metrics.jsonl`, `*.pt`, checkpoints |
| canonical source/config changes | `data/generated/` |
| compact figures when they support the logged result | |

Commit message:

```
exp: <name> — <one-line result>

CHANGE: <the one variable>
RESULT: confirmed/refuted/unclear — <why>
```

## Governing rule

**An experiment without a `RESEARCH_LOG.md` entry does not exist yet.** Running
code is not the experiment — the written interpretation and its commit are.
