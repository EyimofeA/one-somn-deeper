# Experiment layout

**One experiment = one directory = one commit** (after `NOTE.md` is written).

Full decision protocol: [`../../RESEARCH_PROTOCOL.md`](../../RESEARCH_PROTOCOL.md).

## Target tree

```
solving/experiments/
├── OPS.md
├── LAYOUT.md              # this file
├── predictions.md         # append-only CARD / CHANGE / PREDICT / RESULT
├── <YYYY-MM-DD>_<name>/   # atomic unit — full history including failures
│   ├── submission.py      # frozen card copy
│   ├── config.json        # the one changed variable
│   ├── NOTE.md            # required — 3–5 lines what/why/result
│   ├── curve.png          # from scripts/extrapolation_curve.py only
│   └── metrics.jsonl      # gitignored if large
└── archive/               # superseded ablations moved here later
```

`solving/submissions/` = **symlinks to active dated experiment dirs** only.  
All cards (including failures) live as `solving/experiments/2026-07-21_<name>/` (and later dates).

## Commit vs ignore

| Commit | Do not commit |
|--------|----------------|
| `submission.py`, `config.json`, `NOTE.md` | `metrics.jsonl`, `*.pt`, checkpoints |
| `predictions.md` entry | `data/generated/` |
| `STATUS.md` update (same commit) | |

Commit message:

```
exp: <name> — <one-line result>

CHANGE: <the one variable>
RESULT: confirmed/refuted/unclear — <why>
```

## Governing rule

**An experiment folder without `NOTE.md` does not exist yet.** Running code is not the experiment — the written interpretation is.
