# AGENTS.md

## Goal

Research sandbox for [One Layer Deeper](https://github.com/tilde-research/one-layer-deeper). Learn the task, measure baselines, aim for a decent Hard submission.

## Read order (strict — do not duplicate content elsewhere)

1. **This file** — roles, compute, forbidden shortcuts only.
2. **`RESEARCH_PROTOCOL.md`** — decisions, predictions, options format, ban list.
3. **`solving/STATUS.md`** — scoreboard and next actions.
4. **`HYPOTHESES.md`** — uncited ideas (separate from learnings).
5. **`learnings/sessions/`** — day syntheses (start with latest).
6. **`learnings/concepts/01-the-problem.md`** — math and scoring (Unicode, no LaTeX).
7. **`learnings/curriculum.md`** — concept index.
8. **`learnings/readings/one-layer-deeper-notes.md`** — mechanism lecture (Paths A–G).
9. **`solving/RESEARCH_LOG.md`** — append-only experiment facts.
10. **`solving/experiments/`** — `LAYOUT.md`, `predictions.md`, `OPS.md`, metrics, figures.
11. **`solving/submissions/`** — active cards only (`README.md`).

If something belongs in steps 2–11, link it — do not restate it in chat, plans, or rules.

## Who writes what

| Owner | Writes |
|-------|--------|
| **You (human)** | Predictions, which option to run, what a result means, Hard approval, `one-layer login` |
| **Parent agent (planner)** | `learnings/*`, `solving/RESEARCH_LOG.md`, figures, STATUS updates, options lists |
| **Subagents (Composer 2.5)** | `solving/submissions/**/*.py`, experiment dir cards, `colab/*.ipynb`, smoke only |

Subagents return a short findings block (pass/fail, paths, blockers). Parent turns that into research-log entries and teaching notes. Parent never writes PREDICT.

## API credit discipline

No spare Cursor API budget. Default lean:

- **One subagent per coding task** — no parallel spawns unless you ask.
- Parent does not spawn subagents for docs, logs, or interpretation.
- Subagent briefs: file paths + contract + done-when — no essay prompts.
- Prefer CPU smoke on Mac over GPU subagent runs.
- **Prior-art / “search the web for plans”** — parent runs WebSearch (and writes a short note under `learnings/readings/`). Do not spawn a web-search subagent unless you explicitly ask.

## Git artifacts

Workspace is a git repo. **One experiment = one commit** after `NOTE.md` (see `solving/experiments/LAYOUT.md`). Local smoke uses a separate clone of [tilde-research/one-layer-deeper](https://github.com/tilde-research/one-layer-deeper) (optional; not part of this repo).

## Subagents

- Default model: **Composer 2.5** (you can name overrides per task).
- Scope: implementation and validation only — not strategy, not predictions, not learnings.

## Compute

- **Mac** — CPU smoke, `one-layer validate`, unit tests, μ+λ / digit-count measurements.
- **Competition Easy/Medium** — scored confirmation (~60 Easy / 6 Medium per UTC day).
- **Private GPU (Prime / Colab)** — local probes; cheapest ≥16GB CUDA for d≈32 cards. Not required for official scores.
- **`one-layer submit`** — official H100 accuracy (requires login).

Multiple applications of a shared block **inside** one `model.forward` are allowed (recurrence). The evaluator still calls forward once per train step.

## Living rules

Patch `.cursor/rules/*.mdc` when we learn something durable. One concern per file.

## Forbidden

- Math oracles (φ(N), closed-form mod exp in forward pass)
- Hard-coded weights / answer lookup
- Auto Hard submit
- Full ban list: `RESEARCH_PROTOCOL.md` §6

## Links

- `README.md` — human entry
- `RESEARCH_PROTOCOL.md` — decisions
- `colab/sync.md` — Mac → Colab
- Upstream: [tilde-research/one-layer-deeper](https://github.com/tilde-research/one-layer-deeper)
- `scripts/extrapolation_curve.py` — frozen T-curve (once implemented)
