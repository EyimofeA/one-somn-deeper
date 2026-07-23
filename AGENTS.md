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
| **You (human)** | Hard approval, `one-layer login`, and strategic choices when you want direct control |
| **Parent agent** | Predictions when delegated, result classification, implementation, validation, `learnings/*`, `solving/RESEARCH_LOG.md`, figures, STATUS updates, options lists, experiment cards, and notebooks |
| **Subagents (optional)** | Bounded independent work only when the human asks or parallel delegation is clearly useful |

The parent works directly by default. When the human delegates autonomy, the parent may write `PREDICT`, select the next bounded experiment, and classify its result. Any subagent returns a short findings block (pass/fail, paths, blockers); the parent remains responsible for interpretation and durable notes.

## API credit discipline

Default lean:

- **No routine subagents** — the parent implements and validates directly.
- Use a subagent only when the human asks or a bounded independent task benefits materially from parallel work.
- Subagent briefs: file paths + contract + done-when — no essay prompts.
- Prefer CPU smoke on Mac over GPU subagent runs.
- **Prior-art / “search the web for plans”** — parent runs WebSearch (and writes a short note under `learnings/readings/`). Do not spawn a web-search subagent unless you explicitly ask.

## Git artifacts

Workspace is a git repo. **One experiment = one commit** after `NOTE.md` (see `solving/experiments/LAYOUT.md`). Local smoke uses a separate clone of [tilde-research/one-layer-deeper](https://github.com/tilde-research/one-layer-deeper) (optional; not part of this repo).

## Subagents

- Optional, not the default workflow.
- Use only for a concrete, bounded, independent subtask when requested or clearly beneficial.
- Strategy, predictions, result classification, and durable research synthesis stay with the parent.

## Compute

- **Mac** — CPU smoke, `one-layer validate`, unit tests, μ+λ / digit-count measurements.
- **GPU box (Prime L40S)** — local Easy/Medium manifests, **zero quota**. Connect/run/rebuild: [`solving/experiments/OPS.md`](solving/experiments/OPS.md) § GPU box. **Never** `uv sync` / bare `uv run` on that box (breaks cu126 torch).
- **Competition Easy/Medium** — scored confirmation (~60 Easy / 6 Medium per UTC day).
- **`one-layer submit`** — official H100 accuracy (requires login). Hard = hosted only.

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
