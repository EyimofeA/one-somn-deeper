# AGENTS.md

## Goal

Research sandbox for [One Layer Deeper](https://github.com/tilde-research/one-layer-deeper). Learn the task, measure baselines, aim for a decent Hard submission.

## Read order (strict — do not duplicate content elsewhere)

1. **This file** — roles, compute, forbidden shortcuts only.
2. **`learnings/concepts/01-the-problem.md`** — math and scoring (Unicode, no LaTeX).
3. **`learnings/concepts/02-where-the-data-is.md`** — where JSONL lives; local samples.
4. **`learnings/concepts/03-cheating-boundary.md`** — allowed vs ban.
5. **`learnings/concepts/04-arch-bias-optimizer.md`** — baselines, loops, train loop.
6. **`learnings/curriculum.md`** — what to read next.
7. **`solving/RESEARCH_LOG.md`** — experiment facts (append-only).
8. **`solving/experiments/figures/`** — plots (script + PNG per figure).
9. **`solving/submissions/`** — code only.

If something belongs in steps 2–8, link it — do not restate it in chat, plans, or rules.

## Who writes what

| Owner | Writes |
|-------|--------|
| **Parent agent (planner)** | `learnings/*`, `solving/RESEARCH_LOG.md`, `solving/experiments/figures/*`, result interpretation, plan updates |
| **Subagents (Composer 2.5)** | `solving/submissions/**/*.py`, `colab/*.ipynb`, smoke only |
| **You** | Priorities, model overrides, Hard approval, `one-layer login` |

Subagents return a short findings block (pass/fail, paths, blockers). Parent turns that into research-log entries and teaching notes.

## API credit discipline

No spare Cursor API budget. Default lean:

- **One subagent per coding task** — no parallel spawns unless you ask.
- Parent does not spawn subagents for docs, logs, or interpretation.
- Subagent briefs: file paths + contract + done-when — no essay prompts.
- Prefer CPU smoke on Mac over GPU subagent runs.
- **Prior-art / “search the web for plans”** — parent runs WebSearch (and writes a short note under `learnings/readings/`). Do not spawn a web-search subagent unless you explicitly ask.

## Git artifacts

Workspace is a git repo. After each completed coding deliverable (`submission.py`, Colab notebook, figure script), **commit** (when you ask, or when a Phase checkpoint is done). No orphan untracked code. `competition/` stays a nested clone — do not rewrite its history; ignore or submodule later if needed.

## Subagents

- Default model: **Composer 2.5** (you can name overrides per task).
- Scope: implementation and validation only — not strategy, not learnings.

## Compute

- **Mac** — CPU smoke, `one-layer validate`, unit tests.
- **Competition Easy/Medium** — primary experiment budget (~60 Easy / 6 Medium accepted attempts per UTC day). Prefer this over private GPU when iterating on scored accuracy.
- **Colab Student Pro** — optional; only when we need long private ablations without burning quotas.
- **`one-layer submit`** — official H100 accuracy (requires login).

Multiple applications of a shared block **inside** one `model.forward` are allowed (recurrence). The evaluator still calls forward once per train step.

## Living rules

Patch `.cursor/rules/*.mdc` when we learn something durable. One concern per file.

## Forbidden

- Math oracles (φ(N), closed-form mod exp in forward pass)
- Hard-coded weights / answer lookup
- Auto Hard submit

## Links

- `README.md` — human entry
- `colab/sync.md` — Mac → Colab
- `competition/` — upstream clone
- `solving/experiments/figures/` — research plots (see curriculum Phase 1b)
