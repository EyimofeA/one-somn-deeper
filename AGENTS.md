# AGENTS.md

Research sandbox for [One Layer Deeper](https://github.com/tilde-research/one-layer-deeper). Learn the task, measure baselines, aim for a Hard submission.

## How you work

1. **You don't say everything upfront.** The agent asks clarifying questions (use `ask_user_question`) when requirements are ambiguous. When it can proceed with reasonable defaults, it states assumptions explicitly and continues.
2. **The agent describes, not just shows code.** Do not assume the human sees files, diffs, or the editor. Describe what changed, why, and what it means. Show the command when a run starts and the endpoint when it finishes.
3. **Teach, don't just do.** Explain the *why* — mechanism, trade-off, failure mode. Notes in `learnings/` should let a future reader understand the problem without re-reading the competition repo.
4. **Proactive suggestions are welcome.** Flag patterns, call out bugs, suggest what's missing. Still present options for decisions (per RESEARCH_PROTOCOL.md §2), but don't be silent about observations.
5. **Default lazy (ponytail full).** Shortest working diff, stdlib first, no unrequested abstractions. Override only when the human asks for the full version.
6. **Skills are self-service.** Any skill can be loaded at any time — just say "loading the X skill" and use it. Don't wait for the user to ask. If a skill would help, offer it.

## Subagent policy

Use a subagent only when a bounded independent review or parallel task is
actually useful. When one is used, the owner requires a **Luna** subagent.
Do not silently substitute another model. If Luna is not selectable in the
current environment, work directly or tell the owner before delegating.

## Read order (strict — link, don't duplicate)

1. **This file** — roles, behavior, tool map, forbidden shortcuts
2. **[`PITFALLS.md`](PITFALLS.md)** — recurring mistakes and how to avoid them
3. **[`RESEARCH_PROTOCOL.md`](RESEARCH_PROTOCOL.md)** — prediction rule, options format, ban list
4. **[`solving/STATUS.md`](solving/STATUS.md)** — scoreboard, current question, next actions
5. **[`HYPOTHESES.md`](HYPOTHESES.md)** — uncited ideas
6. **[`learnings/sessions/`](learnings/sessions/)** — day syntheses (start with latest)
7. **[`learnings/concepts/01-the-problem.md`](learnings/concepts/01-the-problem.md)** — math and scoring
8. **[`learnings/curriculum.md`](learnings/curriculum.md)** — concept index
9. **[`learnings/readings/one-layer-deeper-notes.md`](learnings/readings/one-layer-deeper-notes.md)** — mechanism lecture
10. **[`solving/RESEARCH_LOG.md`](solving/RESEARCH_LOG.md)** — append-only experiment facts
11. **[`solving/experiments/`](solving/experiments/)** — OPS, metrics, card snapshots
12. **[`solving/submissions/`](solving/submissions/)** — active submissions
13. **[`GRIEVANCES.md`](GRIEVANCES.md)** — append-only operational friction log;
    never use it as research evidence

If something belongs in steps 2–12, link it — don't restate.

## Pi tools for research

| Tool | Use for |
|------|---------|
| `web_search` | Prior art, technique discovery → write note in `learnings/readings/` |
| `studio_repl_send` | Quick Python smoke tests, tensor inspection, data exploration |
| `bash` | GPU box commands, `one-layer validate`, training runs |
| `ask_user_question` | Clarify before coding when requirements are ambiguous |
| `fetch_content` | Read papers/URLs, YouTube talks |

## Forbidden (summary — full list: RESEARCH_PROTOCOL.md §6)

- Math oracles (φ(N), closed-form mod exp)
- Hard-coded weights / answer lookup / hard-coded forward algorithms
- Broken autograd or CPU offload of model state
- Auto Hard submit
- Import `sympy`, `gmpy2`, `math.pow` with three args

## Links

- `README.md` — human entry
- `RESEARCH_PROTOCOL.md` — decisions, prediction rule, ban list
- `solving/STATUS.md` — live scoreboard
- `GRIEVANCES.md` — operational friction log, separate from evidence
- Upstream: [tilde-research/one-layer-deeper](https://github.com/tilde-research/one-layer-deeper)
