# RESEARCH_PROTOCOL.md

Put this file in the repository root. Reference it from `CLAUDE.md`, `AGENTS.md`
and `.cursor/rules`. Written in Simplified Technical English.

---

## 0. Purpose

This project has one bottleneck. The bottleneck is **decisions**, not code.

An agent writes code faster than the human. An agent decides worse than the
human. This protocol moves code to the agent and keeps decisions with the human.

---

## 1. The prediction rule

**No run starts without a written prediction.**

Before any training run, the human writes three lines in
`solving/experiments/predictions.md`.

```
CARD:       <name>
CHANGE:     <the one variable that changed>
PREDICT:    <what the T-extrapolation curve will do, and why>
```

After the run, append one line.

```
RESULT:     confirmed | refuted | unclear   <one sentence>
```

Two reasons for this rule.

1. A prediction you cannot write is a mechanism you do not understand. The run
   will teach you little.
2. A prediction fixes the interpretation before the numbers arrive. Without it,
   any result can be explained after the fact, and the agent will explain it.

**The agent never writes the prediction. The human writes the prediction.**

---

## 2. The options rule

The agent does not choose. The agent presents.

When the agent is asked what to do next, it returns this format and stops.

```
OPTION 1: <change>   COST: <GPU minutes>   PREDICTS: <outcome>   RISK: <what breaks>
OPTION 2: ...
OPTION 3: ...
RECOMMEND: <one line, with the reason>
```

Maximum three options. The human picks. The agent then implements the pick and
nothing else.

The agent is forbidden to implement an option that the human did not pick.

---

## 3. The one-variable rule

One card changes one thing.

A pre-submit script diffs the candidate `submission.py` against the anchor. It
refuses the run if more than one hyperparameter block changed.

Agents improve three things at once. Then the result teaches nothing.

---

## 4. Length contract

Agents are too short or too long, and often both in the same reply. Fix it with
a contract per output type.

| Output type | Length | Format |
| --- | --- | --- |
| Answer to a factual question | 1 to 3 sentences | Prose. No preamble. |
| Explanation of a mechanism | Under 200 words | Prose plus one equation or one diagram. |
| Options for a decision | Section 2 format | No prose outside the format. |
| Result report after a run | A table plus 3 sentences | Numbers first. Interpretation last, and marked as interpretation. |
| Code change | The diff | No summary of the diff. The diff is the summary. |
| Research note in `learnings/` | Under 800 words | Every claim cites a metrics file or a figure path. |

Add one global rule.

> Do not restate the question. Do not summarize what you are about to say. Do
> not summarize what you have said.

---

## 5. Evidence and hypotheses are separate directories

Two directories. Never mixed.

```
learnings/     Every claim cites metrics.jsonl or a figure path.
HYPOTHESES.md  Everything else. Ideas, mechanisms, guesses.
```

A claim moves from `HYPOTHESES.md` to `learnings/` only when a run supports it.

This is the main defence against agent-written research. Agents produce
mechanism stories that read like findings. The story is often correct. It is
still not evidence.

Audit rule: once a week, open a random `learnings/` file and check one citation.

---

## 6. The ban list

Copy rule 10 of the competition into `CLAUDE.md` word for word. Add this.

**Forbidden in `submission.py`:**

- Any closed-form solver for the recurrence.
- Any use of the modulus operator on the task values.
- Any import of `sympy`, `gmpy2`, `math.pow` with three arguments.
- Any inspection of the dataset.
- Any custom training loop.
- Model state above 500,000,000 elements.

Add a pre-submit grep for `pow(`, `%`, `sympy`, `gmpy`, `pow_mod`.

An agent writes a solver when it is stuck. It will look innocent. It will arrive
as a "reference implementation for testing" and then drift into the submission.

---

## 7. Division of work

**Give to the agent.**

- The local harness and the dataset generator.
- Sweep drivers and job scripts.
- The plotting script for the T-extrapolation curve.
- Refactors inside the sandbox.
- "Read this paper. Produce a runnable implementation of the one technique in
  section N." This is the highest-value agent task in the project.

**Keep for the human.**

- Which hyperparameter to change next.
- What a result means.
- Whether a card worked.
- The prediction.
- Every line of `submission.py` that is not boilerplate.

`submission.py` is limited to 256 KiB. Keep the whole file in your own head.

---

## 8. Parallel work

Use one git worktree per card. Use one agent session per worktree.

```
git worktree add ../wt-progressive-loss card/progressive-loss
git worktree add ../wt-quantized        card/quantized
```

The limit is the GPU, not the agent. Do not run more worktrees than you have
GPU hours.

---

## 9. Freeze the measurement

Write the T-extrapolation plotting script once. Freeze it. Mark it read-only.

```
scripts/extrapolation_curve.py   # DO NOT REGENERATE
```

Every card produces `experiments/<card>/curve.png` from this script and no
other.

An agent that regenerates the measurement can produce any conclusion. A frozen
measurement is the only thing in the repository that an agent cannot argue with.

---

## 10. Session open and close

**Open.** The human answers two questions in writing.

1. What shipped since the last session?
2. What are the three actions for today?

Maximum three. Not four.

**Close.** The human writes one line per action: done, or the reason it is not
done.

Known pattern in this project: planning and reading replace shipping. Tangents
open and do not close. The agent must name this pattern when it appears, and
must refuse to open a fourth action.

---

## 11. Experiment folders and commits

Filesystem detail: `solving/experiments/LAYOUT.md`.

**One experiment = one directory = one commit**, after `NOTE.md` is written.

Commit: `submission.py`, `config.json`, `NOTE.md`, the `predictions.md` entry,
and the matching `STATUS.md` update in the **same** commit.

Do not commit: `metrics.jsonl`, checkpoints, `data/generated/`.

An experiment directory without `NOTE.md` does not exist yet.
