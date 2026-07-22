# Agent prompts (handoff)

Use with a **fresh** model/agent that has **not** read this repo’s learnings, hypotheses, or strategy notes unless you paste them.

Packet to attach: [`PRIMARY_SOURCES.md`](PRIMARY_SOURCES.md) only (for Prompt A).  
For Prompt B, attach the **output of Prompt A** (or PRIMARY_SOURCES.md) and nothing else from this sandbox.

---

## Prompt A — Collate primary data only

```
You are compiling a handoff packet about the One Layer Deeper competition
(onelayerdeeper.ai) for solvers who have NEVER seen any prior discussion.
The packet must contain only primary sources and raw facts — zero interpretation,
zero strategy.

INCLUDE, verbatim where possible:
1. Full competition rules text (from the site/README). Include the Hard-tier
   warning about the recurrence and this organizer Q&A, quoted exactly:

   Q: Hard task warning: Hard may change aspects of the recurrence itself; do not
   assume it is repeated squaring. this is mentioned on submission page does this
   mean, instead of X^2 mod N being the single step in serial computation, we
   should assume, it will be some nearby family? like affine transform of that,
   or cube or something else?

   Official response: yeah, some people have tried to guess that slightly new
   family, some approaches have worked, new ones have not lol

   Note from principal: hard recurrence has not been confirmed beyond this.

2. Evaluator/harness source files defining the interface (model construction,
   tokenization, T, output positions, auxiliary, max_steps/batch vs wall-clock,
   whether forward depth may be input-dependent). Paste with paths.

3. Tier table: N bit-ranges, T ranges, fixed-vs-varying N, wall-clock, train/eval
   split semantics. If undocumented, write UNDOCUMENTED — do not guess.

4. 5–10 raw example rows in exact serialized token form per tier type available.

5. Flat results table of every run supplied in attached metrics/logs: run name,
   config if present, metrics. NO commentary column.

6. Submission quota rules and evaluation hardware if documented.

7. Compute available (if stated by the user): 1x L40S 48GB ~$1/hr credits; H100
   via competition submit if needed.

8. One line of leaderboard state: top score, distribution, entrant count.

EXCLUDE: diagnosis, planned experiments, decision trees, paper citations,
hypotheses, which approaches are promising. Do not include CLAUDE.md,
HYPOTHESES.md, or learnings/ strategy notes.

Output: one markdown file, sections numbered 1–8.
```

Optional attach: this repo’s `solving/handoff/PRIMARY_SOURCES.md` as a draft to
verify/extend against upstream GitHub + onelayerdeeper.ai (do not trust learnings/).

---

## Prompt B — Fresh Hard attempt from primary sources only

```
You are a competition solver. You have ONLY the primary-sources handoff packet
(sections 1–8). You have no prior chat history and no research notes.

Task:
1. State assumptions explicitly (label each ASSUMPTION vs FACT from the packet).
2. Design one Hard-tier `submission.py` that fits the Submission contract
   (≤500M state, ≤256 KiB file, exports SUBMISSION).
3. Do NOT implement a closed-form modular squaring/cubing solver, φ(N), dataset
   inspection, or custom training loop (Rules 10–11).
4. Prefer mechanisms that could transfer if the serial step is a “slightly new
   family” near repeated squaring (exact family unknown / unconfirmed).
5. Explain every major design choice in a short ASSUMPTIONS.md next to the file
   (assumptions + which packet facts they rest on). No leaderboard-hacking plan.
6. Deliver:
   - submission.py
   - ASSUMPTIONS.md
   - one-paragraph “how to submit”: one-layer validate; one-layer submit --tier hard
     (principal approval required before you run submit).

Constraints from the packet: Hard is hidden; 3600s train / 1800s eval; 1 attempt
per UTC day; H100; recurrence may differ from Easy/Medium.
```

---

## Prompt C — Cross-check (optional)

```
Diff the attached PRIMARY_SOURCES.md against the live
https://github.com/tilde-research/one-layer-deeper README +
https://onelayerdeeper.ai/problem + competition/service/tiers.py.
List only factual deltas. Do not add strategy.
```
