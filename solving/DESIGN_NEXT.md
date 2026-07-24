# Design note — form vs. content, and what to strip next

Forward-looking design, written 2026-07-24 after the capacity/composition session
(`experiments/2026-07-23_capacity_composition/`). Not a logged result — a plan for
the next passes. Companion to [`HYPOTHESES.md`](../HYPOTHESES.md) and the mechanism
lecture [`learnings/readings/one-layer-deeper-notes.md`](../learnings/readings/one-layer-deeper-notes.md).

## The task, stated exactly

`y = x^(2^T) mod N` — square x, then square the result, T times, then reduce mod N.
Model sees the digits of N, x, T. Scored on exact digit match. Hard tier: (1) N is
**held out** (cross-modulus generalization, memorization = 0), and (2) the recurrence
**may not be squaring** (rules S1.2-1.3 — could be x²+c, x³, …; must be inferred from
the training split, never assumed).

## Correction to the record: the composition wall is NOT error-compounding

Earlier framing said "a 98% one-step, cubed, gives ~50%." That is **wrong** —
0.98³ = 0.94, not 0.5. The observed 50.0% held-T (flat across d=64/128/256,
`comp_d128`/`comp_d256`) is a **structural cliff, not probabilistic decay**: a clean
subset of bases (≈5/10) pass and the rest fail categorically. Reading: the per-step
cell did not learn "squaring" — it learned a **look-alike** that coincides with
squaring only on the training value range; T=3 pushes ~half the bases off that
support and they fail wholesale. This is *more* damning than compounding: the step
function itself is wrong, just wrong invisibly on the training support. Iteration
machinery (UT, ACT, quantized state) faithfully reproduces a wrong step — the problem
is the **step**, not the **applying**.

## Form vs. content (the organizing idea)

The general structure is: **a shared function `g`, applied T times, with a discrete
state passed between applications.** Split it:

**KEEP — load-bearing, cannot be stripped:**
- **Weight-tied recurrence** (`g` reused T times). The only reason T can extrapolate
  (train T≤2, run T=8). This is the compositional prior; removing it loses depth
  generalization. (Lineage: Universal Transformer. Tried as UT K4/K8, ACT halting,
  T-proportional loops — all real, none cracked composition because the *step* was
  the look-alike, see above.)
- **Exact / discrete inter-step state.** Drift is what lets the look-alike survive
  training. Force the passed state to be an exact digit-tuple so a wrong step becomes
  *visible* (loss spikes) instead of smearing. Discreteness is a feature here, not a
  crutch. (Related: HYPOTHESES H2 — STE quantize between steps.)

**STRIP — currently hardcoded, all learnable:**
- **The operation.** The multiply cell computes pairwise products (`digit_i·digit_j`)
  = it *hardcodes squaring*. Replace with a general learned per-step cell that
  **discovers** the map from data. This is the big one and the only version robust to
  Hard's recurrence twist.
- **The routing** (the `i+j` schoolbook column bucketing). Let attention learn which
  positions combine. Diagnostic test: does it *rediscover* the diagonal? If yes, the
  structure is learnable, not merely imposable.
- **Digit geometry / truncation.** Learning to select the last k digits (= mod 10^k)
  is trivial positional selection — not load-bearing. NOTE: truncation only ever gives
  mod (a power of 10); it can never give mod 323. Real mod-N needs division.

One-line thesis: **keep "apply the same exact-state step T times"; strip and learn
everything about what that step *is*.**

## Where attention finally earns its place

The multiply cell used GRUs, not attention, because its routing was fixed and known
(hardcoding beats attention on sample-efficiency when routing is known). Attention is
needed exactly where routing is **content-dependent**: **modular reduction mod N**.
How you reduce depends on the actual digits of N (held out, different every example) —
so the model must *look at N's digits* to decide the quotient. Query = running
remainder, keys/values = N's digits. This is the one step the hardcoded cell had to
skip (it truncated to mod 10^4 instead), and it is where a transformer component is
structurally required.

## Cubing prediction (why hardcoded squaring is dangerous, and diagnostic)

If Hard's hidden operation is x³, the current multiply cell is **structurally dead**:
it has only pairwise products, cubing needs triple products (`digit_i·digit_j·digit_k`),
which the architecture cannot represent at any training length. Expected behavior:
fit a look-alike on small-x training support, **floor on held-out N**, and produce the
wrong exponent (`x^(2^T)` vs the true `x^(3^T)`) so T-extrapolation collapses.
**Diagnostic value:** a square-cell that nails Easy/Medium (known squaring) but floors
Hard is *evidence Hard altered the recurrence* — a signal nobody has isolated. A
learned-operation cell would be robust by construction. (Supersedes/【sharpens】
HYPOTHESES X1: not just "closed-form solver banned" but "hardcoded *squaring* is a
liability, not only a rule risk.")

## The next passes, in order

1. **One-step, held-out N** (`x² mod N`, T=1) with a **learned reduction cell**
   (recurrent remainder + attention over N's digits + learned quotient head +
   structured exact subtract; supervise only on the final remainder — legal, no
   intermediate oracle). Run **fixed-N first** to fail fast, then held-out N. This is
   the true gate — nobody in the project or on the leaderboard has cleared held-out-N
   one-step. Prior on the reduction cell training right even with scaffold: ~30-40%.
2. If (1) clears: **attack composition** — perfect the exact discrete inter-step state
   so the step survives iteration (the real fight after one-step works).
3. **Strip the operation:** replace hardcoded pairwise products with a general learned
   step, so the operation is discovered, not assumed. The least-crutch, most-scientific
   design — and the only one that could survive Hard's recurrence twist.

Scaffolding is a crutch to answer "is this reachable by gradient descent." Build it to
get the answer, then spend the rest of the project deleting the scaffold one piece at a
time until a near-plain transformer does it. That progression *is* the research.
