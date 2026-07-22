# Hypotheses

Ideas and mechanisms **without** a citing metrics file or figure.  
A claim moves to `learnings/concepts/` only when a run supports it.
See [`RESEARCH_PROTOCOL.md`](RESEARCH_PROTOCOL.md) §5.

Canonical mechanism lecture (not yet evidenced here): [`learnings/readings/one-layer-deeper-notes.md`](learnings/readings/one-layer-deeper-notes.md).

## Open (Path D plan)

| ID | Hypothesis | Test |
|----|------------|------|
| H1 | Analog recurrence cannot stay exact at large T for any fixed Lipschitz L of B | T-extrapolation curve after fixed-K UT (Part 9 protocol) |
| H2 | Straight-through quantize between steps → exact-match vs T becomes a step, not a decay | one-change card vs UT anchor |
| H3 | Progressive loss (n free + m grad) closes margin on self-generated states | same curve flattens past training T |
| H4 | Re-inject N (and x) every loop reduces off-task drift | small alone; large with H2 |
| H5 | Init scale α ∈ {0.5, 0.1} moves memorize→generalize earlier than ~64k steps | local step-budget probe |
| H6 | Muon (orthogonalized updates) helps keep L near 1 and speeds wall-clock | after H2–H5 signal |
| H7 | Wide N needs two-level (digit-limb) recurrence; small N does not | count digits of N on h1/m5 first |
| H8 | On our N’s, μ+λ (tail+cycle) is small → Path D gets free T-extrapolation | local μ+λ histogram, no GPU |

## Ruled out as product direction (still listed so we do not re-open)

| ID | Claim | Why parked |
|----|-------|------------|
| X1 | Algebraic closed-form squaring solver | Hard changes recurrence; ban list |
| X2 | More width (d→2048) fixes transfer | `claude_hard_h1`: train 100% / eval 0% |
