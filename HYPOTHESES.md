# Hypotheses

Ideas and mechanisms **without** a citing metrics file or figure.  
A claim moves to `learnings/concepts/` only when a run supports it.
See [`RESEARCH_PROTOCOL.md`](RESEARCH_PROTOCOL.md) §5.

Canonical mechanism lecture (not yet evidenced here): [`learnings/readings/one-layer-deeper-notes.md`](learnings/readings/one-layer-deeper-notes.md).

Independent theorem-first model roundtable (model opinion plus primary-agent critique, not experimental evidence): [`review_packets/model_theory_2026-08-09/README.md`](review_packets/model_theory_2026-08-09/README.md).

**Forward design thesis (read before the next architecture pass):** [`solving/DESIGN_NEXT.md`](solving/DESIGN_NEXT.md) — form-vs-content strip-away plan, the corrected composition-wall reading (structural cliff, not error-compounding), the cubing prediction, and the learned-reduction-cell next pass.

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
| H9 | Digit/token embeddings with a Fourier (cyclic-group) structural prior — either fixed sin/cos basis weights, or a learned embedding matrix initialized/regularized toward one — generalize modular arithmetic better than a plain learned embedding table. Loosenable along a spectrum: fully fixed Fourier basis (strongest prior, no learned embedding at all) → learned weights with a Fourier-shaped inductive bias (e.g. parameterized as amplitude/phase over a fixed frequency set, sin/cos activations) → fully free learned matrix (current baseline, no prior). Motivation: the discrete-log/circular representations found in mechanistic-interp work on modular addition/multiplication (Nanda et al. 2023, cited earlier this project) are *learned*, not designed in — this asks whether building that structure in directly, at varying strengths, helps or is redundant with what SGD already finds. | one-change card: swap token embedding only, same architecture/data otherwise, compare against the plain-embedding anchor on a T=1 held-out-modulus probe first (cheapest gate); if it clears, test whether it also helps degree-of-freedom-limited settings (small D_MODEL) where a free embedding table can't afford to spend capacity discovering the structure itself |
| H10 | The apparent fixed-N quotient frontier is primarily unsupported raw decimal-state magnitude, not a recurrent-length limit: with s=qN+r, increasing q also moves the leading decimal digits outside training support. | After the pre-registered q<=300 curve, compare teacher-forced low/high-r rows at matched q and matched-magnitude rows across seen N values with different effective q. Defer a raw-decimal versus N-relative state encoding test to the unseen-N q=1 branch. |
| H11 | The final-label VDF failure is caused partly by compressing arithmetic-relevant per-position state into pooled/output-aligned registers. An LSD-aligned per-position latent state should retain control information that a pooled latent loses. | Matched small-N VDF comparison: pooled global latent vs structured per-position latent vs prompt/register state; then, within the structured state, final-label-only vs diagnostic trace supervision. Promote only for a held-out-N advantage at both T=1 and T=2; otherwise move to binary/limb representation. |

## Ruled out as product direction (still listed so we do not re-open)

| ID | Claim | Why parked |
|----|-------|------------|
| X1 | Algebraic closed-form squaring solver | Hard changes recurrence; ban list |
| X2 | More width (d→2048) fixes transfer | `claude_hard_h1`: train 100% / eval 0% |
| X3 | A tied learned decimal transition is intrinsically unstable beyond a few self-fed steps | Refuted by the full-trace card: with q=0..100 state support, free terminal exact is 99.22% q=10, 97.27% q=50, and 95.70% q=100; teacher-forced q=100 is 99.61%. |
