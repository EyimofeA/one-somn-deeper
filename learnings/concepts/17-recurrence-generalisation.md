# 17 — Exact recurrence learning (the actual problem)

Written 2026-07-21 after the Hard H1 run. This is the forward-looking design note; the
evidence behind it is in `16-representation-vs-throughput.md`, the scored history is in
`solving/RESEARCH_LOG.md`.

## The result that defines the problem

`claude_hard_h1` (d=2048, 50.5M params, K=4), 3600s, 190,017 steps:

    train loss  0.0000        train exact-match  100.00%
    eval  test  0.0000%       eval loss  15.836
    eval  ood_t 0.0000%       eval loss  16.170
    eval  ood_n_t 0.0000%     eval loss  16.387

Perfect memorisation. Zero transfer. Eval loss ~16 against a ln(10)=2.303 uniform
baseline means the model is not merely wrong but **confidently** wrong — it emits
memorised answers.

**Capacity and optimisation are finished problems.** Everything remaining is inductive
bias. Do not reach for more parameters.

## Not a solver

The tempting shortcut is to exploit the algebra. For fixed N = pq, `(Z/N)* ≅ Z_{p-1} ×
Z_{q-1}`; in the character basis squaring is diagonal, so `x^(2^T)` is just "multiply
every phase by 2^T" — closed form, O(1) in T, no loop at all.

**Rejected, deliberately.** It is a solver for this specific algebra. The organisers
state Hard uses a different algorithm from Easy (`14-discord-beta-meta.md` — e.g. cubing
rather than squaring), so a squaring-specific shortcut transfers to nothing, and the
Easy-100%-via-solver route is explicitly the degenerate path. The benchmark is asking for
learned algorithmic depth.

The target is therefore: **learn one step exactly, then apply it more times than you were
trained on.**

## Why iteration currently fails

If a learned map `f̂` approximates `f` with error ε, then `f̂^k` drifts — errors compound
roughly exponentially in k. This is not a bug in any particular card; it is the problem.

It explains the single most confusing result of the session. Fixed-K cards appear to
"generalise" to unseen T only because they **ignore T entirely** — they are not iterating,
so there is nothing to compound. The moment depth was coupled to T, OOD collapsed:

| card | ood (e1) |
|---|---|
| fixed K=4 | 9.00% |
| `claude_pv_tcoupled` (loops = T+1) | 2.00% |
| `claude_pv_tadapt` (loops = T) | 2.00% |

And on m1 adaptive depth also cost throughput: loops=T gave 30,249 steps and final loss
2.135, against 58,060 steps and 2.056 for fixed K=4.

So "make depth track T" is correct in principle and loses in practice, because the
iteration is not exact.

## The proposal: re-quantised recurrence

Digital computers iterate billions of times without drift because state is re-quantised
every step; analog ones drift. Our latent recurrence is analog.

    state z = digit slots (distributions over 0-9), held at the TAIL positions
    repeat T times:
        z ← Block(z, N, x)          # one shared step, input injected every loop
        z ← requantise(z)           # snap toward one-hot over {0..9}
    answer = final z

Four properties, each tied to something measured:

1. **Re-quantisation bounds error.** Snapping each step back onto the valid-digit-string
   manifold means error is *corrected* per iteration rather than accumulated. This is the
   mechanism that makes recurrence exact rather than approximately-right-then-divergent.
   Implement as straight-through argmax, or a low-temperature softmax annealed during
   training.
2. **State space = output space.** `collate_squaring_mod` reads the answer off the last
   `len(answer)` positions (`competition/data/squaring_mod.py:92`), so keeping the working
   value at the tail means "iterate in place, read the tail" is the entire forward pass.
   No separate decoder to get wrong.
3. **Input injection every loop.** Re-feed x and N at each iteration so the state cannot
   drift off-task. Standard Universal-Transformer / deep-equilibrium practice, currently
   absent from every card here.
4. **A memoriser cannot exploit it.** A discrete bottleneck between steps blocks the
   lookup-table solution that the Hard run found, while leaving the algorithmic solution
   reachable.

## The supervision nobody has used

**T=1 rows supervise the single-step map directly and cleanly.** T=2 and T=3 supervise its
composition. That is depth supervision handed to us in the dataset, and every card so far
has treated T as a nuisance parameter to be embedded rather than as the training signal
for a reusable step.

## The legal aux loss

`forward` returns `(logits, auxiliary)` and `auxiliary` is passed straight to
`training_loss` (`competition/benchmark/runner.py`), so **label-free** regularisers are
reachable today and remain untried. The one that matters here: **entropy of the
intermediate digit distributions.** Penalising it trains "stay discrete between steps"
as an explicit objective rather than leaving it to architecture alone.

(Label-*aware* intermediate supervision stays structurally blocked — neither `forward` nor
`training_loss` receives `target_positions`, so a per-position intermediate target cannot
be addressed. See `11-ideas-backlog.md`.)

## Order of attack

1. **Weight decay 0.1 → 1.0 → 3.0.** One constant. In the grokking literature decoupled
   weight decay is the main thing that moves a network off the memorising solution onto
   the generalising one, and the Hard run landed squarely on memorisation at wd=0.1.
   Cheapest experiment available and it needs no new architecture.
2. **Re-quantised recurrence** as above.
3. **Input injection** each loop.
4. **Entropy aux** via `auxiliary`.
5. **Exploit T=1** as single-step supervision.

**Do not** add capacity. **Do not** build a solver.

## The measurement to run

Train on T ∈ {1,2,3}; evaluate at T = 4, 5, 8, 16; plot exact-match against T. A model
that learned one step degrades gracefully; one that memorised depth falls off a cliff at
the first unseen T. **That curve is the real scoreboard for this project**, and nothing
this session measured it — e1's ood split only tests T=4 and its answer space collapses
to 19 values anyway (`16-representation-vs-throughput.md`).

## Run it locally

`competition/` generates any dataset and runs the real training loop offline with zero
quota. The Hard run's grokking transition begins at **~64,000 steps** — invisible inside
Medium's 600s clock (best Medium run: 58,060 steps) but cheap to find overnight on a
local GPU. Most of what this session spent quota on was answerable for free.

**Iterate locally; spend quota only to confirm.**
