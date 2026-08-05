# 01 — The Problem: Repeated Modular Squaring

## Task definition

Given a composite modulus **N**, a base **x**, and a time step count **T**, compute repeated modular squaring:

- **x₀** = x mod N
- **xₜ** = (xₜ₋₁)² mod N  for t = 1, 2, …, T
- **y** = x^(2^T) mod N  (equivalently, y = xₜ after T squarings)

The model receives a tokenized prompt encoding (N, x, T) and must predict the exact digit tokens of y.

### Prompt format

Tokens: `N` + digits(N) + `X` + digits(x) + `T` + digits(T) → predict digits(y).

Special tokens: PAD, BOS, N, X, T, ANS, EOS; digits 0–9 map to token IDs 7–16.

## Worked example: N = 77, x = 2, T = 4

N = 7 × 11 (composite; factors are private to the evaluator).

| Step | Computation | Result |
|------|-------------|--------|
| x₀ | 2 mod 77 | 2 |
| x₁ | 2² mod 77 | 4 |
| x₂ | 4² mod 77 | 16 |
| x₃ | 16² mod 77 = 256 mod 77 | 25 |
| x₄ | 25² mod 77 = 625 mod 77 | **9** |

**Answer: y = 9** (i.e. 2^(2⁴) mod 77 = 2¹⁶ mod 77 = 9).

Check via Euler totient (evaluator-only shortcut, forbidden for us):

φ(N) = (7 − 1)(11 − 1) = 60 → 2¹⁶ mod 77 = 2^(16 mod 60) mod 77 = 9 ✓

## Data splits

### In-distribution (fixed training data)

Examples are drawn from fixed sets of (N, x, T) combinations seen during training:

- **train** — optimization split (~80% of in-distribution examples per setting)
- **test** — held-out in-distribution evaluation (~20%)

Training T values and modulus/base pairings are fixed per dataset manifest. The model must generalize across seen T and modulus regimes.

### Out-of-distribution (OOD) evaluation

OOD splits use **T values not seen during training** (and sometimes unseen x or modulus groupings):

- **ood** — merged OOD split (practice tiers average test + OOD equally)
- Separate OOD variants may hold out unseen T with seen x, or unseen x with unseen T

OOD measures whether the model learned compositional squaring or merely memorized training tables.

## Scoring

### Easy / Medium (practice)

The evaluator scores **exact token accuracy** per example:

- Every output digit token must match the label exactly.
- An example is correct only if **all** answer tokens are right (row-wise all-or-nothing).
- **Score = mean exact accuracy** across the fixed evaluation splits and seeds.
- The same run also reports diagnostic **Max T** and **OOD N Max T** on the
  ladder T ∈ {1, 2, 4, 8, 16, 32, 64}; those do **not** change the Easy/Medium score.

### Hard (leaderboard)

Hard no longer ranks by mean exact accuracy (upstream `79f0a09`, 2026-07-24;
tie-break refined `8a3c78d`, 2026-08-03).

- Ladder rungs: T = 1, 2, 4, 8, 16, 32, 64.
- A rung is **solved** only if **every** example at that T is exactly correct.
- **Certified Max T** = largest T such that that rung **and every lower rung** are solved (consecutive prefix).
- **Max T** = certified depth on fresh prompts whose modulus identities appeared in training.
- **OOD N Max T** = same certification on unseen modulus identities (nearby bit lengths).
- Leaderboard order: Max T ↓, then OOD N Max T ↓, then exact accuracy at each profile's first uncertified rung ↓, then earlier submission time.
- Exact-accuracy split metrics remain private diagnostics and do **not** set Hard rank (except that next-rung accuracy is the published tie-break).

Partial credit does not exist on either track — one wrong digit fails the whole example (and breaks certification for that rung).

## Evaluator φ(N) shortcut (forbidden for us)

The evaluator knows the prime factors p, q of N = pq. It computes labels via:

- φ(N) = (p − 1)(q − 1)
- e = 2^T mod φ(N)
- y = x^e mod N

This is fast and exact but requires the trapdoor. **We must not use φ(N), factorization, or any closed-form modular exponentiation** in submissions, training data generation, or research code. The model must learn to compose squarings from examples alone.

## What we control vs what the evaluator controls

| We control | Evaluator controls |
|------------|-------------------|
| Model architecture and depth | Data generation and splits |
| Optimizer and LR schedule | Training loop (one forward, one backward) |
| Optional custom loss (`training_loss` or `token_training_loss`) | Gradient clipping, seeds, deadlines |
| Batch sizes (within ceilings) | Final evaluation and aggregation |

## Competition tiers

- **Easy / Medium** — public datasets, practice; bidirectional attention over prompt; score = mean exact accuracy (Max T diagnostics also reported).
- **Hard** — private hidden evaluator; public leaderboard ranks by certified Max T → OOD N Max T → next-rung accuracy (not mean exact %).
- **Deadline:** submissions close **August 31, 2026 at 10:00 PM PT** (`competition/service/competition.py`).
