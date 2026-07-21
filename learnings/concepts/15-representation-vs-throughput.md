# Representation vs throughput (Easy, 2026-07-21)

Session by Claude Code. Twelve scored Easy runs against `depth_d32_k4_ut` as anchor.
Cards live in `solving/submissions/claude_*`. Every claim below is a scored number,
not a smoke result.

## The one thing that worked: place value

`claude_pv_k4_ut` — add two embeddings derived from `input_ids` alone:

- **field** — is this position in the N span, X span, or T span?
- **place** — 1 + distance from the least-significant digit *of its own span*, so the
  units digit of every operand shares `place == 1` whatever the operand's length.

Single change vs the anchor; d=32, 4 heads, shared block ×4, same optimizer/schedule/batch.
Added state ~2.2K elements (~0.0004% of the 500M ceiling).

| dataset | anchor test/ood | C1 test/ood |
|---|---|---|
| e1 | 1.30 / 8.00 | **3.30 / 9.00** |
| e3 | 0.50 / 0.80 | **1.30 / 1.40** |
| e5 | 0.80 / 0.70 | 0.80 / 0.80 |

Up on both splits on all three datasets. That consistency — not the mean — is the
evidence. **But see the noise section: the e3 gap did not survive replication.**

Why it should help: the task is square-and-reduce mod N on digits. With only absolute
position the block must rediscover "index 7 is the units digit of X" from scratch under
a 60s clock, and that mapping *moves* whenever operand lengths move.

## Four things that did not work

All are single changes on top of C1, all scored, all worse.

| card | change | e1 test/ood | vs C1 6.20 |
|---|---|---|---|
| `claude_pv_ansplace` | + output place value (distance from sequence end) | 2.00 / 2.00 | **2.00** |
| `claude_pv_d128` | width 32 → 128, heads 4 → 8 | 2.00 / 2.00 | **2.00** |
| `claude_pv_tcoupled` | loop count = T+1 instead of fixed 4 | 2.00 / 2.00 | **2.00** |
| `claude_pv_noabspos` | remove absolute position entirely | 0.70 / 7.00 | **3.83** |

Read these as four separate lessons:

1. **Width is not the constraint — the clock is.** d=128 (210K params) ran 394 steps vs
   d=32's 377 and scored a third as well. There are four orders of magnitude of unused
   parameter budget and it is not the bottleneck. Stop reaching for it.
2. **(field, place) is a complete positional code but not a sufficient one.** Every role
   is uniquely identified by (field, place) — provably, no collisions — yet deleting
   absolute position dropped test 3.30 → 0.70. Attention needs a smooth metric for
   relative offsets that a categorical embedding does not provide. Place is *additive*
   value, not a replacement.
3. **T-coupled recurrence failed at the thing it was designed for.** Loop count = T+1
   made OOD *worse* (9.00 → 2.00). Plausible reason: a fixed-K model applies the same 4
   loops to an unseen T=4 row as to training, so it is never off-distribution; coupling
   the count puts OOD rows at a loop depth never trained. Adaptive depth needs the depth
   itself to be trained across a range, not merely indexed.
4. **The residual stream is crowded at d=32.** ansplace added a *correct* signal
   (verified: 60/60 rows reconstruct the answer from (digit, answer_place)) and still
   lost. Five embeddings summed into 32 dimensions is past capacity.

## The noise band — read this before trusting any single run

`claude_pv_fast` recomputes (field, place) in ~8 kernels instead of ~15 with **provably
bit-identical output** (verified equal on train/test/ood at three padding widths). Only
step count differs.

- C1 on e3: **1.31%** (1505 steps)
- fast on e3: **0.75%** (2065 steps)

Same semantics. More steps. Nearly half the score. Two readings, both important:

- run-to-run variance on e3 is at least **±0.5pp**, which swallows most e3 "findings"
  including C1's apparent 2.1× win over the anchor; and/or
- **more steps actively hurt** on e3 — the extra throughput bought overfitting.

Either way: **a single e3 run cannot distinguish a 0.6% card from a 1.3% card.** The
manifest pins `seed: 74`, so replicates vary only through GPU nondeterminism and
timing-dependent step count — which is exactly the variance measured above. Replication
is therefore still meaningful, and necessary.

Practical rule: on Easy, treat any gap under ~1pp as a tie until replicated. Prefer e1,
where the effects are several times larger than the band.

## Reusable technique: deriving structure from `input_ids`

Legal (no dataset access, no trapdoor) — markers N/X/T are public token ids 2/3/4,
digits are 7..16, PAD is 0.

- `field = ((ids >= 2) & (ids <= 4)).cumsum(1).clamp(0, 3)` — markers appear in order
  N < X < T, so one cumsum yields 1/2/3 (the sum of three cumsums is the cumsum of the sum).
- `place` — spans are contiguous, so a digit's span ends one position before the next
  *boundary* (next marker, or the first pad). One reverse-cummin gives every position its
  next boundary; place is then arithmetic. No Python loop, no per-field masked reduction.

See `claude_pv_fast/submission.py`. Bit-exact against the naive form, ~2× faster.

## Where the answer actually lives

`collate_squaring_mod` (`competition/data/squaring_mod.py:92`):

    target_positions[row, :target_len] = arange(input_len - target_len, input_len)

The answer is read off the **last len(answer) positions of the input** — positions that
physically hold T-span tokens. So the final real position is always the answer's units
digit, and distance-from-end *is* the output place value.

Note this format (`separate_input_output=True`, the `bidirectional` data roots) puts **no
answer in `input_ids`** — the sequence is just `[N, digits, X, digits, T, digits]`. The
other tokenizer path in the same file appends `ANS` + answer digits into `input_ids`;
that path is not what these manifests use. Do not derive fields by clamping the marker
cumsum at 3 without checking which format you are on — under the causal path that silently
folds the answer region into the T span.

Exploiting distance-from-end directly (`ansplace`) *lost*, so knowing where the answer
lives has not yet been converted into a win. It stays on the backlog for a wider model
where the residual stream can afford it.

## What to try next

- Replicate C1 vs anchor on e1 ~5× each and settle whether the win is real. Nothing else
  is worth doing until the band is known.
- e5 is at floor (0.29–1.00%) for every card ever run. It is not a discriminator yet; stop
  ranking on it.
- The untested axis remains **N-conditioning done digit-locally** (cross-attention from
  each X digit to N's digits, place-aligned) rather than as a pooled FiLM vector. FiLM was
  rejected on e1/e5 but that tested the wrong form of the idea.
