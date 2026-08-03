# 19 — Scientific gates

**Author:** Codex

## Purpose

Stop changing end-to-end architectures while an upstream prerequisite is broken.
The task has one dependency chain:

```text
parse digits
  → compute the local arithmetic primitives
  → compute one complete step for one N
  → compute one step across N
  → stay correct on self-generated states
  → compose across held-out T
  → fit inside the wall-clock budget
```

Later gates do not compensate for an earlier failure.

## Current location

The frontier is **before complete one-step generalization**.

- `The` d=32 fixed-N=1073 T=1 anchor reached 100% train EM but only 3.47% peak
  held-out-x EM in 1,800 seconds
  (`solving/experiments/metrics/rung1_n1073_1800s_monitor.jsonl`).
- The digit micro-scan reached 100% train EM but only 1.49% peak held-out-x EM
  (`solving/experiments/2026-07-23_pathd_digit_microscan/metrics/n1073_monitor.jsonl`).
- Its hosted e5 test and held-out-T OOD scores were both 0.50%
  (`solving/experiments/2026-07-23_pathd_digit_microscan/metrics/e5_c22bc015_metrics.jsonl`).

These measurements do not distinguish multiplication failure from reduction
failure. That is the next scientific question.

## Gate 0 — Measurement integrity

Use separate-input/output data. Split by the variable named in the question.
Always report exact match, per-digit accuracy, the marginal baseline, train EM,
steps, and wall time.

**Pass:** answer leakage is absent; the split and baseline are verified.

**Observed:** copy-X reached 100% train, held-out-x, and four-digit OOD exact
match by step 500
(`solving/experiments/2026-07-23_gate0_copy/metrics/monitor.jsonl`).

## Gate 1 — Decimal multiplication

Local diagnostic only. Input decimal x. Target exact x² without modular
reduction. Hold out x and longer digit lengths.

**Question:** can the representation learn digit products and carry propagation?

**Pass:** ≥99% exact match on held-out x and no collapse at the first held-out
digit length.

**Observed:** The unchanged Gate-0 Transformer reached 98.05% train exact match
but only 7.04% peak held-out same-length exact match and 0.00% on four-digit OOD
after 1,000 steps
(`solving/experiments/2026-07-23_gate1_square/metrics/monitor.jsonl`).
On the isolated 10×10 product table it reached 100% train exact match but stayed
at 15% on 20 symmetrically held-out ordered pairs
(`solving/experiments/2026-07-23_gate1_digit_product/metrics/monitor.jsonl`).
With all 100 products covered, aligned carry-free products reached 100% on
held-out sequences at trained lengths but only 10.0% peak and 7.3% final exact
match at the unseen fourth position
(`solving/experiments/2026-07-23_gate1_aligned_products/metrics/monitor.jsonl`).
With supplied pre-carry column totals for every bounded column count, the plain
Transformer reached 29.69% final train-batch and 30.35% held-out exact match at
1,000 steps; held-out loss was still falling at the cutoff
(`solving/experiments/2026-07-23_gate1_carry_normalize/metrics/monitor.jsonl`).
Extending only the step budget to 4,000 raised held-out exact match to 80.55%
with the peak at the final evaluation
(`solving/experiments/2026-07-23_gate1_carry_normalize_4k/metrics/monitor.jsonl`).
A shared 11,104-parameter continuous carry scan reached 79.45% overall; c6/c7
improved slightly to 71.94% / 56.11%, but exact match still decayed with chain
length
(`solving/experiments/2026-07-23_gate1_carry_scan/metrics/per_c_final.json`).
A fixed right-aligned internal register preserved 99.50% same-length exact
match but produced only 6.70% at length 4
(`solving/experiments/2026-07-23_gate1_fixed_register/metrics/monitor.jsonl`).
Adding 30 length-4 priming rows produced 10.21% peak / 9.28% final exact match
on the other 970 length-4 rows while train and same-length reached 100%
(`solving/experiments/2026-07-23_gate1_length4_priming/metrics/monitor.jsonl`).
Replacing physical RoPE coordinates with coupled decimal-significance
coordinates failed to fit the diagnostic: 48.05% final train-batch, 19.50%
same-length, and 1.50% length-4 exact match
(`solving/experiments/2026-07-23_gate1_position_coupling/metrics/monitor.jsonl`).

**Stop rule:** if this fails, work only on place alignment, digit-pair interaction,
and carry propagation. Do not test modular reduction or T.

## Gate 2 — Modular reduction

Local diagnostic only. Input a decimal dividend a and N. Target a reduced to the
canonical residue. Hold out a, then hold out N.

**Question:** can the representation learn comparison, quotient estimation, and
conditional subtraction independently of squaring?

**Pass:** ≥99% exact match on held-out a for seen N, followed by a clear non-prior
signal on held-out N.

**Stop rule:** if multiplication passes and reduction fails, all research targets
the reduction mechanism. Do not modify the outer T loop.

## Gate 3 — One complete step

Return to x² reduced by N at T=1.

Run three rungs in order:

1. fixed N, held-out x;
2. many training N, held-out x;
3. held-out N.

**Pass:** ≥95% exact match at rung 1; then advance one rung at a time. A model
below this threshold is not a candidate for composition.

## Gate 4 — Reachable-state closure

Evaluate the learned one-step block on states sampled from long true trajectories,
not only clean training starts.

**Question:** is the step correct on the distribution it will create for itself?

**Pass:** no measurable error on the sampled reachable-state suite. Only after
this gate should quantization margin and progressive loss be tested.

## Gate 5 — Held-out-T composition

Freeze the step design. Train on T ∈ {1,2,3}. Evaluate T = 4, 5, 8, 16, 32, 64.

**Pass:** exact match stays flat after the training boundary. A graceful decay is
still failure under exact scoring.

## Gate 6 — Competition viability

Measure steps/s and time-to-gate on the local GPU. Then use one Easy and one
Medium confirmation. Hard remains human-approved only.

**Pass:** the mechanism learns before the scorer’s wall-clock cutoff and repeats
under hosted conditions.

## Budget rule

Use fixed optimizer steps and early stopping for Gates 0–5. A default diagnostic
budget is 1,000 steps, with evaluation every 100 steps and success after three
consecutive perfect evaluations. Use wall-clock budgets only at Gate 6.

## Immediate decision

Do not propose another end-to-end submission. Gate 1 failed. Split it into
single-digit products, aligned partial products, and carry propagation; locate
the first failing primitive before Gate 2 or another T experiment.
