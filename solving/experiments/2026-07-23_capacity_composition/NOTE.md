# 2026-07-23 — Capacity vs. composition on the digit-squaring cell

Session continued from Codex's `solving/research/` gate ladder. Codex spent 6
experiments trying to lift the one-step four-digit squaring cell off its 85%
ceiling by changing the *mechanism* (arity-fold init, balanced-tree fold, soft
carry prototypes, aux-column supervision, digit-3-weighted loss, symmetric pair
table) — all 6 refuted. Every card held `D_MODEL = 64` fixed. This note tests
two variables nobody touched: the LR-schedule basis, and width.

All runs use `scripts_local/monitor_train_ckpt.py` (peak-checkpointing), 600s
budget, on `twoA6000`. Model = `solving/research/soft_digit_squaring_recurrence.py`.

## 1. Schedule fix (wall-clock, one variable)

Codex's `build_optimizer` used a step-count cosine (`TOTAL_STEPS=10_000`) against
a 600s manifest that only fits ~4-4.5k steps — so LR never annealed, and the
one-step run peaked at 85.35% (step 3,500) then *decayed* to 83.55%. Swapped in
the project's validated `time.monotonic()` wall-clock scheduler
(WARMUP_FRACTION=0.05, FINAL_LR_FRACTION=0.05); everything else identical.

- **Result:** decay eliminated — monotone, peak == final == **82.55%**. But the
  peak is ~3 pts *below* the original's transient 85.35%. Reading: the step-count
  "bug" let it briefly bounce to 85% before the still-high LR knocked it back;
  that spike was unstable, not recoverable. The fix trades an unstable 85% spike
  for a trustworthy 82.55% floor (saved checkpoint == the good weights). No peak
  gain. Schedule was not the blocker.

## 2. Width sweep on the one-step gate (the real finding)

Same one-step four-digit square (x² mod 10^4), wall-clock schedule, only D_MODEL
changes:

| D_MODEL | one-step held-x peak |
|---|---|
| 64 (Codex's ceiling) | 82.55% |
| 128 | **97.8%** |

**Capacity was the hidden blocker.** Codex's "representational ceiling" on the
one-step primitive was largely capacity starvation — the 6 mechanism changes all
failed because the answer was just "make it wider." First clean win of the thread.

## 3. Composition at capacity — the wall (also the real finding)

Does capacity also fix the *composed* recurrence? Ran the same cell on the
held-T split (train T=1,2 → test T=3, `local_soft_digit_recurrence.json`) at
d=128 and d=256. Codex's d=64 baseline scored 40% held-T (bases 0..3 only).

| D_MODEL | held-T (T=3) peak | trajectory |
|---|---|---|
| 64 (Codex) | 40.0% | — |
| 128 | **50.0%** | flat from step 300 → 6000 |
| 256 | **50.0%** | flat from step 200 → 6100 |

**Composition is a hard, capacity-immune wall.** d=128 and d=256 land on the
*exact same* 50.0% and plateau immediately — width buys one extra base (4→5 of 10)
and nothing more. The one-step cell is ~98% accurate, but iterating it 3× collapses
to 50% because the sub-100% per-step error compounds through the STE-discretized
recurrence, and capacity cannot fix accumulation.

## Synthesis / next

- One-step digit squaring (mod 10^4): **capacity-solvable** (98% @ d=128).
- Iterating it (composition): **hard wall at 50%**, immune to width and steps.
- The next real attempt must attack **error accumulation across steps**
  (e.g. an exactly-discrete/verified per-step state, or a residual/correction
  path between iterations), not capacity and not the single-step mechanism.
- **Caveat for Hard:** all of this is mod 10^4 (truncation), NOT mod N (the real
  task's held-out-semiprime reduction). Capacity's one-step win does not transfer
  to mod-N — the earlier plain-transformer width sweep
  (`2026-07-23_t1only_fixedn_width/`) stayed flat at 0% for d=32/128/256 on N=1073.
  Modular reduction mod N remains the untouched core.

Metrics: `metrics/{soft_digit_wallclock,soft_digit_d128,comp_d128,comp_d256}_monitor.jsonl`.
Peak checkpoints left box-local on `twoA6000:/tmp/*_peak.pt`.
