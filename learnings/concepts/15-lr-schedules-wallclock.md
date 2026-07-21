# 15 — LR schedules under wall-clock (no fixed T_max)

## Mistake we made (2026-07-21)

Used `CosineAnnealingLR` with

```text
T_max ≈ 8 × training_time_seconds
```

| Tier | T_max we set | Steps we actually took | What happened |
|------|--------------|------------------------|---------------|
| Easy 60s | ~480 | ~400–2500 | Mostly OK / mild mismatch |
| Medium 600s | ~4800 | **~45k–70k** | Cosine **finished and restarted** → LR sawtooth for most of training |

PyTorch `CosineAnnealingLR` is **periodic** past `T_max`. Tiny models + long clocks ⇒ schedule underestimates steps ⇒ optimizer gets kicked repeatedly. Medium scores stayed ~0.1% until we fixed the schedule.

**Do not** size cosine from a small ×seconds factor unless you have measured steps/sec for *that* arch×tier.

## What the evaluator allows

`scheduler.step()` is called **with no arguments** after every optimizer step (`competition/benchmark/runner.py`). So:

| Schedule | OK? |
|----------|-----|
| Warmup + **inv-sqrt / Noam** (step-only) | **Yes — preferred** |
| Warmup + cosine with progress **clamped at 1** (no restart) | Yes (our `optsched` patch) |
| CosineAnnealingLR with T_max ≫ any plausible step count | Risky but workable |
| `ReduceLROnPlateau` | **No** — needs `step(metric)`; evaluator never passes loss |
| Schedules that assume a known `total_steps` from the API | No — API only gives `training_time_seconds` |

## Adaptive default (no fixed horizon)

Noam / transformer inv-sqrt — depends only on step index and warmup:

```text
lr_mult(step) = d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})
```

Or scale-free version (peak at end of warmup, then 1/√step):

```text
if step < warmup:  mult = step / warmup
else:              mult = (warmup / step) ** 0.5
```

Never restarts. Works on Easy 400 steps and Medium 70k steps without retuning `T_max`.

Reference implementation path: prefer this over cosine for new cards; keep `depth_d32_k4_ut_optsched` as the clamped-cosine Medium fix already scored.

## Checklist before Medium/Hard

1. Estimate steps: Easy steps × (tier_seconds / 60) as a lower bound; tiny d=32 can be **much** faster.
2. If using cosine: either clamp progress or set T_max ≥ 2× that estimate.
3. Prefer inv-sqrt/Noam when unsure.
4. One-change rule: don’t change arch and schedule in the same card when diagnosing this class of bug.

---

## Reconciliation of the two fixes (added by Claude Code, 2026-07-21)

Two fixes for the sawtooth landed independently on the same day. **Both correctly remove
the restart**; they differ in whether the anneal actually *completes*. Recording the
comparison so we stop maintaining two answers.

| | mechanism | reproducible? | anneal completes? |
|---|---|---|---|
| `depth_d32_k4_ut_optsched` | step-based, `t_max = seconds × 120`, progress clamped at 1 | **yes** (LR depends only on step index) | only if steps/s ≈ 120 |
| `claude_pv_*_tsched`, `claude_hard_h1` | wall-clock, `progress = elapsed / training_time_seconds` | no (LR depends on machine speed) | **always** |

### The problem with a fixed multiplier: steps/s varies 15× across arch × tier

Measured, this session:

| run | steps | steps/s |
|---|---|---|
| Easy e1, d=32 K=4 | 380 | **6.3** |
| Easy e5, d=32 K=4 | 1,945 | 32.4 |
| Easy e5, d=2048 | 1,765 | 29.4 |
| Medium m1, d=32 K=4 | 58,060 | **96.8** |
| Medium m1, d=32 K=T | 30,249 | 50.4 |
| Hard, d=2048 K=4 | 190,017 | 52.8 |

Throughput depends on the dataset (dataloader-bound on tiny Easy sets), the width, and
the loop count. No single multiplier covers a 6.3 → 96.8 range.

### End-of-training LR as a fraction of base

| run | original ×8 | optsched ×120 | wall-clock |
|---|---|---|---|
| Easy e1, d=32 K=4 | 73.3% | **100.0%** | 1.0% |
| Easy e5, d=32 K=4 | 1.0% *(sawtooth)* | 87.5% | 1.0% |
| Medium m1, d=32 K=4 | 1.0% *(sawtooth)* | **10.8%** | 1.0% |
| Hard, d=2048 K=4 | 1.0% *(sawtooth)* | 64.3% | 1.0% |

Two things to take from this table:

1. **`×120` is calibrated to exactly one configuration** — d=32 K=4 on Medium, the run it
   was derived from (96.8 steps/s → 10.8% end LR, healthy). Everywhere else it
   under-anneals: the Hard run would have ended at **64% of peak LR** with no
   consolidation phase, and Easy e1 would end at **100%** — no annealing whatsoever.
2. **`×120` is a regression on Easy.** The original `×8` was well-calibrated there
   (t_max 480 vs ~380 actual steps). Do **not** apply `optsched` to Easy cards; it turns
   a working schedule into a constant-LR run.

### Recommendation

**Default to the wall-clock form.** The benchmark's objective is literally "best model
after N seconds", so annealing along wall-clock anneals along the axis the objective is
defined on, and it is correct at 60s / 600s / 3600s and at any width or loop count with
no retuning. Reference implementation: `solving/submissions/claude_hard_h1/submission.py`
(`_build_scheduler`); verified over 199,186 steps in a compressed window — monotonic
through the second half, ends at the 1% floor.

**Known cost:** LR becomes a function of machine speed, so step-for-step reproducibility
is lost. That matters, because e1 was previously *bit*-reproducible (4.67×3, 5.83×3
identical). If a future experiment needs exact reproducibility more than a correct
anneal, use the step-based form — but re-measure steps/s for that exact arch × tier
first, and never reuse a multiplier across tiers.

Inverse-sqrt (the preference stated above) also never restarts and needs no horizon, but
it never reaches a low final LR either — it decays as 1/√step, so it has the same
missing-consolidation weakness as an under-annealed cosine. It is the safe choice when
the horizon is genuinely unknown; wall-clock is better when, as here, the horizon is
known in seconds.
