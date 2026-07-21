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
