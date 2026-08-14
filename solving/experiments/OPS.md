# Ops

## Daily quotas

| Tier | / UTC day |
|------|-----------|
| Easy | 60 |
| Medium | 6 |
| Hard | 1 |

**Deadline:** submissions close **August 31, 2026 10:00 PM PT** (upstream `service/competition.py`).

**Upstream pin:** local `competition/` clone should be at `8a3c78d` (Hard = Max T /
OOD N Max T / next-rung tie-break; `token_training_loss` available). On the GPU
box: `cd ~/one-layer-deeper && git pull --ff-only` before smoke; regenerate
datasets if `scripts/generate_datasets.sh` changed.

When two agents share a day: split Medium (e.g. 3+3). Hard = principal only. Update `left` from CLI after submits.

## Noise

Easy/Medium manifests use **one seed (74)**. Estimate σ by resubmitting the **same** file n times.

| Goal | n |
|------|---|
| Rough σ | 3 |
| Promote tiny Δ (~0.3 pp) | 5, or demand larger Δ / dual gate |

Promote only if beat champ by ≳2σ **or** clear win on **e5 + Medium** (not e1 alone).

## Schedule rule

Never `CosineAnnealingLR` with small `T_max ≈ c×seconds` on Medium/Hard. Prefer inv-sqrt/Noam or clamped cosine. See `learnings/concepts/15-lr-schedules-wallclock.md`.

**`solving/experiments/2026-07-21_depth_d32_k4_ut/submission.py` still has the
un-patched buggy scheduler** (`t_max = seconds * 8`) — confirmed sawtoothing
every ~7600 steps on the L40S (2026-07-22, T=1 probe: train accuracy
oscillated 35%→100%→35% for 24k steps instead of converging). Copy
`2026-07-22_t1only_probe_ut_k4/submission.py` instead if you need a UT-loop
card — it's the same architecture with the wall-clock scheduler applied.
Don't `cp` the k4_ut file directly without checking `_build_scheduler` first.

## GPU box — local training, zero quota

**Quick path (preferred):**

```bash
./scripts/osmn gpu start ubuntu@IP          # bootstrap + ssh alias + datasets
./scripts/osmn gpu status
./scripts/osmn gpu kill                    # stop leftover train/monitor procs
./scripts/osmn gpu kill --wipe             # also delete ~/one-layer-deeper on box
```

Agent skill: `.cursor/skills/osmn-gpu-box/`. Kill does **not** stop cloud billing — terminate the instance in the provider UI.

Current: **no active Prime compute pod** (confirmed 2026-08-14 after the final
Neural GPU experiments). The most recent box is recorded as terminated in
`solving/experiments/.gpu_box.json`; `oneL40` is not a live endpoint.

Historical 2026-08-03 box: Prime Intellect **L40** at `216.81.248.94`.
Upstream clone `~/one-layer-deeper` was at **`8a3c78d`**, torch
**2.12.1+cu130**, CUDA 13.0.
**Ephemeral: IP/host below only valid while this instance is up.** Local
`benchmark.runner` runs cost nothing — use this for everything in
`learnings/concepts/17-recurrence-generalisation.md` (wd sweep, T-curve,
re-quantised recurrence). Only spend real quota to confirm a result on the
actual H100 scorer.

Prior boxes dead — do not reuse: `216.81.245.246` (2026-07-24),
`204.52.24.142` (cu126-only; never copy that “never `uv sync`” rule here).

### Connect

```bash
ssh oneL40
# or: ssh ubuntu@216.81.248.94
```

Local alias in `~/.ssh/config`: `ssh oneL40`. Agents: read
`solving/experiments/.gpu_box.json` for the live target.

Cold-start sanity (before trusting the box):

```bash
cd ~/one-layer-deeper && source .venv/bin/activate
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: 2.12.1+cu130 True NVIDIA L40
```

### Environment (set up 2026-08-03)

Repo: `~/one-layer-deeper` (fresh clone of upstream, **not** this workspace).

```bash
cd ~/one-layer-deeper
source .venv/bin/activate
```

Stack: Python 3.13.5, **torch 2.12.1+cu130**, numpy 2.5.0. Driver
`580.126.09`, **CUDA Version: 13.0** — matches `pyproject.toml`'s default
torch resolution, so **plain `uv sync` works** (same as box #2 A6000). Do not
copy the old oneL40 cu126 workaround here.

Rebuild from scratch if the env is toast:

```bash
cd ~/one-layer-deeper
rm -rf .venv
uv venv --python 3.13.5 .venv
source .venv/bin/activate
uv sync
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Datasets generated 2026-08-03 (`bash scripts/generate_datasets.sh`). Acceptance
smoke used the card scp'd by `osmn gpu start`.

### Run something

Datasets are already generated (`bash scripts/generate_datasets.sh`, one-time,
already done). Copy a submission card over and run it against a real manifest:

```bash
# from local machine
scp "solving/submissions/<card>/submission.py" oneL40:~/one-layer-deeper/submissions/<card>.py

# on the box
cd ~/one-layer-deeper && source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python -m benchmark.runner \
  --manifest benchmark/manifests/h100_medium_m1.json \
  --submission-file submissions/<card>.py
```

Manifests available after upstream `4ceff95`: `h100_easy_e1..e10`,
`h100_medium_m1..m10`, and `smoke_cpu`
(no dataset needed, use for a fast end-to-end check). Hard is
hosted-submission only — not runnable locally.

### Acceptance test — run this first on any new/resumed box

Confirms the box isn't launch-bound worse than expected before trusting any
throughput number off it (see `16-representation-vs-throughput.md` — we are
kernel-launch-bound at d=32, so steps/s is a CPU-dispatch measurement, not a
GPU one).

```bash
CUDA_VISIBLE_DEVICES=0 timeout 90 python -m benchmark.runner \
  --manifest benchmark/manifests/h100_medium_m1.json \
  --submission-file submissions/<any d=32 K=4 card>.py
```

Known reference: d=32 K=4 on the real H100 scorer runs **96.8 steps/s**.
This L40S (2026-07-24, `depth_d32_k4_ut_optsched` on m1) measured **~148
steps/s** (step 8600→12500 in 26.4s of wall after warmup) — in line with the
prior L40S ~145 and faster than the H100 baseline (launch-bound at d=32). If a
box measures well under ~70 steps/s, something about the instance (vCPU
allocation, noisy neighbor) is bad — don't trust its wall-clock numbers, get
a different one.

### Local-only diagnostic tooling (agent-owned, not the evaluator runner)

Two things live on the box that are **not** upstream and **not** the
frozen scorer — don't confuse them with `benchmark/runner.py` (untouched)
when reading logs:

- `benchmark/manifests/local_e1_overtrain20x.json` — a copy of
  `h100_easy_e1.json` with `total_training_time_seconds` bumped 60→1200
  (20x) and `log_every` raised, same data/seed. For asking "does this
  card ever generalize given more time," not for scoring.
- `scripts_local/monitor_train.py` — reimplements the real training loop
  by importing `benchmark.runner`'s own internals (`_loss_and_accuracy`,
  `_evaluate`, `_compile_model`, etc.) so behavior stays faithful, but
  adds what the real runner doesn't do: **held-out eval every N steps**
  (real runner evaluates exactly once, at the end) plus weight-norm and
  grad-norm logging. Writes JSONL (`--out`), one line per train/eval
  event. Usage:

  ```bash
  python scripts_local/monitor_train.py \
    --manifest benchmark/manifests/<manifest>.json \
    --submission-file submissions/<card>.py \
    --eval-every 500 \
    --out /tmp/<card>_monitor.jsonl
  ```

  Multiple runs can share the GPU concurrently (46GB VRAM, these are
  small models) — expect wall-clock to stretch under contention, which is
  fine for a diagnostic run but means **never** use a concurrent run to
  measure steps/s for schedule calibration.

## GPU box #2 — RTX A6000, second box to avoid contention

Rented Prime Intellect A6000 (2026-07-23), specifically so two agent sessions
running local diagnostics don't fight over one box's steps/s the way `oneL40`
contention skewed a couple of runs earlier in the day. **oneL40 went fully
unreachable (SSH times out at the connection level, not just idle) the same
day this box was rented — likely terminated, not just idle.** Its
non-git-tracked contents (base `scripts_local/monitor_train.py`, Fable's
`tok_discriminator.py`, and the generated datasets behind
`local_t1only_fixedn_{323,1073}.json`) did not survive and would need
regenerating if oneL40 doesn't come back. Anything checkpointed/committed to
git from that box (peak/final checkpoints, prediction dumps, metrics) is safe
regardless.

### Connect (box #2)

```bash
ssh ubuntu@64.247.206.65
```

Local machine alias: `ssh twoA6000`.

### Environment (set up 2026-07-23)

Fresh clone at `~/one-layer-deeper` (upstream, not this repo — same
convention as oneL40). **Unlike oneL40, this box's driver supports CUDA 13**
(`580.126.09`, `CUDA Version: 13.0`), which matches `pyproject.toml`'s
default pin — so **plain `uv sync` works here**, no cu126 wheel-index
workaround needed. Do not copy the "never `uv sync`" oneL40 rule to this box;
it's driver-specific to oneL40, not a general project rule.

```bash
cd ~/one-layer-deeper && source .venv/bin/activate
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# confirmed: 2.12.1+cu130 True NVIDIA RTX A6000
```

`scripts_local/` seeded with `monitor_train_ckpt.py` (peak-checkpointing
variant, tracked at
`solving/experiments/2026-07-23_t1only_fixedn_width/monitor_train_ckpt.py`)
and `dump_predictions.py` (same dir). Base `monitor_train.py` and
`tok_discriminator.py` not yet re-created here — see the oneL40-loss note
above.

Manifests/datasets not yet generated on this box — run
`bash scripts/generate_datasets.sh` for the standard hosted-mirror set before
using `h100_*` manifests, and regenerate the rung-1 fixed-N datasets
separately if picking that thread back up (params: `split_group=x`, fixed
semiprime, `separate_input_output=true` — the causal_lm default leaks
answers into `input_ids`, see the gotcha already documented in
`2026-07-23_t1only_fixedn_wd01/NOTE.md`).
