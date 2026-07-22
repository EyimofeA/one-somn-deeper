# Ops

## Daily quotas

| Tier | / UTC day |
|------|-----------|
| Easy | 60 |
| Medium | 6 |
| Hard | 1 |

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

## GPU box — local training, zero quota

Rented Prime Intellect L40S. **Ephemeral: IP/host below only valid while this
instance is up.** Local `benchmark.runner` runs cost nothing — use this for
everything in `learnings/concepts/17-recurrence-generalisation.md` (wd sweep,
T-curve, re-quantised recurrence). Only spend real quota to confirm a result on
the actual H100 scorer.

### Connect

```bash
ssh ubuntu@204.52.24.142 -p 22
```

Local machine has an alias in `~/.ssh/config`: `ssh oneL40`.

Cold-start sanity (before trusting the box):

```bash
cd ~/one-layer-deeper && source .venv/bin/activate
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: 2.12.1+cu126 True NVIDIA L40S
```

### Environment (already set up on this box, 2026-07-22)

Repo: `~/one-layer-deeper` (fresh clone of upstream, **not** this repo — this
repo dropped its nested clone in `055928d`).

```bash
cd ~/one-layer-deeper
source .venv/bin/activate
```

Stack: Python 3.13.5, **torch 2.12.1+cu126** (not the pyproject default —
see below), numpy 2.5.0, CUDA confirmed working against the box's driver
(565.57.01, CUDA 12.7 max).

**Why cu126 and not the pinned default:** `pyproject.toml` pins
`torch==2.12.1` unversioned, and `uv sync` resolves that to a CUDA 13 build
by default. This box's driver only supports up to CUDA 12.7, so the CUDA-13
wheel fails with `RuntimeError: CUDA driver too old`. Fixed by installing
from the cu126 wheel index instead:

```bash
uv pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu126
```

Do **not** run `uv sync` or bare `uv run` after this — both re-resolve
against the (unmodified) lockfile/pyproject and will pull the cu13 build
back in, plus a mismatched NCCL (`undefined symbol: ncclCommResume`). Always
`source .venv/bin/activate` and invoke `python` directly instead. If the env
ever gets into that broken mixed cu12/cu13 state, don't chase it — nuke and
rebuild:

```bash
cd ~/one-layer-deeper
rm -rf .venv uv.lock
uv venv --python 3.13.5 .venv
source .venv/bin/activate
uv pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu126
uv pip install numpy==2.5.0 fastapi httpx jsonargparse==4.49.0 "psycopg[binary]" python-multipart "uvicorn[standard]"
uv pip install -e . --no-deps
```

(`pyproject.toml` on the box also has two small local edits vs upstream:
the `[[tool.uv.index]]` block is absent — don't re-add it, that's what
caused the cu13/cu12 mixing — and `build-system.requires` is
`setuptools>=78` not `>=80`, since only `78.1.0` is published for this
Python. Neither edit is committed anywhere; they only exist on this box.)

`modal` is intentionally not installed — it's only needed for the hosted
deploy path, not local training. 70/77 upstream tests pass; the 7 failures
are all `ModuleNotFoundError: No module named 'modal'` in deploy/service
tests, harmless for local runs.

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

Manifests available: `h100_easy_e1..e5`, `h100_medium_m1..m5`, `smoke_cpu`
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
This L40S measured **~145 steps/s** on the same config (2026-07-22, step
6800→12700 in 40.7s) — faster than the H100 baseline, consistent with
launch-bound behavior depending more on host dispatch than raw GPU
bandwidth. If a box measures well under ~70 steps/s, something about the
instance (vCPU allocation, noisy neighbor) is bad — don't trust its
wall-clock numbers, get a different one.
