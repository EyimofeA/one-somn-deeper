# Mac → Colab Student Pro workflow

No notebook code yet — process only.

## Prerequisites

- Git repo for this workspace (push to GitHub or other remote)
- Colab Student Pro subscription (private GPU runtime)
- Competition deps installable via `uv` in Colab

## Steps

### 1. Develop on Mac

- Edit code in `solving/` (submissions, experiments)
- For CPU smoke, clone upstream beside this repo (or anywhere) and point the runner at a submission:
  ```bash
  git clone https://github.com/tilde-research/one-layer-deeper.git competition
  cd competition
  uv venv .venv && source .venv/bin/activate && uv sync
  python -m benchmark.runner \
    --manifest benchmark/manifests/smoke_cpu.json \
    --submission-file ../solving/submissions/<name>/submission.py
  ```
- Commit and push to remote

### 2. Open Colab

- New notebook on [colab.research.google.com](https://colab.research.google.com)
- Runtime → Change runtime type → **GPU** (Student Pro for private sessions)

### 3. Clone / pull workspace

```python
# First session
!git clone <your-repo-url> one-somn-deeper
%cd one-somn-deeper

# Later sessions
%cd one-somn-deeper
!git pull
```

### 4. Install competition environment

```bash
git clone https://github.com/tilde-research/one-layer-deeper.git competition
cd competition
uv venv .venv
source .venv/bin/activate   # Colab: use !source or %%bash cell
uv sync
```

### 5. Run GPU experiments

- Point runner at a manifest with CUDA device (e.g. `h100_easy_e1.json` for full easy tier locally if GPU available)
- Or use `one-layer submit` for official remote H100 evaluation

### 6. Record metrics back

- Download `metrics.jsonl` from Colab or copy key numbers into `solving/RESEARCH_LOG.md`
- Commit log updates and push so Mac session has results

### 7. Official submission (when ready)

```bash
one-layer login          # once per machine
one-layer validate solving/submissions/<name>/submission.py
one-layer submit solving/submissions/<name>/submission.py --tier easy --dataset e1 --wait
```

**Hard tier:** user approval required before submit (see `AGENTS.md`).

## What not to sync

- `competition/.venv/` — recreate per machine (folder stays gitignored if you clone it here)
- API keys (`~/.config/one-layer/config.json`) — per-machine login
- Large generated datasets — use competition manifests or regenerate
