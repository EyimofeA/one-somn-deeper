#!/usr/bin/env bash
# Bootstrap a Prime/GPU box for One Somn Deeper local runner.
# Usage: start.sh ubuntu@IP [--alias oneL40] [--skip-datasets] [--skip-acceptance] [--force]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

need ssh
need git
need python3
need curl

ALIAS="$DEFAULT_ALIAS"
SKIP_DATASETS=0
SKIP_ACCEPTANCE=0
FORCE=0
TARGET_RAW=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --alias) ALIAS="${2:-}"; shift 2 ;;
    --skip-datasets) SKIP_DATASETS=1; shift ;;
    --skip-acceptance) SKIP_ACCEPTANCE=1; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help)
      sed -n '2,4p' "$0"
      exit 0
      ;;
    *) TARGET_RAW+=("$1"); shift ;;
  esac
done

[[ ${#TARGET_RAW[@]} -gt 0 ]] || die "usage: start.sh ubuntu@IP [--alias oneL40]"
TARGET="$(parse_ssh_target "${TARGET_RAW[*]}")"

echo "==> probing $TARGET"
PROBE="$(ssh_run "$TARGET" 'bash -s' <<'REMOTE'
set -euo pipefail
hostname
nvidia-smi -L 2>/dev/null || true
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
nvidia-smi 2>/dev/null | sed -n "s/.*CUDA Version: \([0-9.]*\).*/CUDA \1/p" | head -1
python3 --version 2>/dev/null || true
df -h / | tail -1
REMOTE
)"
echo "$PROBE"
GPU_NAME="$(echo "$PROBE" | sed -n 's/^GPU 0: //p' | head -1)"
[[ -n "$GPU_NAME" ]] || GPU_NAME="$(echo "$PROBE" | head -3 | tail -1)"
CUDA_VER="$(echo "$PROBE" | sed -n 's/^CUDA //p' | head -1)"
CUDA_MAJOR="${CUDA_VER%%.*}"

if [[ -z "$CUDA_VER" ]]; then
  die "no nvidia-smi / CUDA version on box — is this a GPU node?"
fi

echo "==> CUDA $CUDA_VER (major $CUDA_MAJOR)"
if [[ "$CUDA_MAJOR" -ge 13 ]]; then
  TORCH_MODE="uv-sync"
elif [[ "$CUDA_MAJOR" -eq 12 ]]; then
  TORCH_MODE="cu126"
else
  die "unsupported CUDA $CUDA_VER — expected 12.x or 13.x"
fi
echo "==> torch install mode: $TORCH_MODE"

echo "==> upserting ssh Host $ALIAS"
upsert_ssh_config "$ALIAS" "$TARGET"

echo "==> remote bootstrap (uv + clone + venv)"
ssh_run "$TARGET" "TORCH_MODE='$TORCH_MODE' FORCE='$FORCE' bash -s" <<'REMOTE'
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

# Fix root-owned ~/.config if needed (Prime images sometimes ship this)
if [ -d "$HOME/.config" ] && [ ! -w "$HOME/.config" ]; then
  sudo chown -R "$(id -un):$(id -gn)" "$HOME/.config" || true
fi
mkdir -p "$HOME/.config/uv" "$HOME/.local/bin"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

if [ "${FORCE}" = "1" ] && [ -d "$HOME/one-layer-deeper" ]; then
  echo "FORCE: removing existing ~/one-layer-deeper"
  rm -rf "$HOME/one-layer-deeper"
fi

if [ ! -d "$HOME/one-layer-deeper/.git" ]; then
  git clone https://github.com/tilde-research/one-layer-deeper.git "$HOME/one-layer-deeper"
fi
cd "$HOME/one-layer-deeper"
git pull --ff-only || true
echo "HEAD=$(git rev-parse --short HEAD)"

if [ "${FORCE}" = "1" ] && [ -d .venv ]; then
  rm -rf .venv
fi

uv python install 3.13.5 || uv python install 3.13
uv venv --python 3.13.5 .venv || uv venv --python 3.13 .venv
source .venv/bin/activate

if [ "$TORCH_MODE" = "uv-sync" ]; then
  uv sync
else
  # CUDA 12.x driver: plain uv sync pulls cu13 and breaks. Pin cu126 wheels.
  uv pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu126
  uv pip install numpy==2.5.0 fastapi httpx jsonargparse==4.49.0 "psycopg[binary]" python-multipart "uvicorn[standard]"
  uv pip install -e . --no-deps
  echo "NOTE: on this cu126 box, never run bare 'uv sync' afterward."
fi

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
test "$(python -c 'import torch; print(torch.cuda.is_available())')" = "True"
REMOTE

TORCH_LINE="$(ssh_run "$TARGET" 'source ~/one-layer-deeper/.venv/bin/activate && python -c "import torch; print(torch.__version__)"')"

if [[ "$SKIP_DATASETS" -eq 0 ]]; then
  echo "==> generating datasets (one-time, ~20–60s)"
  ssh_run "$TARGET" 'bash -s' <<'REMOTE'
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd ~/one-layer-deeper && source .venv/bin/activate
bash scripts/generate_datasets.sh
ls data/generated | wc -l
REMOTE
else
  echo "==> skipping datasets"
fi

# Seed a known d32 K4 card for acceptance if present locally in git
ACCEPT_CARD="$REPO_ROOT/solving/experiments/2026-07-21_depth_d32_k4_ut_optsched/submission.py"
if [[ ! -f "$ACCEPT_CARD" ]]; then
  # try blob from git
  if git -C "$REPO_ROOT" cat-file -e 26b1c44571415ebaedee7d3e44656417e269dd35 2>/dev/null; then
    mkdir -p /tmp/osmn_gpu
    git -C "$REPO_ROOT" show 26b1c44571415ebaedee7d3e44656417e269dd35 > /tmp/osmn_gpu/submission.py
    ACCEPT_CARD=/tmp/osmn_gpu/submission.py
  fi
fi

if [[ -f "$ACCEPT_CARD" ]]; then
  echo "==> scp acceptance card"
  ssh_run "$TARGET" 'mkdir -p ~/one-layer-deeper/submissions'
  scp -o StrictHostKeyChecking=accept-new "$ACCEPT_CARD" \
    "$TARGET:~/one-layer-deeper/submissions/depth_d32_k4_ut_optsched.py"
fi

if [[ "$SKIP_ACCEPTANCE" -eq 0 && -f "$ACCEPT_CARD" ]]; then
  echo "==> acceptance throughput (~90s wall, timeout expected)"
  ssh_run "$TARGET" 'bash -s' <<'REMOTE' || true
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd ~/one-layer-deeper && source .venv/bin/activate
# macOS/Linux portable: prefer timeout if present
run() {
  if command -v timeout >/dev/null 2>&1; then
    timeout 90 "$@"
  else
    "$@" &
    pid=$!
    sleep 90
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}
CUDA_VISIBLE_DEVICES=0 run python -m benchmark.runner \
  --manifest benchmark/manifests/h100_medium_m1.json \
  --submission-file submissions/depth_d32_k4_ut_optsched.py \
  2>&1 | tee /tmp/osmn_acceptance.log | tail -15
# rough steps/s from last window if possible
python3 - <<'PY'
import re
from pathlib import Path
text = Path("/tmp/osmn_acceptance.log").read_text() if Path("/tmp/osmn_acceptance.log").exists() else ""
rows = [(int(s), float(e)) for s, e in re.findall(r"step=(\d+).*elapsed=([0-9.]+)s", text)]
if len(rows) >= 2:
    (s0, e0), (s1, e1) = rows[len(rows)//3], rows[-1]
    if e1 > e0:
        print(f"approx_steps_per_s={(s1-s0)/(e1-e0):.1f} (window step {s0}->{s1})")
PY
REMOTE
else
  echo "==> skipping acceptance"
fi

write_state "$TARGET" "$ALIAS" "$GPU_NAME" "$CUDA_VER" "$TORCH_LINE"

echo
echo "READY"
echo "  ssh $ALIAS   # or ssh $TARGET"
echo "  cd ~/one-layer-deeper && source .venv/bin/activate"
echo "  state: $STATE_FILE"
echo "  kill:  $SCRIPT_DIR/kill.sh [--wipe]"
