#!/usr/bin/env bash
# Show active GPU box state + quick remote sanity.
# Usage: status.sh [--target ubuntu@IP]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

TARGET=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="$(parse_ssh_target "${2:-}")"; shift 2 ;;
    -h|--help) echo "usage: status.sh [--target ubuntu@IP]"; exit 0 ;;
    *) die "unknown arg: $1" ;;
  esac
done

if [[ -f "$STATE_FILE" ]]; then
  echo "==> local state ($STATE_FILE)"
  cat "$STATE_FILE"
  echo
  [[ -n "$TARGET" ]] || TARGET="$(read_state_field ssh_target || true)"
else
  echo "==> no local state file at $STATE_FILE"
fi

[[ -n "$TARGET" ]] || die "no target (pass --target or run start.sh first)"

echo "==> remote $TARGET"
ssh_run "$TARGET" 'bash -s' <<'REMOTE' || die "ssh failed — box may be terminated"
set -euo pipefail
hostname
nvidia-smi -L 2>/dev/null || echo "no GPU"
if [ -d "$HOME/one-layer-deeper/.venv" ]; then
  source "$HOME/one-layer-deeper/.venv/bin/activate"
  python -c "import torch; print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
else
  echo "no ~/one-layer-deeper/.venv"
fi
echo "user python procs:"
pgrep -au "$(id -un)" python 2>/dev/null || echo "(none)"
REMOTE
