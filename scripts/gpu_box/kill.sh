#!/usr/bin/env bash
# Kill leftover training processes on the active GPU box; optional full wipe.
# Usage: kill.sh [--target ubuntu@IP] [--alias oneL40] [--wipe] [--local-only]
#
# Default: stop python/benchmark/monitor jobs on the remote box.
# --wipe:   also delete ~/one-layer-deeper, uv caches (does NOT terminate the cloud VM).
# --local-only: only clear local .gpu_box.json / note; no SSH.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

need python3

ALIAS=""
TARGET=""
WIPE=0
LOCAL_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="$(parse_ssh_target "${2:-}")"; shift 2 ;;
    --alias) ALIAS="${2:-}"; shift 2 ;;
    --wipe) WIPE=1; shift ;;
    --local-only) LOCAL_ONLY=1; shift ;;
    -h|--help)
      sed -n '2,8p' "$0"
      exit 0
      ;;
    *) die "unknown arg: $1" ;;
  esac
done

if [[ -z "$TARGET" && -f "$STATE_FILE" ]]; then
  TARGET="$(read_state_field ssh_target || true)"
  [[ -n "$ALIAS" ]] || ALIAS="$(read_state_field alias || true)"
fi
[[ -n "$ALIAS" ]] || ALIAS="$DEFAULT_ALIAS"
[[ -n "$TARGET" ]] || TARGET="$ALIAS"

if [[ "$LOCAL_ONLY" -eq 1 ]]; then
  mark_state_dead
  echo "local-only: marked state cleaned (did not SSH)"
  exit 0
fi

need ssh
echo "==> connecting $TARGET (alias hint: $ALIAS)"

ssh_run "$TARGET" "WIPE='$WIPE' bash -s" <<'REMOTE'
set -euo pipefail
echo "host=$(hostname) user=$(whoami)"

# Kill competition / research training leftovers owned by this user.
# Broad but avoids killing sshd / system services.
pkill -u "$(id -un)" -f 'benchmark\.runner' 2>/dev/null || true
pkill -u "$(id -un)" -f 'monitor_train' 2>/dev/null || true
pkill -u "$(id -un)" -f 'scripts_local/' 2>/dev/null || true
pkill -u "$(id -un)" -f 'one-layer-deeper' 2>/dev/null || true
# torch leftover workers
pkill -u "$(id -un)" -f 'torch\.distributed' 2>/dev/null || true
sleep 1

echo "remaining user python (if any):"
pgrep -au "$(id -un)" python 2>/dev/null || echo "(none)"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU processes after kill:"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || echo "(none)"
fi

if [ "${WIPE}" = "1" ]; then
  echo "WIPE: removing ~/one-layer-deeper and uv cache"
  rm -rf "$HOME/one-layer-deeper"
  rm -rf "$HOME/.cache/uv" "$HOME/.local/share/uv" 2>/dev/null || true
  # leave uv binary installed
fi

echo "remote cleanup done"
REMOTE

mark_state_dead

echo
echo "CLEANED remote processes on $TARGET"
if [[ "$WIPE" -eq 1 ]]; then
  echo "  wiped ~/one-layer-deeper (+ uv caches)"
fi
echo "  Note: this does NOT stop billing — terminate the Prime/cloud instance in the UI."
echo "  state: $STATE_FILE"
