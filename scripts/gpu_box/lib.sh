#!/usr/bin/env bash
# Shared helpers for osmn GPU box scripts.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATE_FILE="${OSMN_GPU_STATE:-$REPO_ROOT/solving/experiments/.gpu_box.json}"
DEFAULT_ALIAS="${OSMN_GPU_ALIAS:-oneL40}"
REMOTE_DIR="${OSMN_REMOTE_DIR:-\$HOME/one-layer-deeper}"

die() { echo "error: $*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"; }

parse_ssh_target() {
  # Accept: ubuntu@1.2.3.4 | ssh ubuntu@1.2.3.4 | 1.2.3.4 | user@host -p 22
  local raw="$*"
  raw="${raw#ssh }"
  raw="$(echo "$raw" | xargs)"
  [[ -n "$raw" ]] || die "empty SSH target"
  if [[ "$raw" != *@* ]]; then
    raw="ubuntu@$raw"
  fi
  echo "$raw"
}

ssh_run() {
  local target="$1"; shift
  ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new "$target" "$@"
}

write_state() {
  local target="$1" alias="$2" gpu="$3" cuda="$4" torch="$5"
  mkdir -p "$(dirname "$STATE_FILE")"
  cat >"$STATE_FILE" <<EOF
{
  "ssh_target": "$target",
  "alias": "$alias",
  "gpu": "$gpu",
  "cuda_version": "$cuda",
  "torch": "$torch",
  "remote_dir": "~/one-layer-deeper",
  "updated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "status": "active"
}
EOF
  echo "wrote $STATE_FILE"
}

mark_state_dead() {
  if [[ -f "$STATE_FILE" ]]; then
    python3 - "$STATE_FILE" <<'PY'
import json, sys, datetime
path = sys.argv[1]
try:
    data = json.load(open(path))
except Exception:
    data = {}
data["status"] = "cleaned"
data["cleaned_at"] = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
json.dump(data, open(path, "w"), indent=2)
print("marked", path, "cleaned")
PY
  fi
}

read_state_field() {
  local field="$1"
  [[ -f "$STATE_FILE" ]] || return 1
  python3 -c "import json; print(json.load(open('$STATE_FILE')).get('$field',''))"
}

upsert_ssh_config() {
  local alias="$1" target="$2"
  local user host
  user="${target%@*}"
  host="${target#*@}"
  local cfg="${HOME}/.ssh/config"
  touch "$cfg"
  python3 - "$cfg" "$alias" "$user" "$host" <<'PY'
from pathlib import Path
import re, sys
cfg, alias, user, host = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
text = cfg.read_text() if cfg.exists() else ""
block = f"Host {alias}\n  HostName {host}\n  User {user}\n"
if f"Host {alias}\n" in text:
    text, n = re.subn(rf"Host {re.escape(alias)}\n(?:  .*\n)*", block, text, count=1)
    if n == 0:
        text = block + "\n" + text
else:
    if "\nHost *\n" in text:
        text = text.replace("\nHost *\n", "\n" + block + "\nHost *\n", 1)
    else:
        text = text.rstrip() + "\n\n" + block
cfg.write_text(text)
print(f"ssh config: Host {alias} -> {user}@{host}")
PY
}

detect_cuda_major() {
  local target="$1"
  ssh_run "$target" 'nvidia-smi 2>/dev/null | sed -n "s/.*CUDA Version: \([0-9.]*\).*/\1/p" | head -1'
}
