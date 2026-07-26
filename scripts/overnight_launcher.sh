#!/usr/bin/env bash
# Local process.  It is started by nohup after the requested two-hour delay.
set -u -o pipefail

workspace="/Users/eadebayo/Developer/one somn deeper"
run_id="$(date +%Y%m%dT%H%M%S%z)"
remote_source="/home/ubuntu/one-somn-overnight-$run_id"
deadline_epoch="$(( $(date +%s) + 4 * 60 * 60 ))"
launcher_log="$workspace/research/overnight_launcher_$run_id.log"

exec > >(tee -a "$launcher_log") 2>&1
printf 'launcher start: '; date -Is
ssh twoA6000 'date -Is; hostname; nvidia-smi; find ~/one-layer-deeper/runs -mindepth 1 -maxdepth 1 -type d -printf "%f\\n" 2>/dev/null | sort | tail -n 100' || exit 1

ssh twoA6000 "mkdir -p '$remote_source'"
rsync -a --delete \
  --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
  --exclude 'diagnostics/runs' --exclude 'diagnostics/analysis_out' \
  --exclude 'diagnostics/.venv' \
  "$workspace/" "twoA6000:$remote_source/"

source_commit="$(git -C "$workspace" rev-parse HEAD)"
ssh twoA6000 "printf '%s\\n' '$source_commit' > '$remote_source/source_commit.txt'"

ssh twoA6000 "bash '$remote_source/scripts/overnight_remote.sh' '$remote_source' '$run_id' '$deadline_epoch'"

mkdir -p "$workspace/research/overnight_runs/$run_id"
mkdir -p "$workspace/reports"
rsync -a --ignore-missing-args "twoA6000:$remote_source/diagnostics/runs/overnight_$run_id/" "$workspace/research/overnight_runs/$run_id/"
rsync -a --ignore-missing-args "twoA6000:$remote_source/reports/" "$workspace/reports/"
rsync -a --ignore-missing-args "twoA6000:$remote_source/research/overnight_handoff.md" "$workspace/research/overnight_handoff.md"
printf 'launcher finished: '; date -Is
