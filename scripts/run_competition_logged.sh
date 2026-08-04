#!/usr/bin/env bash
# User-owned local wrapper. It does not alter the evaluator or submission.
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "usage: $0 <name> <manifest> <submission.py> <run-root>" >&2
  exit 2
fi

name="$1"
manifest="$2"
submission="$3"
run_root="$4"
run_dir="$run_root/$name"
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

mkdir -p "$run_dir/checkpoints"
if [ -n "${RESEARCH_COMMIT:-}" ]; then
  printf '%s\n' "$RESEARCH_COMMIT" > "$run_dir/git_commit.txt"
else
  git -C "$repo_root" rev-parse HEAD > "$run_dir/git_commit.txt"
fi
git rev-parse HEAD > "$run_dir/evaluator_commit.txt"
cp "$manifest" "$run_dir/manifest.json"
cp "$submission" "$run_dir/submission.py"
python - "$name" "$manifest" "$submission" "$run_dir" <<'PY' > "$run_dir/config.json"
import json, pathlib, sys
print(json.dumps({"name": sys.argv[1], "manifest": sys.argv[2], "submission": sys.argv[3], "run_dir": sys.argv[4]}, indent=2))
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name --format=csv,noheader > "$run_dir/gpu.txt" || true
  nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits \
    --loop-ms=1000 > "$run_dir/gpu.csv" 2>/dev/null &
  telemetry_pid=$!
  trap 'kill "$telemetry_pid" 2>/dev/null || true' EXIT
fi

started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
python -m benchmark.runner --manifest "$manifest" --submission-file "$submission" \
  --include-structured-metrics \
  2>&1 | tee "$run_dir/train.log"
ended=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf 'started_utc: %s\nended_utc: %s\n' "$started" "$ended" > "$run_dir/summary.md"

# RESULT_JSON and structured_metrics are evaluator-owned. Preserve both the full
# result and the bounded per-event history without changing evaluator behavior.
python - "$run_dir/train.log" "$run_dir/result.json" "$run_dir/metrics.jsonl" <<'PY'
import json
import pathlib
import sys

lines = pathlib.Path(sys.argv[1]).read_text().splitlines()
payload = next((line[len("RESULT_JSON="):] for line in lines if line.startswith("RESULT_JSON=")), None)
if payload is None:
    raise SystemExit("evaluator did not emit RESULT_JSON")
result = json.loads(payload)
pathlib.Path(sys.argv[2]).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
with pathlib.Path(sys.argv[3]).open("w") as stream:
    for record in result.get("structured_metrics", []):
        stream.write(json.dumps(record, sort_keys=True) + "\n")
PY
