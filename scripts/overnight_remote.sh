#!/usr/bin/env bash
# Runs on twoA6000 after the delayed local launcher has copied this workspace.
set -u -o pipefail

source_root="$1"
run_id="$2"
deadline_epoch="$3"
python_bin="$HOME/one-layer-deeper/.venv/bin/python"
diag="$source_root/diagnostics"
run_root="$diag/runs/overnight_$run_id"
report_root="$source_root/reports"

mkdir -p "$run_root/logs" "$run_root/analysis" "$report_root"
date -Is > "$run_root/started_at.txt"
cp "$source_root/source_commit.txt" "$run_root/source_commit.txt" 2>/dev/null || true
nvidia-smi > "$run_root/nvidia_smi_start.txt" 2>&1 || true
find "$diag/runs" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort > "$run_root/existing_run_dirs.txt" 2>/dev/null || true

if [[ ! -x "$python_bin" ]]; then
  printf 'Missing expected Python environment: %s\n' "$python_bin" | tee "$run_root/FAILED.txt"
  exit 1
fi

cd "$diag"
"$python_bin" audit_mod.py > "$run_root/logs/task_b_audit.log" 2>&1
audit_status=$?
cp "$run_root/logs/task_b_audit.log" "$report_root/task_b_pipeline_audit.md"
if [[ $audit_status -ne 0 ]]; then
  printf 'Task B audit failed; baseline was not launched.\n' | tee "$run_root/FAILED.txt"
  exit "$audit_status"
fi

idle_gpus=()
while IFS= read -r gpu; do
  if ! nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]'; then
    idle_gpus+=("$gpu")
  fi
done < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
printf '%s\n' "${idle_gpus[@]}" > "$run_root/idle_gpus_at_launch.txt"

remaining() { [[ "$(date +%s)" -lt "$deadline_epoch" ]]; }

run_task_a_chain() {
  local gpu="$1"
  local seed
  for seed in 0 1 2; do
    remaining || break
    local out="$run_root/task_a_carry_shuffled_seed$seed"
    local cmd="CUDA_VISIBLE_DEVICES=$gpu $python_bin train_aux_ablation.py --condition carry_shuffled --seed $seed --device cuda --out-root $run_root"
    {
      date -Is
      printf '%s\n' "$cmd"
      printf 'gpu=%s source_commit=' "$gpu"; cat "$run_root/source_commit.txt"
      eval "$cmd"
      printf 'exit_status=%s\n' "$?"
      date -Is
    } > "$run_root/logs/task_a_carry_shuffled_seed$seed.log" 2>&1
    mv "$run_root/carry_shuffled_seed$seed" "$out" 2>/dev/null || true
  done
}

run_task_b_chain() {
  local gpu="$1"
  local seed
  for seed in 0 1 2; do
    remaining || break
    local out="$run_root/task_b_seed$seed"
    mkdir -p "$out"
    local cmd="CUDA_VISIBLE_DEVICES=$gpu $python_bin train.py configs/mod.yaml --override device=cuda --override seed=$seed --override out_dir=$out"
    {
      date -Is
      printf '%s\n' "$cmd"
      printf 'gpu=%s source_commit=' "$gpu"; cat "$run_root/source_commit.txt"
      eval "$cmd"
      train_status=$?
      if [[ $train_status -eq 0 ]]; then
        "$python_bin" evaluate.py "$out" --data data/generated/mod --splits val_iid heldout_u heldout_modulus heldout_factor length12 length13 hard
      fi
      if [[ $seed -eq 0 && $train_status -eq 0 ]]; then
        "$python_bin" analysis_task_b.py --run-dir "$out" --out-dir "$run_root/analysis/task_b_seed0"
        cp "$run_root/analysis/task_b_seed0/task_b_analysis.json" "$report_root/task_b_baseline_diagnostics.json" 2>/dev/null || true
      fi
      printf 'exit_status=%s\n' "$train_status"
      date -Is
    } > "$run_root/logs/task_b_seed$seed.log" 2>&1
  done
}

if [[ ${#idle_gpus[@]} -eq 0 ]]; then
  printf 'No idle GPUs; audit completed but no training launched.\n' > "$run_root/NO_IDLE_GPU.txt"
elif [[ ${#idle_gpus[@]} -eq 1 ]]; then
  run_task_b_chain "${idle_gpus[0]}" &
  task_b_pid=$!
else
  run_task_b_chain "${idle_gpus[0]}" &
  task_b_pid=$!
  run_task_a_chain "${idle_gpus[1]}" &
  task_a_pid=$!
fi

while remaining; do
  sleep 60
done

nvidia-smi > "$run_root/nvidia_smi_deadline.txt" 2>&1 || true
{
  printf '# Unattended overnight handoff\n\n'
  printf 'Run root: `%s`\n\n' "$run_root"
  printf 'Deadline reached: '; date -Is
  printf '\n## Logs and run directories\n\n'
  find "$run_root" -maxdepth 2 -type f \( -name '*.log' -o -name 'eval_report.json' -o -name 'run_config.json' -o -name 'config_used.yaml' \) -printf -- '- `%p`\n' | sort
  printf '\n## Active GPU state at deadline\n\n```\n'
  cat "$run_root/nvidia_smi_deadline.txt"
  printf '```\n'
  printf '\nJobs are intentionally not killed at the deadline.\n'
} > "$source_root/research/overnight_handoff.md"
