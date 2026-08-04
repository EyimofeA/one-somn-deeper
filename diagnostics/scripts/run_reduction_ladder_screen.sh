#!/usr/bin/env bash
set -euo pipefail

root=${1:?usage: $0 <diagnostics-root> <python> <run-root>}
python_bin=${2:?usage: $0 <diagnostics-root> <python> <run-root>}
run_root=${3:?usage: $0 <diagnostics-root> <python> <run-root>}

cd "$root"
for regime in b0_copy b1_subtract b2_small_quotient b3_square b4_broad; do
  for arch in baseline deep_control recurrent_k8_input_context; do
    run_dir="$run_root/$regime/$arch/seed0"
    data_dir="data/generated/reduction_ladder/$regime"
    config="configs/mod_fixed_n_${arch}.yaml"
    mkdir -p "$run_dir"
    "$python_bin" train.py "$config" \
      --override "data.train=$data_dir/train.jsonl" \
      --override "data.val=$data_dir/val_iid.jsonl" \
      --override "out_dir=$run_dir" \
      --override "optim.total_steps=2000" \
      --override "log_every=100" \
      --override "eval_every=200" \
      --override "early_stop_patience=null"
    "$python_bin" evaluate.py "$run_dir" --data "$data_dir" \
      --splits train heldout_u --out "$run_dir/eval_report.json"
  done
done
