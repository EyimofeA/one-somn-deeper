#!/usr/bin/env bash
# Reproducible serial launcher for Task B Phase 1.  Run from diagnostics/.
set -uo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
SOURCE_COMMIT=${SOURCE_COMMIT:-294b7b7}
DATA_DIR=${DATA_DIR:-data/generated/mod_fixed_n_1349}
RUN_ROOT=${RUN_ROOT:-runs}
LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$LOG_DIR"

for mode in input_context shuffled_context; do
  for seed in 0 1 2; do
    out="$RUN_ROOT/task_b_fixed_n_recurrent_k8_${mode}/seed${seed}"
    mkdir -p "$out"
    {
      echo "source_commit=$SOURCE_COMMIT"
      echo "started_utc=$(date -u +%FT%TZ)"
      echo "mode=$mode seed=$seed"
      echo "data=$DATA_DIR/train.jsonl,$DATA_DIR/val_iid.jsonl,heldout_u.jsonl"
      echo "command=train.py configs/mod_fixed_n_recurrent_k8_${mode}.yaml --override seed=$seed --override out_dir=$out"
    } > "$out/run_metadata.txt"

    set -o pipefail
    CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" train.py "configs/mod_fixed_n_recurrent_k8_${mode}.yaml" \
      --override "seed=$seed" --override "out_dir=$out" 2>&1 | tee "$out/train.log"
    train_status=${PIPESTATUS[0]}
    echo "train_exit=$train_status ended_utc=$(date -u +%FT%TZ)" >> "$out/run_metadata.txt"

    if [ "$train_status" -eq 0 ]; then
      CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" evaluate.py "$out" --data "$DATA_DIR" \
        --splits val_iid heldout_u --out "$out/eval_report.json" 2>&1 | tee "$out/evaluate.log"
      eval_status=${PIPESTATUS[0]}
      echo "evaluate_exit=$eval_status evaluated_utc=$(date -u +%FT%TZ)" >> "$out/run_metadata.txt"
    fi
  done
done
