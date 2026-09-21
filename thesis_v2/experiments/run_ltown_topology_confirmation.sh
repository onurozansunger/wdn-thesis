#!/usr/bin/env bash
# Confirm the topology contribution on all L-Town data seeds.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${WDN_PYTHON:-/opt/miniconda3/bin/python}"
OUTPUT_ROOT="thesis_v2/outputs/runs/ltown_topology_confirmation"
LOG_DIR="$OUTPUT_ROOT/logs"

cd "$ROOT"
mkdir -p "$LOG_DIR"
export PYTHONPATH=src
export MPLCONFIGDIR=/tmp/wdn-thesis-mpl
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3

run_one() {
  local data_seed="$1"
  local model_seed="$2"
  local marker="$LOG_DIR/d${data_seed}_m${model_seed}"
  if [[ -f "${marker}.done" ]]; then
    echo "skip L-Town no-topology d${data_seed} m${model_seed}"
    return
  fi
  echo "start L-Town no-topology d${data_seed} m${model_seed}"
  "$PYTHON_BIN" -u -m wdn.train_temporal_moe \
    --data_dir "data/thesis_v2/ltown_episode_seed${data_seed}" \
    --gnn_type GraphSAGE \
    --hidden_dim 64 \
    --router_hidden_dim 24 \
    --window_size 6 \
    --num_experts 1 \
    --no_topology \
    --epochs 30 \
    --batch_size 8 \
    --lr 0.001 \
    --norm_mode per_node \
    --lambda_physics 0 \
    --seed "$model_seed" \
    --output_root "$OUTPUT_ROOT" \
    --run_tag "confirm_ltown_topology_none_d${data_seed}_m${model_seed}" \
    > "${marker}.log" 2>&1
  touch "${marker}.done"
  echo "done L-Town no-topology d${data_seed} m${model_seed}"
}

export -f run_one
export PYTHON_BIN OUTPUT_ROOT LOG_DIR
for data_seed in 101 202 303; do
  for model_seed in 1 2 3; do
    echo "$data_seed $model_seed"
  done
done | xargs -P 3 -n 2 bash -c 'run_one "$0" "$1"'

"$PYTHON_BIN" -u thesis_v2/experiments/evaluate_screening_details.py \
  --run-root "$OUTPUT_ROOT" \
  --device cpu

echo "all L-Town topology-confirmation runs complete"
