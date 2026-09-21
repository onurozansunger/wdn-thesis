#!/usr/bin/env bash
# Confirm the spatial-vs-temporal comparison on new data seeds and L-Town.
# Screening data seed 101 is intentionally excluded for Modena; it is already
# present in modena_screening and will be combined at report time.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${WDN_PYTHON:-/opt/miniconda3/bin/python}"
OUTPUT_ROOT="thesis_v2/outputs/runs/spatiotemporal_confirmation"
LOG_DIR="$OUTPUT_ROOT/logs"

cd "$ROOT"
mkdir -p "$LOG_DIR"
export PYTHONPATH=src
export MPLCONFIGDIR=/tmp/wdn-thesis-mpl
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3

run_one() {
  local network="$1"
  local data_seed="$2"
  local architecture="$3"
  local model_seed="$4"
  local data_dir="data/thesis_v2/${network}_episode_seed${data_seed}"
  local marker="$LOG_DIR/${network}_d${data_seed}_${architecture}_m${model_seed}"
  if [[ -f "${marker}.done" ]]; then
    echo "skip ${network} d${data_seed} ${architecture} m${model_seed}"
    return
  fi

  local architecture_args=()
  case "$architecture" in
    spatial) architecture_args=(--window_size 1 --num_experts 1) ;;
    temporal) architecture_args=(--window_size 6 --num_experts 1) ;;
    *) echo "unknown architecture: $architecture" >&2; return 2 ;;
  esac

  echo "start ${network} d${data_seed} ${architecture} m${model_seed}"
  "$PYTHON_BIN" -u -m wdn.train_temporal_moe \
    --data_dir "$data_dir" \
    --gnn_type GraphSAGE \
    --hidden_dim 64 \
    --router_hidden_dim 24 \
    --epochs 30 \
    --batch_size 8 \
    --lr 0.001 \
    --norm_mode per_node \
    --lambda_physics 0 \
    --seed "$model_seed" \
    --output_root "$OUTPUT_ROOT" \
    --run_tag "confirm_${network}_d${data_seed}_${architecture}_m${model_seed}" \
    "${architecture_args[@]}" \
    > "${marker}.log" 2>&1
  touch "${marker}.done"
  echo "done ${network} d${data_seed} ${architecture} m${model_seed}"
}

export -f run_one
export PYTHON_BIN OUTPUT_ROOT LOG_DIR

{
  for data_seed in 202 303; do
    for architecture in spatial temporal; do
      for model_seed in 1 2 3; do
        echo "modena $data_seed $architecture $model_seed"
      done
    done
  done
  for data_seed in 101 202 303; do
    for architecture in spatial temporal; do
      for model_seed in 1 2 3; do
        echo "ltown $data_seed $architecture $model_seed"
      done
    done
  done
} | xargs -P 3 -n 4 bash -c 'run_one "$0" "$1" "$2" "$3"'

echo "all spatio-temporal confirmation runs complete"
