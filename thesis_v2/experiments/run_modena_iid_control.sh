#!/usr/bin/env bash
# Negative control for temporal coherence: spatial vs temporal on IID attacks.
# The data seed and every corruption magnitude match the episode screening;
# only family/target persistence is disabled by the dataset config.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${WDN_PYTHON:-/opt/miniconda3/bin/python}"
DATA_DIR="data/thesis_v2/modena_iid_seed101"
OUTPUT_ROOT="thesis_v2/outputs/runs/modena_iid_control"
LOG_DIR="$OUTPUT_ROOT/logs"

cd "$ROOT"
mkdir -p "$LOG_DIR"
export PYTHONPATH=src
export MPLCONFIGDIR=/tmp/wdn-thesis-mpl
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3

run_one() {
  local architecture="$1"
  local seed="$2"
  local marker="$LOG_DIR/${architecture}_m${seed}"
  if [[ -f "${marker}.done" ]]; then
    echo "skip ${architecture} model-seed ${seed}"
    return
  fi

  local architecture_args=()
  case "$architecture" in
    spatial) architecture_args=(--window_size 1 --num_experts 1) ;;
    temporal) architecture_args=(--window_size 6 --num_experts 1) ;;
    *) echo "unknown architecture: $architecture" >&2; return 2 ;;
  esac

  echo "start IID ${architecture} model-seed ${seed}"
  "$PYTHON_BIN" -u -m wdn.train_temporal_moe \
    --data_dir "$DATA_DIR" \
    --gnn_type GraphSAGE \
    --hidden_dim 64 \
    --router_hidden_dim 24 \
    --epochs 30 \
    --batch_size 8 \
    --lr 0.001 \
    --norm_mode per_node \
    --lambda_physics 0 \
    --seed "$seed" \
    --output_root "$OUTPUT_ROOT" \
    --run_tag "iid_${architecture}_d101_m${seed}" \
    "${architecture_args[@]}" \
    > "${marker}.log" 2>&1
  touch "${marker}.done"
  echo "done IID ${architecture} model-seed ${seed}"
}

export -f run_one
export PYTHON_BIN DATA_DIR OUTPUT_ROOT LOG_DIR

for architecture in spatial temporal; do
  for seed in 1 2 3; do
    echo "$architecture $seed"
  done
done | xargs -P 3 -n 2 bash -c 'run_one "$0" "$1"'

echo "all Modena IID control runs complete"
