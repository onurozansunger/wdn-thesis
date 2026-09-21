#!/usr/bin/env bash
# Graph-structure control: temporal node-independent model vs temporal GNN.
# The latter already exists in modena_screening; this script trains only the
# matched no-message-passing counterpart on the same data and model seeds.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${WDN_PYTHON:-/opt/miniconda3/bin/python}"
DATA_DIR="data/thesis_v2/modena_episode_seed101"
OUTPUT_ROOT="thesis_v2/outputs/runs/modena_topology_control"
LOG_DIR="$OUTPUT_ROOT/logs"

cd "$ROOT"
mkdir -p "$LOG_DIR"
export PYTHONPATH=src
export MPLCONFIGDIR=/tmp/wdn-thesis-mpl
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3

run_one() {
  local seed="$1"
  local marker="$LOG_DIR/no_topology_m${seed}"
  if [[ -f "${marker}.done" ]]; then
    echo "skip no-topology model-seed ${seed}"
    return
  fi
  echo "start no-topology model-seed ${seed}"
  "$PYTHON_BIN" -u -m wdn.train_temporal_moe \
    --data_dir "$DATA_DIR" \
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
    --seed "$seed" \
    --output_root "$OUTPUT_ROOT" \
    --run_tag "topology_none_d101_m${seed}" \
    > "${marker}.log" 2>&1
  touch "${marker}.done"
  echo "done no-topology model-seed ${seed}"
}

export -f run_one
export PYTHON_BIN DATA_DIR OUTPUT_ROOT LOG_DIR
printf '%s\n' 1 2 3 | xargs -P 3 -n 1 bash -c 'run_one "$0"'
echo "all Modena topology-control runs complete"
