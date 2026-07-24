#!/usr/bin/env bash
# Final Part-1 configuration, 5 seeds:
#   per-node normalisation  (rescues replay: AUROC 0.465 -> 0.810)
#   + validation-calibrated decision threshold (built into the train script)
#
# Also re-runs the global-norm baseline with the same calibration so the
# comparison is apples-to-apples.
set -u
ROOT="/Users/ozanbabac5/wdn_thesis"; cd "$ROOT"
LOG="/tmp/wdn_queue"; mkdir -p "$LOG"
COMMON="--data_dir data/temporal_moe_modena --gnn_type GraphSAGE \
  --hidden_dim 64 --router_hidden_dim 24 --num_experts 6 --window_size 6 \
  --epochs 60 --batch_size 8 --lr 0.001"

for s in 1 2 3 4 5; do
    echo "[final] $(date +%H:%M:%S) start pernode seed$s"
    PYTHONUNBUFFERED=1 python3 -u -m wdn.train_temporal_moe $COMMON \
        --seed $s --norm_mode per_node \
        > "$LOG/final_pernode_s$s.log" 2>&1
    echo "[final] $(date +%H:%M:%S) done pernode seed$s"
done
for s in 1 2 3 4 5; do
    echo "[final] $(date +%H:%M:%S) start global seed$s"
    PYTHONUNBUFFERED=1 python3 -u -m wdn.train_temporal_moe $COMMON \
        --seed $s --norm_mode global \
        > "$LOG/final_global_s$s.log" 2>&1
    echo "[final] $(date +%H:%M:%S) done global seed$s"
done
echo "[final] ALL DONE"
