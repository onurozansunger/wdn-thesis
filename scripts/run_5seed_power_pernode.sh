#!/usr/bin/env bash
# Does the per-node normalisation replay fix generalise to the power grid?
# 5-seed IEEE 118-bus with --norm_mode per_node (baseline was global norm).
set -u
ROOT="/Users/ozanbabac5/wdn_thesis"; cd "$ROOT"
LOG="/tmp/wdn_queue"; mkdir -p "$LOG"
for s in 1 2 3 4 5; do
    echo "[pwrpn] $(date +%H:%M:%S) start seed$s"
    PYTHONUNBUFFERED=1 python3 -u -m wdn.train_temporal_moe \
        --data_dir data/temporal_moe_power --gnn_type GraphSAGE \
        --hidden_dim 64 --router_hidden_dim 24 --num_experts 6 --window_size 6 \
        --epochs 60 --batch_size 8 --lr 0.001 --seed $s --norm_mode per_node \
        > "$LOG/pwrpn_s$s.log" 2>&1
    echo "[pwrpn] $(date +%H:%M:%S) done seed$s"
done
echo "[pwrpn] ALL DONE"
