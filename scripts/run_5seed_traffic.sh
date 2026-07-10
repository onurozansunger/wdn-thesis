#!/usr/bin/env bash
# Cross-domain full run: 5-seed synthetic road-traffic sensor network
# with the same temporal-MoE config as the water and power runs.
set -u
ROOT="/Users/ozanbabac5/wdn_thesis"; cd "$ROOT"
LOG="/tmp/wdn_queue"; mkdir -p "$LOG"
for s in $(seq 1 5); do
    echo "[trf5] $(date +%H:%M:%S) start seed$s"
    PYTHONUNBUFFERED=1 python3 -u -m wdn.train_temporal_moe \
        --data_dir data/temporal_moe_traffic \
        --gnn_type GraphSAGE --hidden_dim 64 --router_hidden_dim 24 \
        --num_experts 6 --window_size 6 --epochs 60 --batch_size 8 \
        --lr 0.001 --seed $s \
        > "$LOG/trf5_seed$s.log" 2>&1
    echo "[trf5] $(date +%H:%M:%S) done seed$s"
done
echo "[trf5] ALL DONE"
