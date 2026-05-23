#!/usr/bin/env bash
# 10-seed Modena temporal-MoE with the Part-1 router upgrades:
# small classifier (router_hidden=24), bigger experts (hidden=64),
# confidence-gated rerouting, direct expert supervision.
set -u
ROOT="/Users/ozanbabac5/wdn_thesis"; cd "$ROOT"
LOG="/tmp/wdn_queue"; mkdir -p "$LOG"
for s in $(seq 1 10); do
    echo "[10s] $(date +%H:%M:%S) start seed$s"
    PYTHONUNBUFFERED=1 python3 -u -m wdn.train_temporal_moe \
        --data_dir data/temporal_moe_modena \
        --gnn_type GraphSAGE --hidden_dim 64 --router_hidden_dim 24 \
        --num_experts 6 --window_size 6 --epochs 60 --batch_size 8 \
        --lr 0.001 --seed $s \
        > "$LOG/p1_modena_seed$s.log" 2>&1
    echo "[10s] $(date +%H:%M:%S) done seed$s"
done
echo "[10s] ALL DONE"
