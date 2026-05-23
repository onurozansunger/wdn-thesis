#!/usr/bin/env bash
set -u
ROOT="/Users/ozanbabac5/wdn_thesis"; cd "$ROOT"
LOG="/tmp/wdn_queue"; mkdir -p "$LOG"
for rw in 2.5 4.0; do
    echo "[rw] $(date +%H:%M:%S) start replay_weight=$rw"
    PYTHONUNBUFFERED=1 python3 -u -m wdn.train_temporal_moe \
        --data_dir data/temporal_moe_modena \
        --gnn_type GraphSAGE --hidden_dim 64 --router_hidden_dim 24 \
        --num_experts 6 --window_size 6 --epochs 60 --batch_size 8 \
        --lr 0.001 --seed 1 --replay_weight $rw \
        > "$LOG/replay_rw${rw}.log" 2>&1
    echo "[rw] $(date +%H:%M:%S) done replay_weight=$rw"
done
echo "[rw] ALL DONE"
