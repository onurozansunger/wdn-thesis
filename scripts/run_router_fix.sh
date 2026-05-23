#!/usr/bin/env bash
set -u
ROOT="/Users/ozanbabac5/wdn_thesis"; cd "$ROOT"
LOG="/tmp/wdn_queue"; mkdir -p "$LOG"
for net in net3 modena; do
    echo "[rf] $(date +%H:%M:%S) start $net"
    PYTHONUNBUFFERED=1 python3 -u -m wdn.train_temporal_moe \
        --data_dir data/temporal_moe_$net \
        --gnn_type GraphSAGE --hidden_dim 48 --num_experts 6 \
        --window_size 6 --epochs 60 --batch_size 8 --lr 0.001 \
        > "$LOG/router_fix_$net.log" 2>&1
    echo "[rf] $(date +%H:%M:%S) done $net"
done
echo "[rf] ALL DONE"
