#!/usr/bin/env bash
# Traffic per-node normalisation — completes the 3-domain per_node sweep.
set -u
ROOT="/Users/ozanbabac5/wdn_thesis"; cd "$ROOT"
LOG="/tmp/wdn_queue"; mkdir -p "$LOG"
for s in 1 2 3 4 5; do
    echo "[trfpn] $(date +%H:%M:%S) start seed$s"
    PYTHONUNBUFFERED=1 python3 -u -m wdn.train_temporal_moe \
        --data_dir data/temporal_moe_traffic --gnn_type GraphSAGE \
        --hidden_dim 64 --router_hidden_dim 24 --num_experts 6 --window_size 6 \
        --epochs 60 --batch_size 8 --lr 0.001 --seed $s --norm_mode per_node \
        > "$LOG/trfpn_s$s.log" 2>&1
    echo "[trfpn] $(date +%H:%M:%S) done seed$s"
done
echo "[trfpn] ALL DONE"
