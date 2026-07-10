#!/usr/bin/env bash
# 10-seed Modena temporal-MoE with replay_weight=2.5 to dokument the
# Pareto trade-off: rescuing replay tanks overall F1.
# Twin of scripts/run_10seed.sh but with --replay_weight 2.5.
set -u
ROOT="/Users/ozanbabac5/wdn_thesis"; cd "$ROOT"
LOG="/tmp/wdn_queue"; mkdir -p "$LOG"
for s in $(seq 1 10); do
    echo "[10s-rw25] $(date +%H:%M:%S) start seed$s"
    PYTHONUNBUFFERED=1 python3 -u -m wdn.train_temporal_moe \
        --data_dir data/temporal_moe_modena \
        --gnn_type GraphSAGE --hidden_dim 64 --router_hidden_dim 24 \
        --num_experts 6 --window_size 6 --epochs 60 --batch_size 8 \
        --lr 0.001 --seed $s --replay_weight 2.5 \
        > "$LOG/p1_modena_rw25_seed$s.log" 2>&1
    echo "[10s-rw25] $(date +%H:%M:%S) done seed$s"
done
echo "[10s-rw25] ALL DONE"
