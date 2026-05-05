#!/usr/bin/env bash
# GNN backbone comparison on Modena spatial multitask. Trains the same
# MultiTaskGNN architecture with different message-passing layers so we
# can defend the GraphSAGE choice with an empirical ablation.
set -u

LOGDIR="/tmp/wdn_queue"
mkdir -p "$LOGDIR"

run_one() {
    local gnn="$1"
    local name="modena_spatial_$gnn"
    echo "[gnn-cmp] $(date +%H:%M:%S) starting $gnn"
    PYTHONUNBUFFERED=1 python3 -u -m wdn.train_multitask \
        --data_dir data/modena_attacks \
        --gnn_type "$gnn" \
        --epochs 40 \
        > "$LOGDIR/$name.log" 2>&1
    echo "[gnn-cmp] $(date +%H:%M:%S) finished $gnn (exit $?)"
}

for gnn in GraphSAGE GAT GCN Transformer; do
    run_one "$gnn"
done

echo "[gnn-cmp] $(date +%H:%M:%S) ALL DONE"
