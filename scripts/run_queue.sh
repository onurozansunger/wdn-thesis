#!/usr/bin/env bash
# Sequential training queue. Waits for the running Modena MoE process to
# finish (so MPS isn't oversubscribed), then launches Net1 + Net3 MoE runs
# with the new pattern-detection features, followed by GNN-backbone
# comparison runs on Modena's spatial MultiTaskGNN.
set -u

LOGDIR="/tmp/wdn_queue"
mkdir -p "$LOGDIR"

wait_for_modena() {
    while pgrep -f "train_temporal_moe.*temporal_moe_modena" > /dev/null; do
        sleep 30
    done
}

run_step() {
    local name="$1"; shift
    echo "[queue] $(date +%H:%M:%S) starting $name"
    PYTHONUNBUFFERED=1 "$@" > "$LOGDIR/$name.log" 2>&1
    echo "[queue] $(date +%H:%M:%S) finished $name (exit $?)"
}

echo "[queue] $(date +%H:%M:%S) waiting for Modena MoE to finish"
wait_for_modena
echo "[queue] $(date +%H:%M:%S) Modena MoE done, running queue"

# --- Net1 MoE with new features ---
run_step "net1_moe" python3 -u -m wdn.train_temporal_moe \
    --data_dir data/moe_net1 \
    --gnn_type GraphSAGE \
    --hidden_dim 32 \
    --num_experts 6 \
    --window_size 6 \
    --epochs 60 \
    --batch_size 8 \
    --lr 0.001

# --- Net3 MoE (new dataset) ---
run_step "net3_moe" python3 -u -m wdn.train_temporal_moe \
    --data_dir data/temporal_moe_net3 \
    --gnn_type GraphSAGE \
    --hidden_dim 48 \
    --num_experts 6 \
    --window_size 6 \
    --epochs 60 \
    --batch_size 8 \
    --lr 0.001

# --- GNN backbone comparison on Modena spatial multitask ---
for gnn in GAT GCN Transformer GraphSAGE; do
    run_step "modena_spatial_$gnn" python3 -u -m wdn.train_multitask \
        --data_dir data/modena_attacks \
        --gnn_type "$gnn" \
        --hidden_dim 64 \
        --epochs 40 \
        --batch_size 8 \
        --lr 0.001
done

echo "[queue] $(date +%H:%M:%S) ALL DONE"
