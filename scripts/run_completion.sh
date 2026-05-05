#!/usr/bin/env bash
# Fill the gaps so all three datasets get the same treatment:
#   - Net3 temporal MoE baseline (no pattern features)
#   - Net1 GNN backbone comparison (4 backbones)
#   - Net3 GNN backbone comparison (4 backbones)
set -u

LOG="/tmp/wdn_queue"
mkdir -p "$LOG"

run() {
    local name="$1"; shift
    echo "[fill] $(date +%H:%M:%S) start $name"
    PYTHONUNBUFFERED=1 "$@" > "$LOG/$name.log" 2>&1
    echo "[fill] $(date +%H:%M:%S) done  $name (exit $?)"
}

# Net3 baseline (no pattern features)
run "net3_baseline" python3 -u -m wdn.train_temporal_moe \
    --data_dir data/temporal_moe_net3 --gnn_type GraphSAGE \
    --hidden_dim 48 --num_experts 6 --window_size 6 \
    --epochs 60 --batch_size 8 --lr 0.001 \
    --no_pattern_features

# GNN comparison on Net1 + Net3 (spatial multitask)
for net in net1 net3; do
    if [ "$net" = "net1" ]; then
        data_dir="data/moe_net1"
    else
        data_dir="data/temporal_moe_net3"
    fi
    for gnn in GraphSAGE GAT GCN Transformer; do
        run "${net}_spatial_${gnn}" python3 -u -m wdn.train_multitask \
            --data_dir "$data_dir" --gnn_type "$gnn" --epochs 40
    done
done

echo "[fill] $(date +%H:%M:%S) ALL DONE"
