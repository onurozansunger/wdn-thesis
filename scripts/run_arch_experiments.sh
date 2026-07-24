#!/usr/bin/env bash
# Experiments for the new (cascade) architecture and the replay/normalisation
# question raised by the supervisors.
#
# A) lambda_expert sweep — does training each expert to be a competent
#    STANDALONE detector lift the ceiling for hard/cascade routing?
#    (With the mixture-trained baseline the oracle ceiling is only 0.643,
#     far below the soft mixture's 0.828, so hard routing cannot win.)
# B) per-node normalisation — supervisors' hypothesis that water values sit
#    too close together for replay to be visible.
set -u
ROOT="/Users/ozanbabac5/wdn_thesis"; cd "$ROOT"
LOG="/tmp/wdn_queue"; mkdir -p "$LOG"
COMMON="--data_dir data/temporal_moe_modena --gnn_type GraphSAGE \
  --hidden_dim 64 --router_hidden_dim 24 --num_experts 6 --window_size 6 \
  --epochs 60 --batch_size 8 --lr 0.001"

run () {  # name, extra args
  local name="$1"; shift
  echo "[arch] $(date +%H:%M:%S) start $name"
  PYTHONUNBUFFERED=1 python3 -u -m wdn.train_temporal_moe $COMMON "$@" \
      > "$LOG/arch_${name}.log" 2>&1
  echo "[arch] $(date +%H:%M:%S) done $name -> $(grep -oE 'runs/temporal_moe/[0-9_]+' "$LOG/arch_${name}.log" | tail -1)"
}

# A) expert-supervision sweep (seed 1)
run le0.5_s1  --seed 1 --lambda_expert 0.5
run le2_s1    --seed 1 --lambda_expert 2.0
run le4_s1    --seed 1 --lambda_expert 4.0
run le8_s1    --seed 1 --lambda_expert 8.0

# B) per-node normalisation (seed 1), at baseline and best-guess expert weight
run norm_s1     --seed 1 --norm_mode per_node
run norm_le4_s1 --seed 1 --norm_mode per_node --lambda_expert 4.0

echo "[arch] ALL DONE"
