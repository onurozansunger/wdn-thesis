#!/usr/bin/env bash
set -u
ROOT="/Users/ozanbabac5/wdn_thesis"
LOG="/tmp/wdn_queue"; mkdir -p "$LOG"
cd "$ROOT"
for s in 1 2 3; do
    echo "[ms3] $(date +%H:%M:%S) start seed$s"
    PYTHONUNBUFFERED=1 python3 -u -m wdn.train_selfplay \
        --data_dir data/temporal_moe_net3 \
        --defender_ckpt runs/temporal_moe/20260505_150656/best_model.pt \
        --epochs 30 --batch_size 8 \
        --epsilon_p 5.0 --k_p 10 \
        --lambda_recon 5.0 --lambda_physics 0.05 \
        --attacker_steps 3 --defender_steps 1 \
        --curriculum --curriculum_threshold 0.85 \
        --epsilon_p_max 8.0 --k_p_max 15 \
        --seed $s > "$LOG/sp_net3_seed$s.log" 2>&1
    echo "[ms3] $(date +%H:%M:%S) done seed$s"
done
echo "[ms3] ALL DONE"
