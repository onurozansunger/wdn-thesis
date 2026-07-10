#!/usr/bin/env bash
# Part 2 Phase 1c (aggressive): heavier retention (lambda=5.0) + slower
# defender (lr=1e-4) to fully recover hand-crafted F1 to >= 0.83.
# Twin of run_5seed_selfplay_modena_retention.sh with two flags changed.
set -u
ROOT="/Users/ozanbabac5/wdn_thesis"; cd "$ROOT"
LOG="/tmp/wdn_queue"; mkdir -p "$LOG"
DEF_CKPT="runs/temporal_moe/20260522_204723/best_model.pt"
for s in $(seq 1 5); do
    echo "[sp5r5] $(date +%H:%M:%S) start seed$s"
    PYTHONUNBUFFERED=1 python3 -u -m wdn.train_selfplay \
        --data_dir data/temporal_moe_modena \
        --defender_ckpt "$DEF_CKPT" \
        --gnn_type GraphSAGE --hidden_dim 64 --router_hidden_dim 24 \
        --num_experts 6 --window_size 6 \
        --epochs 30 --batch_size 8 \
        --attacker_lr 1e-3 --defender_lr 1e-4 \
        --attacker_steps 3 --defender_steps 1 \
        --epsilon_p 5.0 --epsilon_q 0.05 --k_p 10 --k_q 4 \
        --lambda_recon 5.0 --lambda_budget 0.01 --lambda_physics 0.05 \
        --lambda_retention 5.0 \
        --curriculum --curriculum_threshold 0.85 \
        --epsilon_p_max 8.0 --k_p_max 15 \
        --attacker_moe --num_attackers 4 \
        --lambda_diversity 0.1 --lambda_atk_balance 0.05 \
        --seed $s \
        > "$LOG/sp5r5_modena_seed$s.log" 2>&1
    echo "[sp5r5] $(date +%H:%M:%S) done seed$s"
done
echo "[sp5r5] ALL DONE"
