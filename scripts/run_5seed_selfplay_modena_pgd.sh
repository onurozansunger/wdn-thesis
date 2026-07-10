#!/usr/bin/env bash
# Part 2 Phase 1a: self-play with PGD inner-loop refinement of the
# attacker's perturbation (5-seed, Modena, MoE attacker).
#
# Builds on the retention=5 + lr=1e-4 config that solved catastrophic
# forgetting; adds --pgd_steps 3 --pgd_alpha 0.25 to make the attacker
# stealthier.  Expected ~40min/seed (2.8x baseline due to extra inner
# defender forward passes), ~3.5h total.
set -u
ROOT="/Users/ozanbabac5/wdn_thesis"; cd "$ROOT"
LOG="/tmp/wdn_queue"; mkdir -p "$LOG"
DEF_CKPT="runs/temporal_moe/20260522_204723/best_model.pt"
for s in $(seq 1 5); do
    echo "[sp5pgd] $(date +%H:%M:%S) start seed$s"
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
        --pgd_steps 3 --pgd_alpha 0.25 \
        --curriculum --curriculum_threshold 0.85 \
        --epsilon_p_max 8.0 --k_p_max 15 \
        --attacker_moe --num_attackers 4 \
        --lambda_diversity 0.1 --lambda_atk_balance 0.05 \
        --seed $s \
        > "$LOG/sp5pgd_modena_seed$s.log" 2>&1
    echo "[sp5pgd] $(date +%H:%M:%S) done seed$s"
done
echo "[sp5pgd] ALL DONE"
