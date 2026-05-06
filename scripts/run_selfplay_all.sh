#!/usr/bin/env bash
# Run pretrained + adversarial fine-tune self-play on all three networks.
# Each run reuses the matching pretrained Temporal MoE checkpoint so the
# defender already knows the hand-crafted attack signatures and the
# adversarial fine-tune adds robustness against the attacker's outputs.
set -u
LOG="/tmp/wdn_queue"; mkdir -p "$LOG"

run() {
    local name="$1"; shift
    echo "[sp] $(date +%H:%M:%S) start $name"
    PYTHONUNBUFFERED=1 "$@" > "$LOG/$name.log" 2>&1
    echo "[sp] $(date +%H:%M:%S) done  $name (exit $?)"
}

# Modena already done (runs/selfplay/20260505_223529).
# Net1 fine-tune. Pretrained Net1 MoE checkpoint:
run "selfplay_net1_ft" python3 -u -m wdn.train_selfplay \
    --data_dir data/moe_net1 \
    --defender_ckpt runs/temporal_moe/20260505_145514/best_model.pt \
    --hidden_dim 32 --epochs 30 --batch_size 8 \
    --epsilon_p 5.0 --k_p 4 \
    --lambda_recon 5.0 --lambda_physics 0.05 \
    --attacker_steps 3 --defender_steps 1 \
    --curriculum --curriculum_threshold 0.85 \
    --epsilon_p_max 8.0 --k_p_max 6

run "selfplay_net3_ft" python3 -u -m wdn.train_selfplay \
    --data_dir data/temporal_moe_net3 \
    --defender_ckpt runs/temporal_moe/20260505_150656/best_model.pt \
    --hidden_dim 48 --epochs 30 --batch_size 8 \
    --epsilon_p 5.0 --k_p 10 \
    --lambda_recon 5.0 --lambda_physics 0.05 \
    --attacker_steps 3 --defender_steps 1 \
    --curriculum --curriculum_threshold 0.85 \
    --epsilon_p_max 8.0 --k_p_max 15

echo "[sp] $(date +%H:%M:%S) ALL DONE"
