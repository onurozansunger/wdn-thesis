#!/usr/bin/env bash
# Three-seed Modena self-play fine-tune for error bars.
set -u
LOG="/tmp/wdn_queue"; mkdir -p "$LOG"

run() {
    local name="$1"; shift
    echo "[ms] $(date +%H:%M:%S) start $name"
    PYTHONUNBUFFERED=1 "$@" > "$LOG/$name.log" 2>&1
    echo "[ms] $(date +%H:%M:%S) done  $name (exit $?)"
}

for s in 1 2 3; do
    run "selfplay_modena_seed$s" python3 -u -m wdn.train_selfplay \
        --data_dir data/temporal_moe_modena \
        --defender_ckpt runs/temporal_moe/20260505_144409/best_model.pt \
        --epochs 30 --batch_size 8 \
        --epsilon_p 5.0 --k_p 15 \
        --lambda_recon 5.0 --lambda_physics 0.05 \
        --attacker_steps 3 --defender_steps 1 \
        --curriculum --curriculum_threshold 0.85 \
        --seed $s
done

echo "[ms] $(date +%H:%M:%S) ALL DONE"
