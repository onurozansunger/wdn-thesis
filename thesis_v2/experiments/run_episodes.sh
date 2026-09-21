#!/usr/bin/env bash
# The first honest test of the temporal components.
#
# Every earlier ablation ran on data where the attack family and the
# compromised sensors were redrawn every snapshot, so a window held four
# or five unrelated attacks and a "slow drift" never drifted on any one
# sensor. The recurrence and the router were being asked to exploit
# structure the generator had removed. This runs the same four
# configurations on episode data, where 71% of windows sit inside a
# single attack.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH=src
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
LOG="$ROOT/thesis_v2/outputs/runs/episode_ablation_logs"
mkdir -p "$LOG"
C="--gnn_type GraphSAGE --hidden_dim 64 --router_hidden_dim 24 --epochs 60 \
   --batch_size 8 --lr 0.001 --norm_mode per_node --data_dir data/ep_hard_modena"
job() {
  local cfg=$1 s=$2
  local t="${cfg}_s${s}"; [ -f "$LOG/$t.done" ] && return
  local X
  case $cfg in
    final)      X="--window_size 6" ;;
    notopo)     X="--window_size 6 --no_topology" ;;
    notemporal) X="--window_size 1" ;;
    nomixture)  X="--window_size 6 --num_experts 1" ;;
  esac
  python3 -u -m wdn.train_temporal_moe $C $X --seed $s > "$LOG/$t.log" 2>&1 \
      && touch "$LOG/$t.done"
}
export -f job; export LOG C ROOT
for cfg in final notopo notemporal nomixture; do
  for s in 1 2 3 4 5 6 7 8 9 10; do echo "$cfg $s"; done
done | xargs -P 4 -n 2 bash -c 'job "$0" "$1"'
echo "[ep] ALL DONE $(date +%H:%M:%S)"
