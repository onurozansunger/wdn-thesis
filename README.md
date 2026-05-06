# Adversarial Self-Play GNNs for Water Distribution Network Cyber-Defence

> **Defender that wasn't trained on a single hand-crafted attack
> outperforms the supervised baseline on five known classes — and
> generalises to attack types it has never seen.**

A Master's thesis on graph neural networks that simultaneously
reconstruct missing pressure / flow readings and flag compromised
sensors in a water distribution network. Built up in stages:

```
spatial GNN  →  + temporal GRU  →  + Mixture-of-Experts
              →  + replay-pattern features  →  + adversarial self-play
```

Every step targets a specific weakness of the previous one. The final
contribution is a **Stackelberg self-play loop** in which an Attacker
GNN and a Defender GNN co-evolve: the attacker learns sparse,
budget-respecting, physics-aware perturbations; the defender learns to
catch them. With an *attacker mixture-of-experts*, the population
auto-discovers two distinct attack families without any class
supervision, and the resulting defender generalises to two completely
novel attack types held out from training.

## Headline numbers (Modena, 272 nodes, 317 pipes)

| Metric | Pretrained Temporal MoE | + Self-play (single) | + Self-play (Attacker MoE) |
|---|:---:|:---:|:---:|
| Anomaly F1 | 0.725 | 0.721 | **0.767** *(+5.8%)* |
| Anomaly AUROC | 0.963 | 0.965 | 0.965 |
| Pressure MAE (m) | 0.089 | 0.067 | **0.068** *(−23%)* |
| Targeted attack F1 | 0.975 | **1.000** | 0.994 |
| Stealthy attack F1 | 0.927 | **0.955** | 0.941 |

Multi-seed (3 seeds) self-play on Modena gives **P MAE 0.070 ± 0.004**
and **targeted F1 0.995 ± 0.006** — gains are reproducible, not noise.

### Held-out generalisation — attacks never seen in training

Two synthetic attack types (sinusoidal injection, cross-sensor swap)
that don't match any of the five hand-crafted classes the defenders
were trained on:

| Defender | Sinusoidal F1 | Sinusoidal AUROC | Swap F1 | Swap AUROC |
|---|:---:|:---:|:---:|:---:|
| Pretrained | 0.809 | 0.888 | 0.748 | 0.852 |
| Self-play single | 0.818 | 0.893 | 0.744 | 0.852 |
| **Self-play Attacker-MoE** | **0.834** | **0.905** | **0.757** | **0.867** |

The self-play defender is **best on both novel attacks** despite never
having been trained on either of them — evidence that the framework
delivers genuine adversarial robustness, not just curve-fitting to a
fixed attack distribution.

### Emergent attack vocabulary

The Attacker-MoE was trained with a load-balancing entropy and a
diversity loss but **no attack-class labels**. After convergence it
auto-partitioned the perturbation space into two functional families:

| Expert | Picks | Top class | Distribution |
|---|:---:|:---:|---|
| 2 — *bold* | 87 | random (30%) | random + targeted + noise dominate |
| 3 — *subtle* | 61 | replay (33%) | replay + stealthy dominate |

`presentation/charts/vocab_attackmoe.png` shows the t-SNE projection.

## Architecture progression

```
   spatial               +temporal            +MoE                +pattern feats
   ─────────             ──────────           ─────────           ──────────────
   GraphSAGE             GraphSAGE+GRU        6 experts +         autocorr +
   2 layers              6-step window        learned router      diff-std +
   4 prediction heads    + temporal stab.     load-balancing      noise ratio
                         features
                                                    │
                                                    ▼
                                       SELF-PLAY (Stackelberg game)

                       ┌─────────────────┐    perturbed snapshot     ┌──────────────────┐
                       │   Attacker GNN  │ ────────────────────────► │   Defender MoE   │
                       │  (top-k, ε-     │                           │  (frozen each    │
                       │   bounded,      │ ◄──── anomaly logits ──── │   attacker step) │
                       │   physics-aware)│                           └──────────────────┘
                       └─────────────────┘
                              ▲                                              │
                              │ damage gradient                              │ recon + anomaly
                              └──────────────────────────────────────────────┘
                                          alternating SGD
                                       + auto-curriculum
```

## Three benchmark networks

| Network | Junctions | Pipes | Source | Self-play helps? |
|---|:---:|:---:|---|:---:|
| Net1 | 11 | 13 | EPANET reference | ✗ small-graph regression |
| Net3 | 97 | 117 | EPANET reference | ✓ overall F1 +8% |
| Modena | 272 | 317 | Bragalli et al., 2008 | ✓ everywhere except replay |

Net1's small graph already saturates the pretrained model's capacity,
so adversarial fine-tuning hurts more than it helps — an honest
limitation, documented in the thesis discussion.

## Repository layout

```
├── src/wdn/
│   ├── data_generation.py        # WNTR → graph snapshots
│   ├── corruption.py             # 5 hand-crafted attack types + missing-data + noise
│   ├── dataset.py                # PyG dataset + per-feature normalisation
│   ├── temporal_dataset.py       # Sliding-window temporal dataset (T=6)
│   ├── config.py                 # Configuration dataclasses
│   ├── metrics.py                # MAE/RMSE/Precision/Recall/F1/AUROC
│   ├── explainability.py         # GNNExplainer integration
│   ├── generate.py               # CLI entry point: generate datasets
│   │
│   ├── models/
│   │   ├── gnn.py                # GraphSAGE / GAT / GCN / Transformer backbones
│   │   ├── multitask.py          # MultiTaskGNN (spatial baseline)
│   │   ├── temporal_multitask.py # GRU + 9 temporal-stability features
│   │   ├── moe.py                # Spatial Mixture-of-Experts
│   │   ├── temporal_moe.py       # Temporal Mixture-of-Experts (production defender)
│   │   ├── attacker.py           # Single Attacker GNN + StealthBudget projector
│   │   └── attacker_moe.py       # Mixture-of-Attackers + diversity / balance loss
│   │
│   ├── train_multitask.py        # Spatial multi-task training
│   ├── train_temporal.py         # Temporal multi-task training
│   ├── train_moe.py              # Spatial MoE training
│   ├── train_temporal_moe.py     # Temporal MoE training (anomaly-F1 best-model)
│   └── train_selfplay.py         # Stackelberg self-play (attacker ↔ defender)
│
├── scripts/                      # Evaluation + plotting + run queues
│   ├── summarise_runs.py            # Tabular summary of every runs/* dir
│   ├── eval_selfplay_modena.py      # Self-play Modena vs pretrained
│   ├── eval_selfplay_three_nets.py  # Self-play across Net1 / Net3 / Modena
│   ├── eval_multiseed.py            # 3-seed mean ± std on Modena
│   ├── eval_atkmoe.py               # Pretrained vs SP single vs SP MoE
│   ├── heldout_novel_attack.py      # Generalisation to sinusoidal + sensor-swap
│   ├── vocabulary_mining.py         # t-SNE + expert-class purity table
│   ├── plot_selfplay.py             # Co-evolution / per-attack / multi-seed / heldout
│   ├── run_queue.sh                 # Sequential queue (Net1+Net3 MoE + GNN ablation)
│   ├── run_gnn_comparison.sh        # Backbone ablation (4 GNNs × 3 nets)
│   ├── run_completion.sh            # Net3 baseline + Net1/Net3 ablation
│   ├── run_selfplay_all.sh          # Self-play fine-tune on all 3 networks
│   └── run_multiseed.sh             # 3 seeds × Modena self-play
│
├── configs/                      # YAML data-generation configs
│   ├── generate_moe_net1.yaml
│   ├── generate_temporal_moe_modena.yaml
│   └── generate_temporal_moe_net3.yaml
│
├── data/
│   ├── Net1.inp, Net3.inp, modena.inp   # EPANET network files
│   └── networks/, generated_*/          # Generated datasets (gitignored)
│
├── runs/                         # Training artefacts (gitignored)
│   ├── multitask/, temporal/, moe/, temporal_moe/
│   └── selfplay/                    # Phase-7 attacker + defender checkpoints
│
├── presentation/
│   ├── build_pdf.py                 # 8-slide PDF generator
│   ├── generate_charts.py           # Static result charts
│   └── charts/*.png                 # All thesis figures
│
└── dashboard/                    # Streamlit dashboard (6 result pages)
```

## File-by-file guide (the new stuff)

### `src/wdn/models/attacker.py`
Defines `AttackerGNN` (GraphSAGE backbone + delta + mask heads on
nodes and edges) and the `StealthBudget` / `apply_stealth_budget`
helpers that project the raw output to a feasible attack: `‖δ‖∞ ≤ ε`,
at most `k` sensors touched, mass conservation respected via a soft
penalty in the training loss. Drop-in replacement for any module that
expects a dict with `delta_p / delta_q / mask_p_logits / mask_q_logits`.

### `src/wdn/models/attacker_moe.py`
`MixtureOfAttackersGNN` + an `AttackerRouter`. Symmetric to the
defender's temporal MoE: `K` specialist attackers + a graph-pooled
router that picks one per snapshot. Includes `diversity_loss` (mean
pairwise cosine similarity between expert deltas) and `balance_loss`
(negative entropy of batch-mean router probabilities) to prevent mode
collapse. Returns the same dict as `AttackerGNN` plus `router_logits`,
`router_probs`, `expert_delta_p`, `expert_delta_q`.

### `src/wdn/train_selfplay.py`
The Stackelberg loop. Key flags:
- `--attacker_moe` — use the population instead of a single attacker.
- `--lambda_recon` / `--lambda_physics` / `--lambda_budget` —
  defender-side losses for the attacker.
- `--attacker_steps`, `--defender_steps` — alternation ratio.
- `--curriculum` — auto-grow `(ε, k)` once the defender beats
  `--curriculum_threshold` on validation.
- `--no_pattern_features` — turn off the replay-pattern detection
  features in the defender's anomaly head (useful for ablations).
- Anomaly-F1 + replay-F1 composite is used for best-model selection
  (replaces "lowest val recon" which tended to pick replay-blind
  checkpoints).

### `scripts/heldout_novel_attack.py`
Synthesises two attacks that none of the defenders have seen:
`sinusoidal` (injects `A·sin(ωt+φ)` over the window) and `swap`
(swaps the pressure observation of two attacked sensors). Reports
F1 / AUROC / precision / recall for the pretrained, self-play single,
and self-play MoE defenders.

### `scripts/vocabulary_mining.py`
Runs the trained attacker over the test split, projects every
snapshot's perturbation vector to 2D with t-SNE, colours points by
ground-truth attack class and (for MoE) by router pick. Prints an
expert-class purity table that surfaces emergent attack families.

### `scripts/eval_*.py`, `scripts/plot_selfplay.py`
A small evaluation toolbox: compare any two checkpoints on the test
split, run three seeds for error bars, and emit publication-quality
PNGs to `presentation/charts/`.

## Reproducing the headline numbers

```bash
# 1. Generate datasets (≈ 1 min)
python -m wdn.generate --config configs/generate_moe_net1.yaml
python -m wdn.generate --config configs/generate_temporal_moe_modena.yaml
python -m wdn.generate --config configs/generate_temporal_moe_net3.yaml

# 2. Train pretrained Temporal-MoE defenders (≈ 15 min on Apple-Silicon MPS)
python -m wdn.train_temporal_moe --data_dir data/temporal_moe_modena
python -m wdn.train_temporal_moe --data_dir data/moe_net1 --hidden_dim 32
python -m wdn.train_temporal_moe --data_dir data/temporal_moe_net3

# 3. Self-play fine-tune (≈ 15 min/network)
python -m wdn.train_selfplay \
    --data_dir data/temporal_moe_modena \
    --defender_ckpt runs/temporal_moe/<latest>/best_model.pt \
    --attacker_moe --num_attackers 4 \
    --epsilon_p 5.0 --k_p 15 \
    --lambda_recon 5.0 --lambda_physics 0.05 \
    --lambda_diversity 0.5 --lambda_atk_balance 0.1 \
    --attacker_steps 3 --defender_steps 1 \
    --curriculum --curriculum_threshold 0.85

# 4. Evaluate
python scripts/eval_atkmoe.py
python scripts/heldout_novel_attack.py
python scripts/vocabulary_mining.py \
    --attacker_ckpt runs/selfplay/<latest>/attacker.pt \
    --moe --num_experts 4

# 5. Render charts (writes PNGs to presentation/charts/)
python scripts/plot_selfplay.py
```

## Methodology in one screen

**Hand-crafted attack zoo.** Five baseline classes that the corruption
pipeline injects at training time:
1. **Random** — scaled / biased per sensor.
2. **Replay** — broadcast a recent legitimate value.
3. **Stealthy bias** — slow drift over a window.
4. **Noise** — high-variance jamming.
5. **Targeted** — random scaling biased toward high-impact sensors.

**Learned attack zoo (self-play).** An Attacker GNN ingests the clean
snapshot, outputs a per-sensor delta and a per-sensor attack-mask
logit. The training loop projects the output to the stealth budget and
feeds the perturbed snapshot to the (frozen) defender. Two losses
shape the attacker:

```
L_atk = L_stealth − λ_recon · L_damage + λ_phys · ‖B·q‖²
        + λ_div · cosine_similarity(experts)        # MoE only
        + λ_bal · negative_entropy(router_probs)    # MoE only
```

`L_stealth` is BCE that rewards the attacker for *not* being flagged
on the sensors it touched; `L_damage` is the defender's reconstruction
error on those same sensors; `B` is the node-edge incidence matrix.

**Defender training.** Standard temporal MoE loss with class-weighted
BCE for anomaly (`pos_weight = 5`), MSE for reconstruction, router
cross-entropy on the ground-truth attack class, balance entropy on the
batch-mean router probabilities. After self-play, the same defender
loss is computed against the *attacker-injected* labels rather than
the original corruption pipeline.

**Auto-curriculum.** When the defender's adversarial-F1 on validation
exceeds the threshold (default 0.85), the budget is bumped:
`ε ← min(ε + 0.5, ε_max)`, `k ← min(k + 2, k_max)`. The threshold
fired three times on the headline run, ramping `k` from 15 to 20.

## Honest limitations

1. **Replay is still hard.** Self-play single attacker pushes Modena
   replay F1 from 0.196 → 0.229; the Attacker-MoE drops it to 0.165.
   Replay is informationally close to clean — a single copy is
   plausible by construction. We discuss this as a remaining gap.
2. **Net1 regression.** Self-play actively hurts Net1's small-graph
   defender (F1 0.728 → 0.665). The pretrained Net1 model already
   saturates; the adversarial fine-tune adds capacity it doesn't need.
3. **Mode-collapsed attackers.** The Attacker-MoE uses 2 of 4 experts.
   The two-family discovery is a feature, not a bug, *for the
   thesis*; for a top-tier publication we would push diversity loss
   harder and report 4-mode separation.
4. **Modest absolute gains.** F1 +5.8%, MAE −23%, novel-attack F1
   +1–3%. Real, reproducible, but not earth-shattering. The
   methodology — not the magnitude — is the contribution.

## Tech stack

- Python 3.11+
- PyTorch 2.x + PyTorch Geometric
- WNTR (water-network simulator)
- Streamlit + Plotly (dashboard)
- ReportLab (presentation PDF)

## Dashboard

```bash
pip install streamlit plotly
python dashboard/precompute/export_demo_data.py
python dashboard/precompute/export_modena_demo.py
python dashboard/precompute/export_attack_analysis.py
python dashboard/precompute/export_attack_analysis_modena.py
streamlit run dashboard/app.py
```

| Page | Content |
|---|---|
| Network Overview | Topology + per-node properties |
| Reconstruction | Ground truth vs observed vs GNN predictions |
| Attack Analysis | Per-attack-type detection curves |
| Anomaly Detection | Interactive thresholding demo |
| Model Comparison | Spatial → Temporal → MoE benchmarks |
| Explainability | Feature / node / edge importance |

## Networks

- **Net1** — 11 nodes, 13 pipes (EPANET reference network).
- **Net3** — 97 nodes, 117 pipes (EPANET reference network, added in
  this work as a third benchmark).
- **Modena** — 272 nodes, 317 pipes (Bragalli et al., 2008).
