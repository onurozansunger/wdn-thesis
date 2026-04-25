# WDN State Reconstruction & Anomaly Detection

Master's thesis project: state reconstruction and adversarial-attack
detection in Water Distribution Networks using Graph Neural Networks.

## Overview

The model takes a mixture of true, falsified, and missing sensor
readings from a water network and:

1. **reconstructs** the complete plausible network state (pressures
   and flows), and
2. **flags** sensors whose readings have been compromised.

It is trained on synthetic data generated with WNTR and evaluated on
two benchmark networks: **Net1** (11 nodes) and **Modena** (272
nodes).

## Headline results

The story is the architecture progression — **single GNN → temporal
GNN+GRU → temporal Mixture-of-Experts**. Each step targets a specific
attack class the previous one cannot handle.

### Overall anomaly detection (pressure)

| Model | Net1 F1 | Net1 AUROC | Modena F1 | Modena AUROC |
|-------|:---:|:---:|:---:|:---:|
| MultiTaskGNN (spatial) | 0.766 | 0.883 | 0.749 | 0.856 |
| TemporalMultiTaskGNN | 0.688 | 0.874 | 0.765 | 0.937 |
| MoE (spatial) | 0.628 | 0.894 | 0.755 | 0.944 |
| **MoE (temporal)** | **0.646** | **0.914** | **0.780** | **0.972** |

### Replay attack — the hardest class

A single replayed reading is by construction plausible, so spatial
models cannot flag it. Temporal context plus expert specialisation
unlocks detection.

| Model | Net1 replay F1 | Modena replay F1 |
|-------|:---:|:---:|
| MoE (spatial) | 0.050 | 0.002 |
| **MoE (temporal)** | **0.318** | **0.177** |
| Improvement | **6.4×** | **88×** |

## Project structure

```
├── configs/                         # YAML configuration files
│   ├── generate_moe_net1.yaml       # Net1 MoE training data
│   └── generate_temporal_moe_modena.yaml
│
├── src/wdn/                         # Source code
│   ├── data_generation.py           # WNTR simulation → graph snapshots
│   ├── corruption.py                # Missing data, noise, 5 attack types
│   ├── dataset.py                   # PyG dataset + normalisation
│   ├── temporal_dataset.py          # Sliding-window temporal dataset
│   ├── config.py                    # Configuration dataclasses
│   ├── metrics.py                   # MAE/RMSE, P/R/F1/AUROC
│   ├── sensor_oracle.py             # MC-Dropout sensor placement
│   ├── explainability.py            # GNNExplainer integration
│   ├── generate.py                  # CLI entry point
│   ├── models/
│   │   ├── gnn.py                   # GNN backbones + temporal wrapper
│   │   ├── multitask.py             # MultiTaskGNN (spatial)
│   │   ├── temporal_multitask.py    # GNN + GRU
│   │   ├── moe.py                   # Spatial Mixture-of-Experts
│   │   └── temporal_moe.py          # Temporal Mixture-of-Experts
│   ├── train_multitask.py
│   ├── train_temporal.py
│   ├── train_moe.py
│   └── train_temporal_moe.py
│
├── data/                            # Network files + generated datasets
│   ├── Net1.inp
│   ├── modena.inp
│   └── networks/
│
├── runs/                            # Training outputs (per timestamp)
│
└── dashboard/                       # Streamlit dashboard
    ├── app.py                       # Landing page
    ├── pages/                       # 6 result pages
    ├── utils/                       # Theme + data loaders
    └── precompute/                  # Scripts to build demo data
```

## Tech stack

- Python 3.11+
- PyTorch + PyTorch Geometric
- WNTR (water-network simulator)
- Streamlit + Plotly (dashboard)

## Dashboard

```bash
pip install streamlit plotly

# Build precomputed demo data once
python dashboard/precompute/export_demo_data.py
python dashboard/precompute/export_modena_demo.py
python dashboard/precompute/export_attack_analysis.py
python dashboard/precompute/export_attack_analysis_modena.py

streamlit run dashboard/app.py
```

| Page | Content |
|------|---------|
| Network Overview | Topology + per-node properties |
| Reconstruction | Ground truth vs observed vs GNN predictions |
| Attack Analysis | Per-attack-type detection curves |
| Anomaly Detection | Interactive thresholding demo |
| Model Comparison | Spatial → Temporal → MoE benchmarks |
| Explainability | Feature / node / edge importance |

## Methodology

### Networks
- **Net1** — 11 nodes, 13 pipes (EPANET reference network).
- **Modena** — 272 nodes, 317 pipes (Bragalli et al., 2008).

### Data generation
- Hydraulic simulations via WNTR.
- 50 scenarios × 25 hourly timesteps per network for the temporal
  models (the original Modena `.inp` ships with `duration = 0`, so the
  generator overrides it from the YAML config).
- Corruption: 50% missing sensors (Bernoulli), Gaussian noise, plus 5
  adversarial attack types.

### Attack types
1. **Random falsification** — scaled / biased readings.
2. **Replay** — past legitimate readings re-broadcast.
3. **Stealthy bias** — gradual drift over time.
4. **Noise injection** — high-variance jamming.
5. **Targeted** — random falsification, but biased toward
   high-impact sensors.

### Models
- **MultiTaskGNN.** Shared GNN backbone with four heads (pressure /
  flow reconstruction, pressure / flow anomaly).
- **TemporalMultiTaskGNN.** Adds a GRU over a 6-step sliding window
  and explicit temporal-stability features (Δ, std, range, log-std,
  half-window drift, change count) on the pressure-anomaly head.
- **MixtureOfExpertsGNN.** Six MultiTaskGNN experts (one per attack
  class) plus a small attack-classifying router. Outputs are mixed by
  the softmax router weights; a load-balancing entropy term prevents
  router collapse.
- **TemporalMixtureOfExpertsGNN.** MoE wrapper around the temporal
  experts — the configuration that produces the headline replay
  numbers above.

### Loss
- MSE on reconstruction.
- Class-weighted BCE on anomaly heads (`pos_weight = 5`) since attacks
  affect ~15% of observed sensors.
- Cross-entropy on the router against the ground-truth attack class.
- Mass-conservation residual on the predicted flow vector.
- Negative-entropy load-balancing penalty over batch-mean router
  probabilities.
