# WDN State Reconstruction & Anomaly Detection

Master's thesis project: State reconstruction and malicious anomaly detection in Water Distribution Networks using Graph Neural Networks.

## Overview

This project develops a machine learning model that takes a mix of true, false, and missing sensor data from a water distribution network and aims to:

1. **Reconstruct** the complete and most likely network state (pressures and flows)
2. **Detect anomalies** — in particular, malicious or profit-driven falsifications of sensor data

The approach uses deep learning (Graph Neural Networks) to capture spatial and temporal relationships between network nodes and flows, trained and validated on synthetic data generated with WNTR.

## Key Results

### Spatial Model — MultiTaskGNN (GraphSAGE)

| Metric | Net1 (11 nodes) | Modena (272 nodes) |
|--------|-----------------|-------------------|
| Pressure MAE (unobserved) | 1.417 m | 0.406 m |
| Anomaly detection F1 | 0.766 | 0.749 |
| Anomaly detection AUROC | 0.883 | 0.856 |
| Anomaly precision | 93.6% | 98.3% |

### Spatio-Temporal Model — TemporalMultiTaskGNN (GraphSAGE + GRU)

| Metric | Net1 | Modena |
|--------|------|--------|
| Pressure MAE (unobserved) | 0.986 m | 0.867 m |
| Anomaly detection F1 | 0.688 | 0.765 |
| Anomaly detection AUROC | 0.874 | 0.937 |
| Reconstruction improvement | ~30% | ~31% |

The temporal model uses a 6-hour sliding window with GRU to capture diurnal demand patterns, improving reconstruction by ~30% on both networks. Modena achieves the best overall anomaly detection (AUROC 0.937) thanks to its denser topology providing richer spatio-temporal context.

### GNN vs Analytical Baselines (Net1)

| Method | Pressure MAE (unobserved) |
|--------|--------------------------|
| **GNN (GraphSAGE)** | **1.417 m** |
| WLS | 18.2 m |
| Pseudo-inverse | 18.2 m |
| Mean imputation | 23.2 m |

The GNN achieves **~13x better** reconstruction accuracy compared to the best analytical baseline.

## Project Structure

```
├── configs/                  # YAML configuration files
│   ├── generate.yaml         # Data generation (50 scenarios, 50% missing)
│   ├── generate_attacks.yaml # Data generation with adversarial attacks
│   ├── generate_modena.yaml  # Modena network data generation (1250 scenarios)
│   └── train_recon.yaml      # Training hyperparameters
│
├── src/wdn/                  # Core source code
│   ├── data_generation.py    # WNTR simulation → graph snapshots
│   ├── corruption.py         # Missing data, noise, and 5 attack types
│   ├── dataset.py            # PyTorch dataset + normalization
│   ├── temporal_dataset.py   # Temporal dataset (sliding windows for GNN+GRU)
│   ├── config.py             # Configuration dataclasses
│   ├── baselines.py          # Pseudo-inverse and WLS baselines
│   ├── metrics.py            # MAE/RMSE, Precision/Recall/F1/AUROC
│   ├── sensor_oracle.py      # Optimal sensor placement via MC Dropout
│   ├── explainability.py     # GNNExplainer integration
│   ├── generate_temporal_modena.py  # Temporal data generation for Modena
│   ├── models/
│   │   ├── gnn.py            # GNN backbone (GAT, GATv2, Transformer, GraphSAGE, GCN, GPS)
│   │   ├── recon.py          # ReconGNN — state reconstruction with physics loss
│   │   ├── multitask.py      # MultiTaskGNN — joint reconstruction + anomaly detection
│   │   └── temporal_multitask.py  # TemporalMultiTaskGNN — GNN + GRU
│   ├── train_recon.py        # Training script for ReconGNN
│   ├── train_multitask.py    # Training script for MultiTaskGNN
│   ├── train_temporal.py     # Training script for TemporalMultiTaskGNN
│   ├── eval_temporal_attacks.py   # Per-attack evaluation for temporal model
│   ├── run_architecture_comparison.py  # Benchmark 5 GNN architectures
│   ├── run_comparison.py     # 30% vs 50% missing + GNN vs baselines
│   └── eval_baselines.py     # Evaluate analytical baselines
│
├── data/                     # Data and result files
│   ├── Net1.inp              # EPANET Net1 network file
│   ├── modena.inp            # EPANET Modena network file
│   ├── generated/            # Clean simulation snapshots
│   ├── generated_attacks/    # Snapshots with adversarial attacks (Net1)
│   ├── modena_attacks/       # Snapshots with adversarial attacks (Modena)
│   ├── modena_temporal/      # Temporal snapshots with diurnal patterns (Modena)
│   ├── architecture_comparison.json
│   └── comparison_30_50.json
│
├── runs/                     # Training outputs
│   ├── <run_id>/             # ReconGNN runs
│   ├── multitask/<run_id>/   # MultiTaskGNN runs
│   ├── temporal/<run_id>/    # TemporalMultiTaskGNN runs
│   └── sensor_oracle/<run_id>/
│
└── dashboard/                # Streamlit interactive dashboard
    ├── app.py                # Landing page
    ├── pages/                # Dashboard pages (see below)
    ├── utils/                # Shared visualization and data loading
    ├── precompute/           # Scripts to prepare demo data
    └── data/                 # Pre-computed demo snapshots
```

## Tech Stack

- Python 3.11+
- PyTorch + PyTorch Geometric
- WNTR (Water Network Tool for Resilience)
- scikit-learn, NumPy, Pandas
- Streamlit + Plotly (dashboard)

## Dashboard

An interactive Streamlit dashboard for exploring the results.

### Setup

```bash
# Install dependencies (if not already installed)
pip install streamlit plotly

# Pre-compute demo data (only needed once)
python dashboard/precompute/export_demo_data.py

# Launch the dashboard
streamlit run dashboard/app.py
```

### Pages

All pages support both Net1 and Modena networks via a network selector.

| Page | Description |
|------|-------------|
| **Network Overview** | Interactive network topology with node/edge properties |
| **Reconstruction** | Side-by-side comparison of ground truth, observed (50% missing), and GNN predictions with error analysis |
| **Attack Analysis** | Per-attack-type detection performance (F1, precision, recall) across varying attack fractions |
| **Anomaly Detection** | Attack detection on the network graph with adjustable threshold and confusion matrix |
| **Model Comparison** | Cross-network comparison, spatial vs temporal, attack benchmarks, GNN vs baselines, architecture benchmark |
| **Sensor Oracle** | Uncertainty-based sensor placement with interactive greedy placement simulation (Net1) |
| **Training History** | Loss curves and validation metrics for both spatial and temporal models |
| **Explainability** | GNNExplainer analysis — node, edge, and feature importance for both networks |

## Methodology

### Benchmark Networks
- **Net1** — 11 nodes, 13 pipes. 50 scenarios with randomized demand patterns, 24h simulation each (1,250 snapshots)
- **Modena** — 272 nodes, 317 pipes (Bragalli et al., 2008). 1,250 steady-state scenarios with 30% demand variation; temporal variant uses diurnal demand patterns (200 scenarios × 25 timesteps = 5,000 snapshots)

### Data Generation
- Hydraulic simulations using WNTR (Water Network Tool for Resilience)
- Corruption: 50% missing sensors (Bernoulli), Gaussian noise, and 5 adversarial attack types
- Temporal data: 24h diurnal demand curve (1h resolution) added programmatically for Modena

### Attack Types
1. **Random falsification** — scaled and biased readings
2. **Replay attack** — past legitimate readings replayed to mask changes
3. **Stealthy bias injection** — gradual drift over time
4. **Noise injection** — large random noise simulating sensor jamming
5. **Targeted attack** — preferentially attacks high-impact sensors

### Models
- **ReconGNN**: GNN backbone + pressure/flow prediction heads + physics-informed loss (mass conservation)
- **MultiTaskGNN**: Shared GNN backbone with 4 heads — pressure reconstruction, flow reconstruction, pressure anomaly detection, flow anomaly detection
- **TemporalMultiTaskGNN**: GraphSAGE backbone + GRU for spatio-temporal modeling with sliding windows (6 timesteps). Same 4-head architecture as MultiTaskGNN but with temporal context
- **MC Dropout**: Uncertainty quantification via multiple stochastic forward passes at inference time

### Explainability
- **GNNExplainer** integration for both networks
- Node importance: identifies which nodes contribute most to predictions
- Feature importance: ranks input features (elevation, base demand, observed pressure, etc.)
- Edge importance: highlights critical pipes for information propagation

### Baselines
- Pseudo-inverse reconstruction (incidence matrix)
- Weighted Least Squares with mass conservation and Laplacian smoothness constraints
- Mean imputation
