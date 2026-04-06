# WDN State Reconstruction & Anomaly Detection

Master's thesis project: State reconstruction and malicious anomaly detection in Water Distribution Networks using Graph Neural Networks.

## Overview

This project develops a machine learning model that takes a mix of true, false, and missing sensor data from a water distribution network and aims to:

1. **Reconstruct** the complete and most likely network state (pressures and flows)
2. **Detect anomalies** — in particular, malicious or profit-driven falsifications of sensor data

The approach uses deep learning (Graph Neural Networks) to capture spatial relationships between network nodes and flows, trained and validated on synthetic data generated with WNTR.

## Key Results

### Net1 (11 nodes, 13 pipes)

| Metric | Value |
|--------|-------|
| Pressure MAE (unobserved nodes) | 1.417 m |
| Improvement over best analytical baseline (WLS) | ~13x |
| Anomaly detection AUROC | 0.883 |
| Anomaly detection precision | 93.6% |
| Anomaly detection F1 | 0.766 |

### Modena (272 nodes, 317 pipes)

| Metric | Value |
|--------|-------|
| Pressure MAE (unobserved nodes) | 0.406 m |
| Pressure RMSE (unobserved nodes) | 0.566 m |
| Anomaly detection AUROC | 0.856 |
| Anomaly detection precision | 98.3% |
| Anomaly detection F1 | 0.749 |

The model achieves **3.5x better** reconstruction accuracy on the larger Modena network — denser graph topology provides richer spatial context for GNN message passing.

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
│   ├── dataset.py            # PyTorch Geometric dataset + normalization
│   ├── config.py             # Configuration dataclasses
│   ├── baselines.py          # Pseudo-inverse and WLS baselines
│   ├── metrics.py            # MAE/RMSE, Precision/Recall/F1/AUROC
│   ├── sensor_oracle.py      # Optimal sensor placement via MC Dropout
│   ├── models/
│   │   ├── gnn.py            # GNN backbone (GAT, GATv2, Transformer, GraphSAGE, GCN, GPS)
│   │   ├── recon.py          # ReconGNN — state reconstruction with physics loss
│   │   └── multitask.py      # MultiTaskGNN — joint reconstruction + anomaly detection
│   ├── train_recon.py        # Training script for ReconGNN
│   ├── train_multitask.py    # Training script for MultiTaskGNN
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
│   ├── architecture_comparison.json
│   └── comparison_30_50.json
│
├── runs/                     # Training outputs
│   ├── <run_id>/             # ReconGNN runs
│   ├── multitask/<run_id>/   # MultiTaskGNN runs
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
| **Model Comparison** | Cross-network comparison, attack detection benchmarks, GNN vs analytical baselines, architecture benchmark |
| **Sensor Oracle** | Uncertainty-based sensor placement with interactive greedy placement simulation (Net1) |
| **Training History** | Loss curves and validation metrics (F1, AUROC) over training epochs |

### Usage Tips

- **Hover** over nodes and edges on any network graph to see detailed information
- On the **Reconstruction** page, use the snapshot selector to browse different test cases
- On the **Anomaly Detection** page, drag the **threshold slider** to see how precision/recall changes in real time
- On the **Sensor Oracle** page, drag the **placement slider** to simulate adding sensors and watch the error reduction curve

## Methodology

### Benchmark Networks
- **Net1** — 11 nodes, 13 pipes. 50 scenarios with randomized demand patterns, 24h simulation each (1,250 snapshots)
- **Modena** — 272 nodes, 317 pipes (Bragalli et al., 2008). 1,250 steady-state scenarios with 30% demand variation

### Data Generation
- Hydraulic simulations using WNTR (Water Network Tool for Resilience)
- Corruption: 50% missing sensors (Bernoulli), Gaussian noise, and 5 adversarial attack types

### Attack Types
1. **Random falsification** — scaled and biased readings
2. **Replay attack** — past legitimate readings replayed to mask changes
3. **Stealthy bias injection** — gradual drift over time
4. **Noise injection** — large random noise simulating sensor jamming
5. **Targeted attack** — preferentially attacks high-impact sensors

### Models
- **ReconGNN**: GNN backbone + pressure/flow prediction heads + physics-informed loss (mass conservation)
- **MultiTaskGNN**: Shared GNN backbone with 4 heads — pressure reconstruction, flow reconstruction, pressure anomaly detection, flow anomaly detection
- **MC Dropout**: Uncertainty quantification via multiple stochastic forward passes at inference time

### Baselines
- Pseudo-inverse reconstruction (incidence matrix)
- Weighted Least Squares with mass conservation and Laplacian smoothness constraints
- Mean imputation

## Status

Work in progress — next steps include GNN explainability (GNNExplainer / attention analysis) and temporal modeling (GNN+GRU).
