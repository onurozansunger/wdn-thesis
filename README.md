# WDN State Reconstruction & Anomaly Detection

Master's thesis project: State reconstruction and malicious anomaly detection in Water Distribution Networks using Graph Neural Networks.

## Overview

This project develops a machine learning model that takes a mix of true, false, and missing sensor data from a water distribution network and aims to:

1. **Reconstruct** the complete and most likely network state (pressures and flows)
2. **Detect anomalies** — in particular, malicious or profit-driven falsifications of sensor data

The approach uses deep learning (Graph Neural Networks) to capture spatial relationships between network nodes and flows, trained and validated on synthetic data generated with WNTR.

## Tech Stack

- Python 3.11+
- PyTorch + PyTorch Geometric
- WNTR (Water Network Tool for Resilience)
- scikit-learn, NumPy, Pandas, matplotlib

## Status

Work in progress.
