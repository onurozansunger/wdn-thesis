# WDN State Reconstruction & Anomaly Detection

This project provides a complete research codebase for state reconstruction and malicious anomaly detection in Water Distribution Networks (WDNs) using synthetic data from WNTR. It includes dataset generation, baselines, GNN models, training/evaluation pipelines, and reproducible experiment runs.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Quickstart (End-to-End Demo)

```bash
python -m wdn.generate --config configs/generate.yaml
python -m wdn.train_recon --config configs/train_recon.yaml
python -m wdn.train_multitask --config configs/train_multitask.yaml
python -m wdn.eval --config configs/eval.yaml
python -m wdn.sweep --config configs/sweep.yaml
```

All outputs are stored in `runs/<run_id>/` with metrics, plots, and copied configs.

## CLI Entrypoints

- `python -m wdn.generate --config configs/generate.yaml`
- `python -m wdn.train_recon --config configs/train_recon.yaml`
- `python -m wdn.train_multitask --config configs/train_multitask.yaml`
- `python -m wdn.eval --config configs/eval.yaml`
- `python -m wdn.sweep --config configs/sweep.yaml`

## Project Structure

- `src/wdn/` core library
- `configs/` YAML configs
- `data/` datasets and network `.inp` files
- `runs/` outputs
- `tests/` unit tests

## Notes

- Default device is CPU, with optional MPS acceleration on macOS if available.
- The included network is a small EPANET `.inp` to keep iteration fast.
- The design is extensible to larger networks and temporal sequences.
