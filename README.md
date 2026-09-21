# Sensor-Attack Detection in Water Distribution Networks

Research code and final experimental evidence for **A Hybrid Mixture-of-Experts Architecture for Sensor-Attack Detection in Water Distribution Networks**.

The system combines a normal-pressure reference, a general detector and dedicated drift/noise specialists to detect five sensor-attack families on the **Modena** and **L-Town** networks. The final evaluation covers ten independently fitted model seeds and six source datasets per network.

[Final results](results/final/final_results.json) · [Reproduction guide](REPRODUCE.md) · [Architecture](docs/architecture.md) · [Evidence package](results/evidence/README.md)

## Final results

Each entry is the equal-weight mean over **60 model/source cells per network** (model seeds 701–710 × six fixed sources). F1 scores and false-positive rates are expressed on a 0–1 scale.

| Metric | Modena | L-Town |
|:---|---:|---:|
| Random attack F1 | 0.954947 | 0.991953 |
| Replay attack F1 | 0.928270 | 0.731161 |
| Drift attack F1 | 0.827377 | 0.898975 |
| Noise attack F1 | 0.843871 | 0.927854 |
| Targeted attack F1 | 0.968018 | 0.991789 |
| Family macro F1 | 0.904497 | 0.908346 |
| Pooled F1 | 0.866941 | 0.895023 |
| Clean false-positive rate ↓ | 0.000584 | 0.000550 |

Family macro F1 gives equal weight to the five attack families. Pooled F1 is computed within each evaluation cell before averaging across cells. Clean false-positive rates correspond to approximately **0.0584%** and **0.0550%**.

Modena exceeds 0.80 mean F1 for every family. L-Town replay remains the limiting case: **0.731161**, below the 0.75 target. In its matched comparison, the L-Town received-pressure-history extension improves replay F1 in all 60 cells, with a mean gain of **0.024681**.

The ten seeds are replicate fits, not an inference ensemble. The last five seeds reuse the original six confirmation sources; this measures sensitivity to model randomness and is not a fresh independent confirmation. The 60 cells are not 60 independent datasets. Pressure and flow missingness remain fixed at 0.50, and the locked test was not read. See the [frozen protocol](results/final/protocol_frozen.json) for the complete evaluation contract.

## Method

The final architecture has three top-level branches:

- **General:** a tree-based mixture with general, abrupt, replay, drift and noise components. L-Town adds 42 received-pressure-history features, bringing its General input to 158 features.
- **Drift:** a specialist using temporal evidence for gradual changes.
- **Noise:** a specialist using temporal evidence for increased variability.

Routing, evidence feedback, verification and calibrated operating points control how the branches contribute to the final decision. Drift and Noise use a declared **three-hour decision delay**. This information allowance is part of the evaluation protocol.

The repository also retains earlier graph/recurrent model implementations and experimental controls needed to understand the research progression. Their historical scores describe different protocols and should not be read as direct comparisons with the final system. The supplementary single-classifier results document the trade-off between family coverage and false alarms; they do not establish universal superiority over external methods.

## Quick start

Use Python 3.11 or later. Model training requires the scientific dependencies listed in `pyproject.toml`; the evidence checks below use only the Python standard library.

```bash
git clone https://github.com/onurozansunger/wdn-thesis.git
cd wdn-thesis
python3 scripts/verify_release.py
python3 results/evidence/verify.py
```

To work with the code:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pytest
```

Artifact-dependent tests skip when their large campaign inputs are absent. See [REPRODUCE.md](REPRODUCE.md) before attempting training or evaluation: this checkout contains source code and auditable reports, while fitted models and large simulation/prediction arrays remain outside Git.

## Repository layout

| Path | Contents |
|:---|:---|
| `src/wdn/` | Simulation, attack generation, features, detectors and evaluation utilities |
| `configs/` | Benchmark and operational campaign configurations |
| `thesis_v2/experiments/` | Research drivers, including the final campaign under `early_warning/` |
| `tests/` | Metric, feature, calibration and information-boundary checks |
| `results/final/` | Frozen ten-seed results, protocol, selection and artifact audit |
| `results/evidence/` | 218 original records, their SHA-256 manifest and verification script |
| `data/` | EPANET network files and a small generator configuration fixture |
| `docs/` | Architecture, evidence scope and branch history |

## Branches and research history

**`main`** is the curated final research snapshot. **[`dev`](https://github.com/onurozansunger/wdn-thesis/tree/dev)** preserves the earlier self-play version unchanged. The former `main` is retained by the `archive/main-before-final-2026-09-21` tag. See [branch history](docs/branch-history.md) for exact commits.

## Attribution and scope

This is research software evaluated on simulated sensor attacks, not a validated production deployment. Network models retain their embedded source information. No new license grant for third-party assets is implied by this snapshot.
