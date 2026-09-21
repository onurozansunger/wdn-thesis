# Reproducing and inspecting the results

## 1. Verify the published evidence

From the repository root, run:

```bash
python3 scripts/verify_release.py
python3 results/evidence/verify.py
```

The first command checks the release manifest, the final protocol and evaluation-freeze hashes, all 120 model/source cells and their reported network means. It regenerates no data and fits no models. The second checks the 223 original evidence records, the original 30-pair L-Town replay comparison, supplementary baseline counts and thresholds, and the historical Modena latency comparison.

`results/final/final_results.json` is the authoritative final score record. `results/final/final_artifact_audit.json` identifies the fitted artifacts by SHA-256. Original records retain their original paths, including author-machine absolute paths. Those paths are provenance, not portable download locations; they must be mapped to a restored archive for a full rerun.

## 2. Install and test the implementation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pytest
```

The project requires Python 3.11+. LightGBM may require an OpenMP runtime on macOS. Optional optimization experiments use `python -m pip install -e '.[optimization]'`.

The dependency ranges describe installation requirements, not a fully pinned reconstruction of the original training environment. `docs/validation.md` records the environment used to check this release. Tests that require omitted campaign artifacts are expected to skip.

## 3. Understand the training pipeline

The campaign entry points are:

| Stage | Implementation |
|:---|:---|
| Corpus definitions and feature caches | `thesis_v2/experiments/early_warning/build_feature_cache.py` |
| Repeated fitting, calibration and evaluation | `thesis_v2/experiments/early_warning/run_campaign.py` |
| L-Town received-pressure-history extension | `thesis_v2/experiments/early_warning/protected_history_system.py` |
| Final ten-seed extension | `thesis_v2/experiments/early_warning/final_ten_model_seeds.py` |
| Supplementary single-classifier controls | `thesis_v2/experiments/early_warning/single_model_baseline.py` |

These are the retained research drivers. They depend on frozen reference models, split records and feature banks from the experiment archive. Running `--stage all` in a fresh checkout is not a complete reproduction recipe: the initial five-seed campaign, L-Town confirmation and ten-seed extension have different stages and source sets.

Before a full rerun, restore the simulation corpora, reference models, model bundles and feature banks at the paths specified by the frozen protocol and artifact audit. Check their hashes, preserve train/calibration/evaluation separation, and freeze all models and decision rules before evaluation. Large `.npz`, `.pkl`, `.joblib` and `.pt` artifacts are deliberately excluded from Git. No public binary archive or one-command full rerun is provided by this release.

## 4. Build the thesis

The current manuscript sources, generated figures and tables are included. With a TeX installation providing `latexmk` and the packages in `main.tex`:

```bash
cd thesis/manuscript
mkdir -p build/chapters build/appendices
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

The output is `build/main.pdf`. Compiling the manuscript uses the supplied figures and tables and requires no training. Historical project paths mentioned in the manuscript identify original evidence locations; the portable public evidence copy is under `results/evidence/records/`.
