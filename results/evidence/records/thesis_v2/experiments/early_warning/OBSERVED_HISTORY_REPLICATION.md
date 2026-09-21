# Frozen received-observation history replication

This campaign evaluates the 42 received-pressure history features inside the General residual mixture. The deployed architecture remains General, Drift, and Noise, with the existing router, feedback, and three-hour delay for specialized decisions. General contains five internal tree components; they are not five top-level branches.

The original fold-0/701 pilot is exploratory TRAIN evidence, with General mixture replay F1 0.7173 → 0.7432. Independently confirmed full-system replay F1 remains 0.6757. Neither the pilot nor this replication estimates independent deployed performance.

## Frozen scope

Run all 24 remaining combinations of five source-held TRAIN folds and model seeds 701–705. The 20 combinations in folds 1–4 determine the progression gate; four additional fold-0 seeds are supplemental. Existing fold references exclude their held source. Both arms use identical positive rows, negative samples, population weights, seeds, and tree settings. The candidate adds the original 42 columns to every internal component and the internal router.

The feature recipe remains received-pressure differences at 1–12, 24, and 48 hours, normalized by the existing reference scale and binned at 0.1 sigma, with explicit availability indicators. It receives only readings, masks, timestamps, sensor indices, and reference scales. The primary path uses unrounded readings, as in the pilot; pilot rounding sensitivity remains a separate feature-path diagnostic.

Even-numbered held-source TRAIN scenarios select thresholds using the pilot's fixed budget grid and pooled-F1/worst-family-F1/clean-FPR ordering. Odd-numbered scenarios supply diagnostics. Each arm selects its own thresholds with the same rule. No result-dependent feature, model, or threshold-grid changes are allowed.

## Progression gate, declared before execution

All 20 primary pairs must complete. Replay mean F1 must increase in every held source and overall, with at least 16/20 positive pairs. Mean pooled F1 may not decrease; no source mean pooled F1 may decrease more than 0.005. Each other family's mean F1 may not decrease more than 0.005, and its source mean may not decrease more than 0.01. Mean clean FPR may not increase; no source mean clean FPR may increase more than 0.0001.

The family tolerances operationalize material regression for this TRAIN screening stage. A gate pass permits full-TRAIN training and protected calibration-only full-system selection; it does not authorize a deployed-performance claim. A gate failure stops progression for this frozen recipe. All failures must be reported without relaxing the gate. Model seeds reuse scenarios; they are not independent experimental datasets.

Pressure and flow missingness remain 0.50. Generator distribution, severity, scenarios, splits, and locked test remain fixed. Calibration and evaluation are not read by this runner.

## Execution and artifacts

Run from the repository root:

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 /opt/miniconda3/bin/python thesis_v2/experiments/early_warning/replicate_observed_history.py --workers 2
```

Output: `runs/operational/early_warning_multiseed_v1/observed_history_replication_v1/`.

`protocol_frozen.json` records code, input, reference, and pilot hashes, software versions, the full pair plan, feature recipe, and gate. Each pair records its sample/weight hashes, models, prediction banks, frozen thresholds, diagnostic metrics, event counts, and artifact hashes. `summary.json` and `report.md` are emitted only after all 24 pairs complete. The runner refuses changed frozen artifacts and validates source/scenario membership.

Verification: `tests/test_observed_history.py` covers causality, exact timestamp matching, masks, quantization, and source/scenario separation; `tests/test_observed_history_replication.py` covers source leakage, incomplete replication, family/FPR protection, and exclusion of supplemental evidence from the gate.
