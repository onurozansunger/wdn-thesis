# Supplementary evidence, 8 October 2026 (E10)

This supplement contains the completed retrospective operating-point comparison,
frozen-rule component diagnostics and three retained flow-feature development
studies cited as E10 in the revised thesis. It adds evidence to the original
[submission snapshot](https://github.com/onurozansunger/wdn-thesis/tree/49036da8016097f2ebb627b3b58cea1a07432a47)
without changing its models, source files or reported results.

The new comparison uses ten existing model fits and six existing evaluation
sources in each network: **120 model/source cells**. All **20 calibration
selection records** precede evaluation analysis. There was no retraining, new
data generation or extraction of the original reserved test scenarios.

## Read the results

- [Analysis policy](analysis/operating_points/protocol.json) and [complete numerical summary](analysis/operating_points/complete_summary.json)
- [Results and interpretation](analysis/report/report.txt)
- [Selected operating points](analysis/report/selected_operating_points.csv), [paired differences](analysis/report/paired_differences.csv) and [component diagnostics](analysis/report/frozen_component_diagnostics.csv)
- [Modena figure](analysis/report/modena_operating_point_sensitivity.pdf) and [L-Town figure](analysis/report/ltown_operating_point_sensitivity.pdf)
- [Evidence guide](../../docs/evidence-supplement.md) and [original-to-published path manifest](manifest.json)

The three-hour single LightGBM classifier is the principal comparison because
it shares the hybrid's maximum observation allowance. The zero-hour classifier
is retained for context. Both pooled-F1 and worst-family-F1 calibration objectives
are reported at all seven declared clean-FPR caps.

The hybrid follows a prespecified threshold path with branch allocation,
internal routing, verifier and replay-veto settings fixed. The comparisons use
common calibration FPR caps; achieved evaluation FPRs can differ. The plots
describe this path and do not estimate an exhaustive ROC frontier, AUROC or
AUPRC for the hybrid. Evaluation sources
were already used in the original research; this is a supplementary comparison,
not fresh independent confirmation.

The eight component variants alter final alarm contributions or guards while
retaining fitted models, upstream scores and original thresholds. Removing a
specialist's final alarm contribution leaves its score channels in shared
router, feedback and verifier inputs. Disabling the external replay veto does
not disable General's internal mixture routing. These conditional diagnostics
do not isolate early warning, target-group exclusion or robust fitting, and
do not estimate a reduced architecture's performance after retraining.

## Verify the saved evidence

From the repository root, with Python 3.11 or later:

```bash
python3 results/supplementary_20261008/verify.py
```

The checker uses only the Python standard library. It verifies the manifest,
completion and provenance records, calibration/evaluation hash links, retained
original-result checks, and summary calculations from the saved JSON cells.
It does not load models, regenerate data, run inference or choose thresholds.
The original release checks remain available:

```bash
python3 scripts/verify_release.py
python3 results/evidence/verify.py
```

## Files and reproduction boundary

The manifest maps each original relative path to its published path and records
its byte count and SHA-256 hash. The **955 archived scientific files** are copied
byte for byte. They include 210 score-extraction JSON sidecars, 20 calibration
validation records, 20 calibration selections, 120 evaluation cells, the frozen
analysis policy and drivers, CSV/PDF/PNG outputs, and the supporting historical
records. Publication documentation and the standard-library checker are new;
their hashes are listed separately in the manifest.

Fitted models, large feature caches, raw simulator datasets and NPZ prediction
arrays are not included. Their recorded hashes retain the connection to the
research archive, but their bytes cannot be verified from this checkout alone.
The saved summaries can be checked without those artifacts; reproducing model
predictions or the threshold sweep requires them and the original scientific
runtime. The frozen drivers retain their original workspace-relative paths.
The manifest allows those paths to be reconstructed when the external artifacts
are available; copying the drivers alone is not a full rerun.

Five frozen historical records contain the original local workspace path in
artifact provenance or Markdown links. Those strings were retained to preserve
the records' hashes. Use the relative links in the evidence guide and manifest
to locate the published copies. Newly written documentation uses repository
paths.

