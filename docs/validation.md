# Release validation

Validated on 21 September 2026 in the separate publication checkout, without training models or running new evaluations.

- **140 tests passed, 14 skipped.** Skips concern campaign artifacts omitted from the public checkout. Dependency deprecation warnings were emitted; no test failed.
- **223 evidence records passed SHA-256 verification.** The evidence verifier also recomputed the original replay comparison, supplementary baseline confusion-count metrics and historical latency comparison.
- **The current thesis compiled successfully** using `latexmk` and the included sources, figures and tables. This was a build check, not a new editorial or visual review.
- The publication-only test fixture reloads the shared trajectory-calibration module to prevent import-time configuration from another experiment leaking between tests. The required L-Town feature schema and Modena generator/split metadata are included. Model implementations and frozen scores were not changed.

## Validation environment

This records the local verification environment; it is not a claim that every historical model was fitted with these package versions.

| Package | Version |
|:---|:---|
| Python | 3.13.5 |
| NumPy | 2.3.1 |
| pandas | 2.3.2 |
| PyYAML | 6.0.2 |
| PyTorch | 2.11.0 |
| PyTorch Geometric | 2.7.0 |
| Matplotlib | 3.10.8 |
| WNTR | 1.4.0 |
| scikit-learn | 1.8.0 |
| LightGBM | 4.7.0 |
| pytest | 9.0.2 |

Commands used from the publication root:

```bash
PYTHONPATH=src python -m pytest -ra
python scripts/verify_release.py
python results/evidence/verify.py
```

`results/release_manifest.json` identifies the published files by relative path and SHA-256, excluding itself. The release verifier recomputes both network summaries from all 120 recorded evaluation cells and checks their frozen protocol and evaluation hashes.
