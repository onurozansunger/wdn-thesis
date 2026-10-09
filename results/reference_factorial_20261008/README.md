# E11: reference factorial evidence

This archive contains the completed target-group exclusion × IRLS experiment: four conditions, two networks, three paired model seeds and six reused evaluation sources per network. All 24 calibration records and 144 evaluation cells are included; four policies give 576 selected cell records.

Read the [study guide](../../docs/reference-factorial.md), [frozen design](records/runs/operational/reference_factorial_20261008/protocol_design.json), and [results summary](records/runs/operational/reference_factorial_20261008/report/summary.json). The original E1–E9 and E10 artifacts remain unchanged.

From the repository root, run:

```sh
python results/reference_factorial_20261008/verify.py
```

The standard-library verifier checks publication hashes, recorded provenance, count-derived metrics, calibration path selections, all five CSV tables, and paired source-level effects and interactions. FPR values are stored as rates; multiply effects by 100 for percentage-point changes.

Models, score arrays and raw telemetry are omitted. Their fingerprints document identity but do not permit independent training, inference, score-quantile reconstruction or endpoint-alignment checks. The evaluation sources were used previously, and the fixed allocation/gate hyperparameters were selected for the retained control. These limits apply to all reported comparisons.
