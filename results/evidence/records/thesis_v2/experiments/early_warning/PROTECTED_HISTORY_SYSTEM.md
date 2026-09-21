# Protected received-history full-system experiment

This is a new development experiment authorized after the completed TRAIN replication. The earlier candidate's failed noise-protection gate remains recorded and is not reclassified.

The architecture retains General, Drift, and Noise as its three top-level branches. General has five internal tree components: general, abrupt, replay, drift, and noise. Specialized temporal evidence, the three-hour decision delay, external router, feedback, and verifier remain frozen. General's router calibration applies the same temperature/shrinkage transformation to all five internal outputs; it has no replay-specific inference rule.

## Declared candidate sequence

1. Fit the original 42 received-observation history features into all five internal components and the internal router on full TRAIN, using the deployed component seeds (1301–1305), the same negative sampling, weights, and tree settings.
2. If this candidate fails protection on any model seed or its mean calibration replay F1 is below 0.80, also test replacing only the internal general and replay components, retaining the baseline abrupt/drift/noise components and internal router. This candidate still has exactly five internal components.
3. Select one variant globally across all five model seeds. A variant must be feasible on every seed. Rank feasible variants by mean worst-family F1, then mean pooled F1. No best-seed reporting or per-seed architecture selection.

Each seed's baseline is the existing symmetrically recalibrated full system. Its saved calibration metrics must reproduce before candidate selection proceeds.

## Calibration selection

The original four calibration sources and all their scenarios are used. The predeclared symmetric router grid and General clean-budget grid are recorded in `protocol_frozen.json`. Specialized thresholds, verifier cutoffs, external router, and feedback are fixed.

For each seed, replay F1 must improve, pooled F1 must not fall, clean FPR must not rise or exceed 0.005, and each other family's F1 must remain within 0.005 of the paired baseline. On each calibration source, the maximum allowed other-family drop is 0.01, pooled drop is 0.005, clean-FPR increase is 0.0001, and replay drop is 0.01. Feasible settings maximize worst-family F1, then pooled F1, then lower clean FPR.

A failure on all declared variants retains the baseline and does not consume fresh confirmation data. Calibration findings are development results, not independent performance.

## Fresh confirmation

Six new generator seeds (110811–115811 in increments of 1000), each with 24 scenarios, are reserved before generation. Only after freezing the selected variant, all five models, and all five calibration rules may these sources be generated/read. The 30 paired baseline/candidate runs are evaluated once, with no rule or model changes based on confirmation scores.

Protected improvement requires positive mean replay change in every fresh source, at least 24/30 positive paired replay changes, non-decreasing mean pooled F1, non-increasing mean clean FPR, and each other family's mean F1 within 0.005 of baseline. Per-source protection matches the calibration tolerances; every run must satisfy the absolute 0.005 clean-FPR budget. Reaching mean replay F1 0.80 is reported separately, along with the number of runs below 0.80. Model seeds share source scenarios and are not 30 independent datasets.

Pressure and flow missingness stay 0.50. Attack severity, generator distribution, scenario splits, and the locked test remain fixed. Earlier confirmation outcomes and the locked test are not read for this experiment. Seed-collision checks inspect configuration/manifest seed metadata only.

## Execution

```sh
MPLCONFIGDIR=/tmp/wdn-mpl-cache OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 /opt/miniconda3/bin/python thesis_v2/experiments/early_warning/protected_history_system.py --stage all --workers 2
```

Artifacts are under `runs/operational/early_warning_multiseed_v1/protected_history_system_v1/`. The protocol records hashes of code, input data, references, existing models, and baseline rules. Each selected General bundle contains exactly five component estimators, their feature columns, and its internal router. The design manifest freezes those bundles and selected rules before fresh confirmation.
