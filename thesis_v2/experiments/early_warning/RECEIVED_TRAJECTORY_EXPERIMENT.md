# Received trajectories and protected negative-example training

Execution of the next-stage plan, authorized 2026-09-06. The machine-readable protocol is `runs/operational/early_warning_multiseed_v1/received_trajectory_v1/protocol_frozen.json`. It, rather than this explanation, fixes the experiment choices.

## Audit and candidate choice

The TRAIN audit uses all five existing held-source folds and model seeds 701–705. Thresholds for every component and mixture use a fixed clean budget of 0.0005 on the existing even held-source scenarios. Odd scenarios provide diagnostics. These are reused TRAIN development partitions, not independent confirmation.

The mixture misses 3,628 of 10,425 repeated replay-positive decisions. Only 646 misses (17.8%) have an alarm from any internal component at its corresponding matched budget, and 360 (9.9%) have an internal replay-component alarm. This upper-bound diagnostic does not implement a label-selected oracle. The mixture also alarms on 16.46% of unchanged readings at replay-targeted sensors, compared with 0.0324% at untargeted sensors. Consequently, the conditional second candidate is negative-example training rather than a new router objective.

The four arms are:

- **Reference:** existing 42 received-history features, original training weights, same General mixture.
- **Trajectory:** reference plus 140 common trajectory features.
- **Hard negative:** reference features, modified training weights only.
- **Combined:** both changes.

All five internal General components and the internal router retain their original tree hyperparameters and component count. No new top-level branch or inference rule is introduced.

## Fixed feature recipe

Use trailing windows 3 and 6 hours and lags 1–12, 24, and 48 hours. For each lag/window, compute five features from timestamp-aligned received pairs: median signed difference, median absolute difference, median absolute deviation, pair count, and age of the newest usable pair. Distribution statistics require two pairs; insufficient support is explicit. Differences and distribution statistics are quantized at 0.1 reference sigma. The existing 42 point features remain unchanged.

Only received pressure values, masks, timestamps, sensor indices, and the existing reference scale cross the feature boundary. Each source/scenario is isolated. All inputs precede or coincide with decision time; no alarm is extended or retrospectively changed. The General decision remains immediate, and the existing three-hour Drift/Noise delay remains.

The training change multiplies weights of negative TRAIN rows during any attack family by three, then renormalizes the total negative weight to its original value. Positive weights and the exact sampled rows are preserved. The rule is symmetric across attack families and uses no target identities, true lags, event identity, or severity. Family labels and anomaly labels are training supervision only, not inference inputs.

## Evaluation and stopping

Complete all 25 fold/seed pairs before selecting an arm. Each arm uses the same existing threshold budget grid and even/odd TRAIN partition. Select among candidates with at least 20/25 replay wins, positive mean replay change in every source, and mean replay gain at least 0.01. Rank by mean worst-family F1, pooled F1, then lower clean FPR. Report every other-family change. General-only family regressions are not evidence of full-system protection or failure.

If none qualify, stop without calibration or new confirmation. Otherwise, evaluate only the selected arm under 0.01 m and 0.05 m additional-feature-path rounding on all five folds at seed 701, using its frozen thresholds. Mean replay improvement and nonnegative source changes must persist. These are feature-path sensitivity checks, not end-to-end sensor quantization robustness.

Fit a passing design on full TRAIN at component seeds 1301–1305. Preserve the external specialists, router, feedback, verifier, and their frozen operating rules. On calibration only, select symmetric internal router temperature/shrinkage and a General threshold. Reproduce the received-history reference exactly on the same cached calibration cases. Require pooled F1 and clean FPR mean nonregression, every other-family mean loss at most 0.005, and source-level guards of 0.01 family F1, 0.005 pooled F1, and 0.0001 clean FPR. The absolute clean-FPR budget remains 0.005.

Fresh confirmation is allowed only if the design is protected for every model seed, improves replay on every source mean, and has every family mean at least 0.78 both overall and within each calibration source. Values 0.78–0.80 are explicitly near target, not attainment of 0.80. Otherwise stop and retain the previously confirmed candidate. Do not design new features from calibration failures or generate confirmation sources to search for a favorable outcome.

The General-only source-held results do not become source-held full-system results by attaching full-TRAIN specialists. This experiment makes no such claim. The complete system's protection is evaluated on calibration, then on fresh confirmation only if it qualifies.

## Commands

```sh
MPLCONFIGDIR=/tmp/wdn-mpl-cache OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 /opt/miniconda3/bin/python thesis_v2/experiments/early_warning/audit_replay_train.py
MPLCONFIGDIR=/tmp/wdn-mpl-cache OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 /opt/miniconda3/bin/python thesis_v2/experiments/early_warning/experiment_received_trajectory.py all --workers 3
```

For the selected arm only, run `check_trajectory_rounding.py --fold N` for every N in 0–4, then run that script without arguments to aggregate. Only after passing that check run `calibrate_received_trajectory.py all` under the same environment variables. The calibration runner refuses to load calibration if its TRAIN prerequisites fail. No runner in this campaign opens the locked test or consumed confirmation sources.

## Verification and execution record

Eleven focused tests initially passed for feature causality, exact timestamp handling, masked-value independence, unchanged point features, quantization, source/scenario isolation, and training-weight invariants. Scientific code and data are fingerprinted before candidate training. Existing sources, missingness probabilities (0.50 pressure and flow), generator configuration, and scenario splits are unchanged.

A parallel cache-preparation conflict on one shared feature-name temporary file was corrected before any candidate model was trained or scored. Each fold now writes its own schema. The initial protocol and repair explanation are retained under `pretraining_cache_write_repair/`; caches were regenerated and the corrected runner frozen before training. Feature definitions and selection criteria did not change.
