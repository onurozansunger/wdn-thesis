# Causal seasonal drift/noise expert results

## What changed

The previous 109-feature bank used only short lags even though the hydraulic
simulation has a fixed 24-hour demand cycle. The promoted experts add seven
strictly past-only pressure features: signed/absolute differences and support at
24 and 48 hours, plus their signed agreement. Drift uses a full seasonal tree
blended 90/10 with the frozen expanded-TRAIN Optuna trial 2. Noise uses a
95/5 blend of a fast-reset seasonal tree and a full seasonal tree. Pressure and
flow missing probabilities remain 0.50; attacks, severities, splits and labels
are unchanged.

## Expanded-TRAIN source-held result

The four complete generator sources were each held out once. The architecture
was selected after TRAIN OOF diagnostics, so these are development-selected OOF
scores and the F1 thresholds are oracle diagnostics.

| Family | Mechanism macro AP | Seasonal macro AP | Seasonal oracle F1 | Precision | Recall | Clean FPR | Post-event FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Drift | 0.5661 | 0.6322 | 0.7040 | 0.9130 | 0.5729 | 0.000886 | 1 |
| Noise | 0.7112 | 0.7674 | 0.7399 | 0.8374 | 0.6628 | 0.000972 | 36 |

Drift improved in 20/21 event scenarios and noise in 18/20. Both passed every
frozen AUPRC, F1, clean-FPR, post-event and worst-event gate. The independent
audit reconstructed causal features, loaded-model predictions, blends, metrics
and gates exactly.

## Full fit and late development validation

The final reference and experts were fitted on all 62 allowed TRAIN scenarios.
One threshold per specialist was selected on the three original calibration
scenarios with a 0.005 clean and all-negative FPR ceiling. Only after selection
was frozen were the three reused development-validation scenarios evaluated.

| Specialist | Calibration F1 | Validation F1 | Validation precision | Validation recall | Validation AUPRC | Validation clean FPR |
|---|---:|---:|---:|---:|---:|---:|
| Drift | 0.8348 | **0.7273** | 0.9333 | 0.5957 | 0.7228 | 0.002032 |
| Noise | 0.8244 | **0.6966** | 0.9688 | 0.5439 | 0.7652 | 0.001883 |

Noise misses 0.70 by 0.0034. Its calibration threshold is already constrained
by the false-alarm ceiling; lowering it after seeing validation would be
validation leakage. A separate TRAIN-only commissioning-normalisation probe did
not improve cross-source threshold transfer consistently, so it was not added.

## Conservative whole-detector integration

The existing strong general/abrupt/replay path was kept unchanged. Calibration
selected the prespecified max-evidence addition of the two new specialists.

| Candidate | Overall F1 | Macro family F1 | Random | Replay | Drift | Noise | Targeted | AUPRC | Clean FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Old mixture, recalibrated | 0.7797 | 0.6856 | 0.9853 | 0.8657 | 0.2759 | 0.3714 | 0.9297 | 0.7999 | 0.001007 |
| Seasonal max evidence | **0.7895** | **0.8170** | 0.9781 | 0.8333 | 0.6849 | 0.6364 | 0.9524 | **0.8744** | 0.001883 |

The integration greatly improves balance and ranking quality but does not beat
the historical mixture's separately chosen 0.8037 overall F1, and replay falls
slightly while remaining well above 0.50. Strict-online 0.80 for both weak
families is therefore not established. The locked test was not opened.

## Post-development family-balanced operating mode

After the primary reused-validation result had been inspected, a separate
calibration-only operating rule allocated the fixed 0.005 false-positive budget
across the old mixture, drift expert and noise expert, then OR'ed their binary
decisions. The calibration objective explicitly maximised the lower of drift
and noise F1. This is a post-validation development iteration and cannot count
as independent confirmation.

| Mode | Overall F1 | Precision | Random | Replay | Drift | Noise | Targeted | Overall FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Primary max evidence | 0.7895 | 0.7302 | 0.9781 | 0.8333 | 0.6849 | 0.6364 | 0.9524 | 0.002138 |
| Family-balanced OR | 0.7656 | 0.6889 | **0.9706** | **0.7470** | **0.7027** | **0.7021** | **0.9579** | 0.002621 |

The balanced mode is the first strict row/hour online development result in
this campaign with every attack-family F1 above 0.70. It trades 0.0238 overall
F1 and 0.0413 precision for the weak-family floor. It does not establish the
0.80 goal and needs a fresh untouched benchmark for confirmation.

## Reproducibility

- TRAIN OOF run: `runs/operational/seasonal_family_experts_v3`
- Full deployment run: `runs/operational/seasonal_family_deployment_v2`
- Post-development balanced operating mode:
  `runs/operational/seasonal_balanced_union_v1`
- Machine-readable audited results:
  `thesis_v2/outputs/seasonal_family_experts_results.json` and
  `thesis_v2/outputs/seasonal_family_deployment_results.json` and
  `thesis_v2/outputs/seasonal_balanced_union_results.json`
- All three independent audit files report `all_checks_pass: true`.
