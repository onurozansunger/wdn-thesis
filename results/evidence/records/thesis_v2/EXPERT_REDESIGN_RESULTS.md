# Expert redesign: frozen-data development results

Pressure-only detection, not the original temporal GNN. Same data/splits and pressure/flow missing probabilities 0.50.
This is a heterogeneous expert stack. Its logistic combiner is not a normalised, input-dependent MoE gate; neural/gated MoE integration is not claimed.
No locked test evaluation. Previous validation findings informed the design; these are not independent generalisation results.

## Method and selection

Noise-scaled, leverage-adjusted robust fitting on three sensor subsets; complete target-group exclusion in every computation.
Causal drift/ramp likelihood and variance-state evidence; no oracle targets, future smoothing or point adjustment.
The likelihood assumptions are approximate and balanced-risk scores are not calibrated attack probabilities.
Three inner scenario folds refit the normal reference. Their held-out expert scores train the logistic combiner.
Expert risk is balanced across families/events and normal scenarios. The new tree control shares this recipe; it is not an isolated reference-only ablation.
Trial 0 uses tree experts; trials 1–3 use logistic mechanism readouts for drift/noise and trees for the other experts.
Optuna tunes regularisation in this small, four-trial campaign. Its best parameters are not claimed to be globally optimal.
The old four-trial GNN study was neither extended nor restarted.

**Calibration-selected trial: 1 (mechanism); parameters: {'C': 0.1}.**
Selection maximises the worst calibration family F1, with macro/overall F1 tie breaks; normal/clean FPR <= 0.005 and replay F1 >= 0.50.
The choice and threshold were saved before validation extraction/scoring. Other trials below are disclosed, not substituted as winners based on validation.

For the selected model, a second calibration-only overall/replay operating point is also reported as a diagnostic; it does not replace the primary worst-family point.

## Validation comparison

| Variant | Overall F1 | Random | Replay | Drift | Noise | Targeted | Worst family F1 | FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Previous blind-reference mixture | 0.8037 | 0.9697 | 0.8923 | 0.2222 | 0.2985 | 0.9275 | 0.2222 | 29 |
| Previous dynamic, overall priority | 0.8181 | 0.9618 | 0.8788 | 0.3793 | 0.4384 | 0.9000 | 0.3793 | 52 |
| Previous dynamic, family balance | 0.7325 | 0.9781 | 0.8286 | 0.5079 | 0.5750 | 0.8251 | 0.5079 | 178 |
| Reference-only, fixed old mixture_legacy estimator/threshold | 0.8335 | 0.9624 | 0.8923 | 0.4068 | 0.4324 | 0.9735 | 0.4068 | 33 |
| Reference-only, fixed old mixture_macro_f1 estimator/threshold | 0.7115 | 0.9710 | 0.7838 | 0.5538 | 0.5417 | 0.9766 | 0.5417 | 213 |
| Trial 0: trees | 0.7788 | 0.9853 | 0.8406 | 0.5538 | 0.4810 | 0.9860 | 0.4810 | 122 |
| Trial 1: mechanism [CALIBRATION SELECTED] | 0.7889 | 0.9853 | 0.8529 | 0.5312 | 0.4675 | 0.9837 | 0.4675 | 106 |
| Trial 2: mechanism | 0.8024 | 0.9853 | 0.8529 | 0.5312 | 0.4474 | 0.9883 | 0.4474 | 90 |
| Trial 3: mechanism | 0.8014 | 0.9853 | 0.8657 | 0.5312 | 0.4533 | 0.9860 | 0.4533 | 91 |
| Selected model, diagnostic overall-priority point | 0.8340 | 0.9701 | 0.8923 | 0.5079 | 0.3478 | 0.9857 | 0.3478 | 41 |

| Trial | Calibration worst F1 | Validation worst F1 | Validation AUPRC | Sensor FPR (%) | Clean FPR (%) |
|---|---:|---:|---:|---:|---:|
| 0 | 0.7636 | 0.4810 | 0.8481 | 0.1961 | 0.1995 |
| 1 | 0.7706 | 0.4675 | 0.8507 | 0.1704 | 0.1734 |
| 2 | 0.7680 | 0.4474 | 0.8505 | 0.1447 | 0.1473 |
| 3 | 0.7581 | 0.4533 | 0.8534 | 0.1463 | 0.1510 |

## Selected model: independent expert AUPRC

One global calibration-F1 threshold per expert is used for the clean FPR column; it is not a per-family oracle threshold.

| Expert | Random | Replay | Drift | Noise | Targeted | Clean FPR (%) |
|---|---:|---:|---:|---:|---:|---:|
| general | 0.9992 | 0.9083 | 0.6185 | 0.5819 | 0.9990 | 0.0186 |
| abrupt | 0.9996 | 0.9009 | 0.5769 | 0.5738 | 0.9994 | 0.0373 |
| replay | 0.9994 | 0.9613 | 0.3224 | 0.3712 | 0.9896 | 0.0541 |
| drift | 0.9943 | 0.7558 | 0.6567 | 0.4260 | 0.9740 | 0.0466 |
| noise | 1.0000 | 0.8492 | 0.5774 | 0.5179 | 0.9972 | 0.1100 |

## Selected model: event and early-warning diagnostics

| Scenario / family | TP / positives | First 3h TP / positives | Sensors detected / targets | FP on unattacked sensors during event | FP on former targets in next 16h |
|---|---:|---:|---:|---:|---:|
| 6 / random | 67 / 68 | 18 / 19 | 14 / 14 | 1 | 5 |
| 6 / targeted | 124 / 125 | 15 / 16 | 14 / 14 | 2 | 0 |
| 10 / noise | 18 / 57 | 4 / 26 | 10 / 14 | 2 | 4 |
| 10 / targeted | 87 / 87 | 20 / 20 | 14 / 14 | 4 | 3 |
| 11 / replay | 29 / 35 | 10 / 10 | 11 / 14 | 1 | 0 |
| 11 / stealthy | 17 / 47 | 1 / 21 | 10 / 14 | 0 | 0 |

## Decision: not an all-family replacement

The selected point changes overall F1 from 0.7325 to 0.7889 and FP from 178 to 106 versus the previous balanced point.
However, noise F1 falls from 0.5750 to 0.4675; the requested all-family improvement is not achieved.
Independent drift-expert AUPRC changes from 0.5827 to 0.6567.
Independent noise-expert AUPRC changes from 0.6705 to 0.5179; the new noise readout is not an improvement.
The robust reference is a useful component candidate, supported by the fixed-old-estimator controls. It does not justify promoting the complete new stack.
All old models are preserved. No default production or thesis-final model was replaced. The four-trial campaign is complete; no further trial is silently added.

## Verification and limits

Original raw-data hashes and trained-source hashes match. Reloaded selected-model predictions, calibration selection and independent expert audit reproduce exactly.
All validation labels/family order match the original frozen benchmark. The internal scenario folds cover TRAIN once without overlap.
Remaining selected-model overall-F1 gap to 0.90: 0.1111.
Only one drift and one noise validation event are available. An event alarm is not equivalent to detecting each corrupted reading.
The FPR constraint is a calibration constraint, not a field guarantee. No claim of all-family success is made unless the table supports it.
Reference-only controls reuse the old estimator and threshold; input-distribution changes may make them suboptimal.
The previous oracle target-exclusion diagnostic is not used anywhere in the trained pipeline.
