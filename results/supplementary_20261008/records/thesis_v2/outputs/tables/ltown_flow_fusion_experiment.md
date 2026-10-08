# Received flow and internal probability combination

These are development findings. The independently confirmed received-pressure-history candidate remains replay F1 **0.730307** until a new independent confirmation is completed. No deployment change is implied.

## Source-held TRAIN: General branch only

All 25 outer fold/model-seed pairs completed before design selection. Model seeds reuse five source datasets and are not independent datasets. Teachers used nested source exclusions, including fitted normal references.

| Arm | Replay F1 | Pooled F1 | Drift | Noise | Random | Targeted | Clean FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| reference | 0.704881 | 0.873689 | 0.793541 | 0.794960 | 0.993272 | 0.989612 | 0.000265 |
| fusion | 0.703202 | 0.875739 | 0.800002 | 0.792384 | 0.995274 | 0.993960 | 0.000266 |
| flow | 0.718817 | 0.881726 | 0.806399 | 0.808454 | 0.995278 | 0.994778 | 0.000268 |
| combined | 0.706480 | 0.882298 | 0.812453 | 0.826405 | 0.992425 | 0.992240 | 0.000277 |

| Candidate | Replay paired gain | Positive pairs | Source mean replay gains |
|---|---:|---:|---|
| fusion | -0.001679 | 12/25 | +0.000505, -0.005188, -0.001367, -0.012669, +0.010323 |
| flow | +0.013935 | 23/25 | +0.016356, +0.003657, +0.011071, +0.023287, +0.015307 |
| combined | +0.001598 | 14/25 | +0.012282, -0.002859, -0.000214, -0.005997, +0.004779 |

TRAIN-selected candidate: **flow**.

Pressure feature-path rounding checks: **PASS**. These do not establish end-to-end sensor quantization robustness.

## Full-system calibration

These calibration results use a different corpus from independent confirmation and cannot be compared as a paired change from 0.730307.

Protection across all five seeds: **FAIL**.
Confirmation readiness: **FAIL**.

All five model seeds had **zero protected operating points** in the frozen grid. As a rejection diagnostic, the settings maximizing worst-family F1 averaged replay **0.702639**, versus **0.705392** for the paired reference (-0.002753). These settings were rejected and are not an accepted full-system candidate or an independent result. This is the maximum-worst-family selection objective, not an unconstrained maximum-replay claim.

| Model seed | Reference replay | Rejected maximum-worst-family grid replay |
|---|---:|---:|
| 701 | 0.703773 | 0.700803 |
| 702 | 0.700458 | 0.697899 |
| 703 | 0.711689 | 0.708313 |
| 704 | 0.709827 | 0.707417 |
| 705 | 0.701214 | 0.698766 |

stop; retain confirmed received-history candidate.

## Interpretation and verification

The flow arm adds 44 shared features based on actually received flows and pressures. The fusion arm calibrates five component probabilities and trains a regularized correction of internal routing using binary detection loss. The original three top-level branches and three-hour Drift/Noise delay remain.

Verified 167 frozen code/input files and 790 artifact hashes, all nested source exclusions and all 75 candidate model interfaces.

Pressure and flow missingness remain 0.50. Generator distribution, severity and scenario splits are unchanged. No locked test or consumed independent confirmation cases were accessed by this experiment. Other-family protections apply to the complete system; General-only family changes must not be represented as deployed regressions or improvements.

Thirteen focused tests passed. Additional checks verified the original 42-feature path exactly on 7,253,913 outer endpoints, all five full-TRAIN model hyperparameters and component interfaces, and the unchanged full-TRAIN flow-reference fingerprint.
