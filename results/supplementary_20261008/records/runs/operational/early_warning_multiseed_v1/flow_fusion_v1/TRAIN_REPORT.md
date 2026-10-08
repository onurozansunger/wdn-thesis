# Received-flow and probability-combination TRAIN campaign

General-only TRAIN development, not full-system or independent

| Arm | Replay | Pooled | Drift | Noise | Random | Targeted | Clean FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| reference | 0.704881 | 0.873689 | 0.793541 | 0.794960 | 0.993272 | 0.989612 | 0.000265 |
| fusion | 0.703202 | 0.875739 | 0.800002 | 0.792384 | 0.995274 | 0.993960 | 0.000266 |
| flow | 0.718817 | 0.881726 | 0.806399 | 0.808454 | 0.995278 | 0.994778 | 0.000268 |
| combined | 0.706480 | 0.882298 | 0.812453 | 0.826405 | 0.992425 | 0.992240 | 0.000277 |

Selected: flow. Next: rounding checks then protected full-system calibration.

All 25 fold/model-seed pairs completed before selection. Source-held TRAIN folds have been used in development before and are not independent confirmation. Other-family protection must be assessed in the full system.
