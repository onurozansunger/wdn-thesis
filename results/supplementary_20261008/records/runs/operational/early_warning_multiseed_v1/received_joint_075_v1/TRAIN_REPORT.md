# Joint received trajectory and flow TRAIN campaign

General-only TRAIN development, not full-system or independent

| Arm | Replay | Pooled | Drift | Noise | Random | Targeted | Clean FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| reference | 0.704881 | 0.873689 | 0.793541 | 0.794960 | 0.993272 | 0.989612 | 0.000265 |
| joint | 0.738154 | 0.882706 | 0.804969 | 0.806403 | 0.995168 | 0.995057 | 0.000270 |

Selected: joint. Next: rounding checks then protected full-system calibration.

All 25 fold/model-seed pairs completed before selection. Source-held TRAIN folds have been used in development before and are not independent confirmation. Other-family protection must be assessed in the full system.
