# Received-observation history: frozen TRAIN replication

General branch only, containing five internal tree components. These are TRAIN diagnostics, not independent or deployed performance. Confirmed full-system replay F1 remains **0.6757**.

Progression gate: **FAIL**. Primary replay gains: 20/20 paired runs across four held sources. Model seeds reuse scenarios and are not independent datasets.

| Metric | Baseline mean | History mean | Paired mean change |
|---|---:|---:|---:|
| random | 0.991450 | 0.992434 | +0.000984 |
| replay | 0.651197 | 0.696536 | +0.045338 |
| drift | 0.795235 | 0.794241 | -0.000994 |
| noise | 0.802238 | 0.796917 | -0.005321 |
| targeted | 0.987369 | 0.988955 | +0.001586 |
| pooled | 0.862833 | 0.869115 | +0.006282 |
| clean_fpr | 0.000292 | 0.000261 | -0.000031 |

## Frozen guard results

- replay_mean_positive: PASS
- replay_positive_every_source: PASS
- replay_at_least_16_of_20_wins: PASS
- pooled_mean_preserved: PASS
- pooled_source_guard: PASS
- clean_fpr_mean_preserved: PASS
- clean_fpr_source_guard: PASS
- random_mean_guard: PASS
- random_source_guard: PASS
- drift_mean_guard: PASS
- drift_source_guard: PASS
- noise_mean_guard: FAIL
- noise_source_guard: FAIL
- targeted_mean_guard: PASS
- targeted_source_guard: PASS

## All new paired TRAIN diagnostics

| Fold | Seed | Replay before | Replay after | Pooled change | Clean FPR change |
|---|---|---:|---:|---:|---:|
| 1 | 701 | 0.5954 | 0.6667 | +0.007586 | +0.000016 |
| 1 | 702 | 0.6047 | 0.6633 | +0.006840 | +0.000006 |
| 1 | 703 | 0.6057 | 0.6701 | +0.007009 | +0.000027 |
| 1 | 704 | 0.6050 | 0.6709 | +0.008653 | -0.000008 |
| 1 | 705 | 0.5936 | 0.6575 | +0.007788 | +0.000006 |
| 2 | 701 | 0.6557 | 0.6977 | +0.008812 | -0.000025 |
| 2 | 702 | 0.6545 | 0.7021 | +0.006779 | +0.000016 |
| 2 | 703 | 0.6460 | 0.7092 | +0.010453 | +0.000019 |
| 2 | 704 | 0.6600 | 0.6950 | +0.006610 | +0.000018 |
| 2 | 705 | 0.6539 | 0.7121 | +0.010597 | -0.000008 |
| 3 | 701 | 0.6918 | 0.7336 | +0.010321 | -0.000226 |
| 3 | 702 | 0.6884 | 0.7089 | +0.009913 | -0.000240 |
| 3 | 703 | 0.6823 | 0.7254 | +0.002831 | +0.000029 |
| 3 | 704 | 0.6882 | 0.7311 | +0.003061 | +0.000003 |
| 3 | 705 | 0.6924 | 0.7196 | +0.009470 | -0.000243 |
| 4 | 701 | 0.6609 | 0.6951 | +0.002014 | +0.000013 |
| 4 | 702 | 0.6667 | 0.6918 | +0.000856 | +0.000006 |
| 4 | 703 | 0.6643 | 0.6972 | +0.002700 | -0.000006 |
| 4 | 704 | 0.6536 | 0.6931 | +0.001195 | +0.000006 |
| 4 | 705 | 0.6608 | 0.6907 | +0.002147 | -0.000024 |
| 0 | 702 | 0.7222 | 0.7284 | -0.000852 | +0.000013 |
| 0 | 703 | 0.7339 | 0.7455 | +0.000738 | +0.000003 |
| 0 | 704 | 0.7081 | 0.7356 | +0.001699 | +0.000002 |
| 0 | 705 | 0.7170 | 0.7387 | +0.002816 | +0.000005 |

Fold 0 is supplemental and excluded from the progression gate. The original fold-0/701 pilot is preserved separately. All 24 new pairs were required before assessing the gate. No feature, hyperparameter, threshold-selection rule, generator, severity, missingness, split, or locked test was changed.
