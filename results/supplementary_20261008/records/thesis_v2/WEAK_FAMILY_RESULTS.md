# Drift/noise training campaign: development results

Pressure detection, optionally using observed pressure + flow as inputs. A heterogeneous expert stack, not the original GNN or a normalised MoE gate.
Same seed811 data, attacks, outer splits and 0.50 missing probabilities. No test extraction/evaluation.
Validation has been used in earlier development; these are not independent generalisation results.

**Calibration-selected candidate: 1 (context).** Parameters: {'leaves': 15, 'regularisation': 10.0, 'early_weight': 1.0, 'C': 0.1}
The first three candidates use fixed matching hyperparameters to compare feature recipes; the fourth is a small Optuna refinement of the calibration-selected representation.
All models and thresholds were frozen before new validation extraction/scoring. No validation winner substitution.
General/abrupt/replay experts and their OOF scores are preserved; only weak experts and fusion are refitted.
New feature/reference blocks exclude target groups; each joint reference is fitted inside its TRAIN scenario fold.
Score memory is a causal feature, not a calibrated state posterior. Event metadata only weight TRAIN examples and annotate evaluation.
OOF specialist outputs train fusion; its own fitting scores are not an independent held-out evaluation.

## Validation comparison

| Variant | Overall F1 | Random | Replay | Drift | Noise | Targeted | FP | Clean FPR (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Previous family-balanced model | 0.7325 | 0.9781 | 0.8286 | 0.5079 | 0.5750 | 0.8251 | 178 | 0.1547 |
| Previous selected redesign | 0.7889 | 0.9853 | 0.8529 | 0.5312 | 0.4675 | 0.9837 | 106 | 0.1734 |
| New 0: local | 0.7730 | 0.9855 | 0.8529 | 0.5538 | 0.4810 | 0.9859 | 127 | 0.2125 |
| New 1: context [CALIBRATION SELECTED] | 0.8176 | 0.9927 | 0.8657 | 0.4590 | 0.4054 | 0.9788 | 64 | 0.0988 |
| New 2: joint | 0.8484 | 0.9855 | 0.8788 | 0.2593 | 0.3429 | 0.9736 | 14 | 0.0130 |
| New 3: context | 0.8231 | 0.9784 | 0.8657 | 0.4839 | 0.4054 | 0.9811 | 60 | 0.0895 |

## Selected independent expert AUPRC

Per-expert thresholds use global calibration F1; these are independent diagnostics, not per-family oracle thresholds.

| Expert | Random | Replay | Drift | Noise | Targeted | Clean FPR (%) |
|---|---:|---:|---:|---:|---:|---:|
| general | 0.9992 | 0.9083 | 0.6185 | 0.5819 | 0.9990 | 0.0186 |
| abrupt | 0.9996 | 0.9009 | 0.5769 | 0.5738 | 0.9994 | 0.0373 |
| replay | 0.9994 | 0.9613 | 0.3224 | 0.3712 | 0.9896 | 0.0541 |
| drift | 0.9938 | 0.8480 | 0.6161 | 0.4879 | 0.9904 | 0.0298 |
| noise | 0.9873 | 0.6093 | 0.5460 | 0.4684 | 0.9857 | 0.2125 |

## Selected event diagnostics

| Scenario / family | TP / positives | First 3h TP / positives | Sensors detected / targets | FP on unattacked during event | Next 16h FP on former targets |
|---|---:|---:|---:|---:|---:|
| 6 / random | 68 / 68 | 19 / 19 | 14 / 14 | 1 | 0 |
| 6 / targeted | 122 / 125 | 15 / 16 | 14 / 14 | 1 | 0 |
| 10 / noise | 15 / 57 | 3 / 26 | 8 / 14 | 2 | 1 |
| 10 / targeted | 86 / 87 | 20 / 20 | 14 / 14 | 4 | 0 |
| 11 / replay | 29 / 35 | 10 / 10 | 11 / 14 | 0 | 0 |
| 11 / stealthy | 14 / 47 | 0 / 21 | 8 / 14 | 0 | 0 |

## 0.80 goal and remaining errors

Both weak families >= 0.80: **False**.
Drift gap: 0.3410; noise gap: 0.3946.

- noise: 42 misses; 8 follow an earlier same-sensor hit in the event; 34 precede the first hit or belong to a never-detected sensor. The owner expert recovers 6 at its frozen independent threshold.
- stealthy: 33 misses; 1 follow an earlier same-sensor hit in the event; 32 precede the first hit or belong to a never-detected sensor. The owner expert recovers 2 at its frozen independent threshold.

## Generalisation and reference diagnostics

These are independent specialist scores on held-out TRAIN scenarios, not the fusion model's own fitting scores.
| Candidate | OOF drift AUPRC | OOF noise AUPRC | Calibration noise AUPRC | Validation noise AUPRC |
|---|---:|---:|---:|---:|
| 0 | 0.5994 | 0.6135 | 0.9235 | 0.5803 |
| 1 | 0.5715 | 0.5029 | 0.8961 | 0.4684 |
| 2 | 0.5736 | 0.4647 | 0.9270 | 0.5857 |
| 3 | 0.5368 | 0.5395 | 0.8827 | 0.4980 |

calibration normal-observation reference MAE: pressure-only 0.1892 m; joint 0.2043 m.

validation normal-observation reference MAE: pressure-only 0.1678 m; joint 0.1810 m.

The joint low-rank reference did not lower normal prediction error. This does not disprove physical pressure-flow fusion.
Calibration alone favoured a representation whose OOF weak-expert discrimination was worse. This is a development warning, not a validation-based reselection.

Event-boundary bookkeeping above is diagnostic only, not an attainable new F1 or a deployable oracle.
Only one drift and one noise validation event exist. Early/missed observations may overlap normal readings; no 0.80 guarantee.
The four-candidate budget is complete. No old model was replaced and no extra search is silently started.

## Verification

Selected-model reload reproduces calibration/validation scores, calibration choice and independent audit exactly; source/data/cache hashes match.
39 tests passed before training; the extended final suite passed 40 tests, including test-split and premature-validation guards. No test dataset examples were used.

Further interpretation after this audit is recorded in WEAK_FAMILY_FOLLOWUP.md when needed.
