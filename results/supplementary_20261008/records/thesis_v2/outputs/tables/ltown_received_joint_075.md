# Joint received trajectory and flow experiment

**Protected calibration improvement, but insufficient replay gain for fresh confirmation toward 0.75.** The experiment completed its frozen TRAIN, rounding, and calibration stages. All family/FPR/pooled protections passed; the predeclared +0.02 calibration replay-gain hurdle failed (+0.004544). No new independent confirmation or deployment occurred.

Target: mean full-system replay F1 0.75 with pooled, other-family, source, and false-alarm protection.

The independently confirmed received-pressure-history reference remains replay F1 **0.730307** until another confirmation qualifies. It has not been deployed.

## TRAIN General branch only

All 25 paired comparisons completed before selection. Five sources each reuse five model seeds; these are development results, not independent or full-system performance.

| Arm | Replay | Pooled | Drift | Noise | Random | Targeted | Clean FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| reference | 0.704881 | 0.873689 | 0.793541 | 0.794960 | 0.993272 | 0.989612 | 0.000265 |
| joint | 0.738154 | 0.882706 | 0.804969 | 0.806403 | 0.995168 | 0.995057 | 0.000270 |

Replay wins: 25/25; mean paired gain +0.033272.
Source mean replay gains: +0.020937, +0.044725, +0.028197, +0.037644, +0.034858.
Selected: joint.

Pressure feature-path rounding passes: True. This is not an end-to-end sensor quantization test.

## Full-system calibration

Protection in every model seed: True. Confirmation readiness: False.
Paired mean replay: 0.705392 to 0.709936 (+0.004544).
Candidate pooled F1 0.907480; clean FPR 0.000360.
Source mean replay gains: 70811: +0.000412, 71811: +0.007021, 72811: +0.003251, 73811: +0.006879.
Calibration is a different corpus from independent confirmation; gains cannot be added to 0.730307 as a forecast.
stop; retain confirmed received-history reference.

No new independent confirmation or deployment was performed.

## Integrity and fixed design

Verified 458 frozen input/code files and 125 TRAIN artifacts. Existing baseline metrics reproduced exactly in every TRAIN pair.
The 342 inputs contain the original 158 inputs, 140 received-trajectory additions, and 44 received-flow additions. The 42 point-history inputs appear once. All added features reach the same five General components and internal router.
General, Drift, and Noise remain the three branches. Drift/Noise delay remains three hours. Pressure and flow missingness remain 0.50. Severity, generator distribution, scenario splits, labels, and the locked test remain unchanged.

## Complete paired calibration metrics

Means across five model seeds, each scored on the same four calibration sources. These are calibration results, not fresh confirmation.

| Metric | Reference | Joint | Change |
|---|---:|---:|---:|
| replay | 0.705392 | 0.709936 | +0.004544 |
| pooled | 0.892981 | 0.907480 | +0.014499 |
| random | 0.993608 | 0.994274 | +0.000666 |
| drift | 0.912608 | 0.914158 | +0.001550 |
| noise | 0.920550 | 0.920381 | -0.000169 |
| targeted | 0.992668 | 0.994739 | +0.002071 |
| clean_fpr | 0.000562 | 0.000360 | -0.000202 |
| all_negative_fpr | 0.000645 | 0.000468 | -0.000177 |
| attack_period_negative_fpr | 0.001216 | 0.001212 | -0.000004 |

Noise mean F1 decreased by 0.000169, within the 0.005 protection limit; the other-family means improved. All four calibration source means improved on replay, but the gains ranged only from +0.000412 to +0.007021. The replay target remains unmet; calibration gains cannot be added to the confirmed 0.730307 as a predicted confirmation result.

## Additional TRAIN false-alarm audit

At the fixed TRAIN-selected rules, replay precision improved from 0.861591 to 0.878161 and recall from 0.597093 to 0.636928. Mean replay false positives fell from 41.08 to 37.88 per run. Across all attack families, however, the negative-reading false-positive rate increased from 0.000591 to 0.000735, and clean FPR increased slightly. These TRAIN-only tradeoffs are distinct from the protected full-system calibration results above.

## Final verification and retained artifacts

Eighteen focused tests passed. All 25 TRAIN model interfaces and all five full-TRAIN model interfaces, random seeds, hyperparameters, training samples, and shared feature coverage passed verification. The full raw calibration baseline reports reproduced exactly in all five seeds. The fresh-confirmation guard was exercised and correctly refused access after the readiness failure.
Calibration-stage frozen input/code entries verified: 510.
All workers finished. The locked test and consumed independent confirmation cases were not used for this experiment.

[Prospective plan](/Users/ozanbabac5/wdn_thesis/thesis_v2/experiments/early_warning/REPLAY_TARGET_075_PLAN.md)
[Frozen experiment](/Users/ozanbabac5/wdn_thesis/runs/operational/early_warning_multiseed_v1/received_joint_075_v1/protocol_frozen.json)
[TRAIN selection](/Users/ozanbabac5/wdn_thesis/runs/operational/early_warning_multiseed_v1/received_joint_075_v1/train_selection.json)
[Calibration decision](/Users/ozanbabac5/wdn_thesis/runs/operational/early_warning_multiseed_v1/received_joint_075_v1/full_system/decision.json)
[Calibration integrity checks](/Users/ozanbabac5/wdn_thesis/runs/operational/early_warning_multiseed_v1/received_joint_075_v1/full_system/calibration_integrity_verification.json)
