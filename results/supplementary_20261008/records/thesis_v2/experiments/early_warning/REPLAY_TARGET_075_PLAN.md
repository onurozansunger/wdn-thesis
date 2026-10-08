# Prospective replay target: 0.75

Prepared 2026-09-06 following the user's target revision. This document supersedes the target and next-action sections of REPLAY_NEXT_STAGE_PLAN.md for future work. It does not amend any completed experiment's frozen protocol, acceptance gates, or verdict.

## Objective and reference

Aim for mean full-system replay F1 at least **0.75** in one fresh paired confirmation after qualifying and freezing a new design. The independently confirmed received-pressure-history candidate remains the reference: replay **0.730307**, pooled **0.895097**, clean FPR **0.000548**, random **0.992003**, drift **0.898622**, noise **0.928197**, targeted **0.991808**. It was not deployed. The mean replay gap is **0.019693**, approximately two F1 percentage points.

This is a mean-performance target, not a claim that every source or model run exceeds 0.75. Report every source mean, the individual-run range, and uncertainty clustered by generator source. Protect the strong families at their existing levels; 0.75 is not permission to lower their performance.

The completed confirmation is evidence only. Its individual cases must not inform new features, settings, or training. Candidate gains measured on TRAIN or calibration cannot be added to 0.730307 as a forecast: those datasets differ.

## Completed designs retain their verdicts

- Received trajectory: repeatable TRAIN gain; protected full-system calibration replay 0.705392 to 0.709372. One calibration source regressed. It failed its original readiness gate and received no fresh confirmation.
- Received flow: repeatable TRAIN gain; zero fully protected calibration operating points in every model seed. It remains rejected regardless of the revised target.
- Probability combination and broad attack-period negative weighting did not produce repeatable qualifying replay gains.

Lowering the target does not qualify any of these designs retrospectively. Do not rerun calibration selection with relaxed protection constraints or spend a new confirmation on these rejected settings.

## New TRAIN-only overlap audit

Executed `audit_replay_complementarity.py` on saved predictions from all five source-held TRAIN folds and five seeds. No fitting, candidate alarm combination, calibration-case analysis, or confirmation access occurred. All 75 arm/fold/seed sets of saved family confusion counts and F1 values reproduced exactly. Prediction hashes were verified against their campaign manifests for the two challenger arms, and all audit inputs were fingerprinted.

At the unchanged TRAIN-selected thresholds, average counts per run were:

| Diagnostic | Count |
|---|---:|
| Replay positives | 417.00 |
| Trajectory replay misses | 161.32 |
| Flow detects a replay positive missed by trajectory | 14.32 |
| Flow-only additional replay false positives | 13.56 |
| Flow-only additional clean false positives | 72.96 |
| Trajectory detects a replay positive missed by flow | 18.68 |
| Both models miss a replay positive | 147.00 |

Flow detects 8.88% of trajectory misses in these aggregate counts. At a fixed 0.0005 clean-FPR budget selected on even TRAIN scenarios, the fraction is 9.83%, with 13.64 additional replay true positives, 18.00 additional replay false positives, and 136.36 additional clean false positives per run. These are error-overlap diagnostics, not a combined detector's performance, an attainable gain, or an upper bound. Seeds reuse the same five source datasets.

The evidence gives a limited rationale for learning jointly from the representations. It does not justify a union of alarms or establish that a combined representation will succeed.

## Next bounded candidate

Test one joint representation inside the same five General components and internal router: the existing 158-column received-pressure reference, the frozen 140 additional received-trajectory columns, and the frozen 44 additional received-flow context columns, for **342 columns total**. Include the original 42 point-history columns only once. Retain all existing transformations, bin widths, lags, windows, support indicators, fitting exclusions, model hyperparameters, population weights, and seed-specific training samples. Do not include the rejected probability-combination layer or broad negative reweighting.

This joint feature representation has not yet been trained. Earlier experiments named “combined” tested different combinations: trajectory plus broad negative weighting, or flow plus probability combination. Neither tested this joint feature bank.

Before fitting, freeze the joint runner, feature schema, hashes, rounding checks, and stage-specific evaluation code. Reuse existing pressure and flow references only where their training-source exclusions exactly match the new fold. Compare with the received-pressure-history reference on all five TRAIN folds and seeds 701–705, completing all 25 comparisons before selection. The reused TRAIN sources are development evidence, not independent validation.

Advance only with at least 20/25 replay improvements, strictly positive mean replay gain in each TRAIN source, and mean replay gain at least 0.01. Report every family's changes and both clean-period and attack-period false positives. The new joint model must pass the same 0.01 m and 0.05 m pressure feature-path checks at seed 701 on all five folds, with frozen thresholds and no reselection. These checks do not establish end-to-end sensor quantization robustness.

## Full-system qualification and final success

After TRAIN selection and rounding checks, fit the candidate on full TRAIN. Use calibration exclusively for the established threshold/router operating grid and compare with the reference on those same cases. Preserve the existing per-seed and per-source protections:

- Each model seed: replay improves, pooled F1 does not decrease, clean FPR does not increase and remains at most 0.005, and no other-family F1 loss exceeds 0.005.
- Each calibration source within a seed: replay loss at most 0.01, other-family loss at most 0.01, pooled loss at most 0.005, and clean-FPR increase at most 0.0001.
- Across seeds: positive mean replay gain on every calibration source.

For this prospective campaign, also require mean paired full-system calibration replay gain at least **0.02** before spending a fresh confirmation. This is a predeclared development effect-size hurdle motivated by the remaining target gap; it is not a claim that gains transfer additively between datasets. Other-family mean F1 must remain at least 0.80 alongside the tighter relative protections. The old absolute 0.78 calibration/source readiness rule belongs to earlier campaigns and remains unchanged in their records.

Only a qualifying design proceeds to one fresh confirmation with the original generator distribution and scenario protocol, six new sources and all five model seeds. Reserve source IDs after checking the registry; do not assume previously proposed but unused IDs remain available. Freeze all model artifacts, operating settings, code, and checks before generating or reading fresh confirmation data.

Success requires mean replay F1 at least 0.75, improvement in at least 24/30 paired runs and in every source mean, no mean pooled-F1 loss or clean-FPR increase, each other-family mean at least 0.80 with loss at most 0.005, and the existing source-level other-family, pooled, and FPR protections. Report any lower individual/source replay results explicitly. If either qualification or confirmation fails, record that result and retain the confirmed reference. Do not continue drawing fresh sources until a desired score appears.

## Fixed constraints and status

Retain General, Drift, and Noise as the three top-level branches and five internal General components. No new expert, replay-specific inference rule, alarm holdover, hidden generator input, clean hydraulic value, or future reading. General remains causal; Drift/Noise retain their declared three-hour delay. Pressure and flow missingness remain 0.50. Do not change severity, generator distribution, scenario splits, labels, or the locked test.

Completed in this target-revision step: prospective target/protection plan and saved TRAIN error-overlap audit. No joint-model training, new calibration sweep, fresh confirmation, or deployment was performed.

Artifacts: `runs/operational/early_warning_multiseed_v1/replay_target_075_v1/audit_protocol.json` and `train_complementarity.json`.
