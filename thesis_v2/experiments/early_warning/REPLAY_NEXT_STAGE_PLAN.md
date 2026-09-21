# Plan: improve replay while preserving the other families

Prepared 2026-09-06. Planning document; no new training or confirmation was launched for this plan. Final feature definitions, training choices, and selection gates must be frozen before the corresponding experiment begins.

## Starting point and objective

Use the protected received-history candidate as the development reference, reproduced on the same development partitions as every challenger. Its completed fresh simulated-benchmark confirmation gave replay F1 0.730307, pooled F1 0.895097, clean FPR 0.000548, random F1 0.992003, drift 0.898622, noise 0.928197, and targeted 0.991808. Replay improved in 30/30 paired runs but remained below 0.80 in every run. These are candidate results; deployed artifacts were not replaced. The attributable paired replay gain was 0.706383 to 0.730307 on those sources, not 0.6757 to 0.730307 across different source sets.

The target is full-system F1 at least 0.80 for every family, with source-level robustness. Interpret 0.78–0.80 as near target, explicitly reported as such; it is not achievement of 0.80. Preserve the current strong families rather than allowing their performance to fall toward 0.80. Evaluate candidates using the worst-family objective, subject to pooled-F1 and false-alarm protection.

The consumed confirmation sources remain evaluation evidence. Do not mine their individual failures, optimize on them, or reuse them as independent confirmation. All new error analysis and representation development use existing TRAIN data.

## What the existing evidence says

Across the 20 primary source-held TRAIN comparisons (folds 1–4, seeds 701–705), the General mixture with received history averaged replay F1 0.696536, precision 0.856161, recall 0.587817, and AUPRC 0.608334. The internal replay component averaged F1 0.753499, precision 0.940557, recall 0.629532, and AUPRC 0.655992. These are means of per-run metrics, not pooled confusion counts and not full-system performance. The gap motivates examining information loss in the mixture, but component-specific operating points do not establish recoverable router gain.

Recall is the larger weakness in these diagnostics. Illustratively, F1 0.80 at precision 0.85 requires recall about 0.7568; at precision 0.90 it requires recall 0.72. Raising recall while preserving precision therefore needs better discrimination, not merely more alarms. These calculations are illustrative, not forecasts from mean TRAIN metrics.

The current 42 features describe individual current-versus-past received readings. They do not explicitly describe whether a short trajectory has consistent lagged evidence. The earlier rejected 56-feature experiment summarized existing residual features; it did not test this received-trajectory representation.

Code inspection of `src/wdn/operational_data.py` establishes two limitations to account for:

1. Replay payloads come from the pre-tampering historical observation stream. The detector only sees the actually received archive, which may contain earlier attacks. Historical availability does not imply that the stored value equals the payload source value.
2. A selected sensor is changed only when current and historical masks permit replay, and numerical no-ops are not positive labels. Event persistence is therefore not equivalent to every current reading being anomalous. Blindly carrying an alarm forward can increase false positives even within replay events.

Neither fact establishes an F1 ceiling. They motivate an explicit observability and false-positive audit.

## Stage 1: locate the recoverable errors on TRAIN

Use saved source-held predictions and recompute only missing diagnostic outputs. Separate missed positives from false alarms, with source, event, and sensor summaries so that one large event cannot dominate the conclusion.

- Measure errors against available received-history support, current residual magnitude, and onset/middle/end of events. Event metadata and labels may be used for offline audit only; never expose them to features, models at inference, or routing rules.
- Compare individual component scores with the mixture at comparable clean-FPR operating points selected on the existing inner TRAIN threshold partitions. Count cases where reliable component evidence is diluted, versus cases where all components miss the positive. A label-selected best-component result is only an optimistic diagnostic, not an admissible detector or claimed attainable performance.
- Measure false positives on clean periods, unaffected sensors within attacks, and unchanged readings inside replay events. Do not rely on clean-period FPR alone.
- Audit exactly which received pairs remain useful throughout an event; do not substitute the generator's hidden pre-tampering archive or true replay lag into model inputs.

Output: a compact error-budget table identifying representation, routing, and false-positive limitations. Freeze at most two new candidate changes from the following priorities before scoring them across the development folds.

## Stage 2A: shared causal received-trajectory features — first priority

Retain the 42-feature reference and test one compact additional bank. Proposed starting specification: trailing windows of 3 and 6 hours, with the existing lag bank 1–12, 24, and 48 hours. Compare timestamp-aligned received pairs within each trailing window. Summarize robust binned signed/absolute differences, their dispersion, usable-pair support, and age of usable evidence. Keep lag identities available so that the model can distinguish consistent evidence at one lag from unrelated matches across lags.

The exact statistics and output count must be fixed in the implementation protocol before the fold campaign. Avoid an unrestricted search over window lengths, lags, distance functions, or equality tolerances. Supply features to the existing five General components and its internal router; do not create an additional expert or a hard replay trigger.

Implementation requirements:

- Use actual received values, masks, and exact timestamps within the same source/scenario/sensor. Represent insufficient support explicitly. Do not interpolate across missing observations or scenario boundaries.
- Preserve 0.1-sigma difference quantization and the existing 0.01 m / 0.05 m feature-path robustness checks. No floating-point equality detector.
- Every value must be available by decision time t. A trailing six-hour window adds context, not a six-hour decision delay. Do not retrospectively relabel earlier decisions.
- Current-reading features remain available alongside context. Training and evaluation must penalize persistent alarms on unchanged readings.
- All fitted scales or transformations use the training portion only. Feature construction must not read clean hydraulic values, event labels, targets, true lags, or attack configuration.

The research motivation is that subsequence distances can expose discriminative temporal structure: [Lines et al., A Shapelet Transform for Time Series Classification](https://ueaeprints.uea.ac.uk/id/eprint/40201/1/LinesKDD2012.pdf). This proposal is a small causal, missingness-aware feature bank, not a reproduction of that algorithm. The paper does not establish effectiveness or a 0.80 target for this benchmark.

## Stage 2B: evidence-based training of the existing internal router — conditional second priority

The current internal router is trained to predict five mechanism classes. If Stage 1 shows a meaningful loss of useful component evidence at comparable false-alarm rates, test a router training objective based on out-of-fold component reliability. It should learn which of the same five component predictions is useful for the current observed evidence, treating every component symmetrically.

Create training targets or reliability weights from component predictions on inner source-held TRAIN data, with the original labels and population weights used only during training. Never train the gate on the components' in-sample errors or held development-source outcomes. Preserve the five-output mixture interface; inference uses observed features and learned weights only. There is no true-family switch, replay-specific threshold, extra branch, or forced replay weight.

Freeze one regularized objective after the diagnostic. Compare reference, trajectory-only, router-only, and their combination if both changes were justified. This is at most four configurations, not a broad hyperparameter search. Keep the external router, feedback, verifier, and specialized architecture unchanged.

If Stage 1 instead identifies false-positive separation as the limiting factor, replace the router candidate with one bounded TRAIN-only hard-negative weighting experiment. Use observed lookalikes and unaffected attack-period readings without changing the dataset or generator. Preserve population weighting and protect every family. Do not add this as an unplanned fifth candidate.

## Stage 3: development evaluation and selection

Use all five existing source-held TRAIN folds and seeds 701–705, with fixed splits and no best-seed selection. These folds have informed earlier development and are not fresh independent evidence. Inner partitions select training settings or operating points; outer held sources must not train any model being scored on them. The general reason for separating selection from evaluation is described in [scikit-learn's nested-validation example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

General-only screening is useful for diagnosis and expense control, but full-system family protection is the decisive criterion. To claim source-held full-system results, fit every learned part of that evaluated system without the held source, including specialized branches, feature fitting, and gates. Do not reuse a full-TRAIN fitted specialist on its own held TRAIN source and call the resulting system score out of fold.

Compare every challenger against the received-history reference fitted under the same partition and seed. Proposed protection gates to freeze before the campaign:

- Pooled F1 must not decrease on average; clean FPR must not increase on average or exceed the existing absolute budget.
- Each other-family mean F1 loss must be at most 0.005, with no source-mean loss above 0.01. Aim for nondecrease, not for spending this tolerance.
- Replay must improve on each source mean and in at least 80% of paired model/source comparisons. Report worst source, worst individual run, and all per-family changes.
- Retain existing source-level pooled/FPR safeguards: pooled loss no more than 0.005 and clean-FPR increase no more than 0.0001 per source. Report within-event false positives separately.
- Rank eligible candidates by worst-family F1, then pooled F1, then lower clean FPR. Do not select on replay alone.

Distinguish a protected incremental improvement from a target-ready design. A candidate still around 0.73 after development is not target ready. A mean near or above 0.80 must also be checked for weak sources; 0.78 is the proposed near-target source floor, with all lower results disclosed. Fix how these readiness requirements apply before the campaign rather than relaxing them after seeing scores.

## Stage 4: full TRAIN, calibration, and one final confirmation

After selecting the representation and training design on TRAIN, fit all five seeds on full TRAIN. Select only permitted thresholds and symmetric router operating settings on calibration, comparing against the reference on the same calibration cases and applying the protection gates. Calibration cannot trigger further feature or training-objective tuning under this campaign.

Advance at most one globally selected, frozen design to fresh confirmation if it is target ready: every mean family F1 at least 0.80, or explicitly near target at 0.78–0.80, with source robustness and protection intact. Freeze which of these two outcomes is being attempted before generating any confirmation data. If the calibration gate fails, record the result and stop this campaign without spending another confirmation set.

For a qualifying candidate, reserve one fresh confirmation campaign using the same source-distribution and scenario-split protocol. Evaluate paired reference and candidate across all five model seeds. Report source-clustered uncertainty, since model seeds on the same scenarios are not independent datasets. Publish all family, pooled, FPR, and source-level results whether or not 0.80 is achieved. A failed confirmation is the result, not permission to continue drawing sources until it passes.

## Invariants and next action

Maintain three top-level branches (General, Drift, Noise), five internal General tree components, and the declared three-hour Drift/Noise decision delay. Keep pressure and flow missingness at 0.50, the attack generator/severity/distribution, existing scenario splits, evaluation labels and units, and the locked test unchanged. No new expert, replay-specific inference rule, future information, alarm holdover, or generator-derived input is permitted.

The next execution step is the TRAIN error-budget audit and a frozen specification for the compact trajectory bank. The router candidate is conditional on evidence from that audit. The plan offers a plausible way to recover recall while preserving the other families; it does not promise that the current observations contain enough information to achieve 0.80.
