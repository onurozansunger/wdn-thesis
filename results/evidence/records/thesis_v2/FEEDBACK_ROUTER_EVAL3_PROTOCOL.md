# Feedback-guided evidence router: TRAIN/calibration protocol and EVAL-3 gate

Date: 2026-09-03. This protocol is written before fitting the new router or
feedback heads and before generating or reading EVAL-3.

## Fixed scope

- The operational benchmark is unchanged. Pressure and flow missing
  probabilities remain 0.50; attack severities, durations, ramps, replay lags,
  network, timestep and split logic remain unchanged.
- Existing EVAL-1 and EVAL-2 are diagnostic history only. They may motivate the
  architecture, but they are not used to fit or select the new candidate.
- The original locked test remains unopened.
- Development uses the 62-scenario generator-held TRAIN OOF predictions and the
  existing 99-scenario calibration corpus.
- If the calibration gate passes, EVAL-3 uses six new generator seeds:
  `30811, 31811, 32811, 33811, 34811, 35811` (144 scenarios total).

## Architecture

The frozen deployed detector remains the safety baseline. A graph-time evidence
router ranks five states: clean, abrupt, replay, drift and noise. It aggregates
only observable feature-bank and expert-score evidence across observed sensors
at the same `(source, scenario, timestep)`; labels or event boundaries never
enter inference.

Two expert-specific feedback heads judge whether the routed drift or noise
candidate is locally consistent with its evidence. The router and feedback
scores continuously re-rank the existing drift/noise branch scores. They do not
hard-disable the general mixture branch. Final alarms still use three separately
calibrated branches (mixture, drift, noise) under one clean-FPR budget.

This is an explicit `router -> candidate expert -> feedback -> reweight`
mechanism with a permanently available general fallback. It is not the earlier
generic reconstruction/physics cascade, which was empirically unreliable.

## Fixed candidate family

- Router strength: `{0.15, 0.30, 0.50}`.
- Feedback strength: `{0.15, 0.30, 0.50, 0.75}`.
- Both strengths must be positive in the promoted candidate.
- Mixture latency remains zero; drift/noise decision latency remains three
  hours.
- Final decision remains a budgeted OR. Candidate clean-FPR shares are searched
  only over mixture `{0.05, 0.10, 0.15}`, drift `{0.10, 0.15, 0.20, 0.25}` and
  the positive remainder for noise.

The adjusted branch logit is the frozen branch logit plus the weighted router
margin against replay and the weighted feedback logit. This directly tests the
measured failure mode in which the noise specialist fires on replay evidence.

## Calibration promotion gate

Relative to the existing frozen EVAL-2 selection evaluated on the same
99-scenario calibration corpus, the candidate must satisfy all of:

1. clean-period FPR `<= 0.005`;
2. random F1 `>= 0.90`, targeted F1 `>= 0.90`, replay F1 `>= 0.82`;
3. drift and noise F1 each `>= 0.81`;
4. worst-family F1 no lower than the frozen baseline;
5. overall pooled F1 no lower than the frozen baseline minus 0.005;
6. both router and feedback have non-zero influence.

If no candidate passes, EVAL-3 is not generated and the negative result is
reported. If one passes, its bundle, formula, thresholds, source hashes and
baseline are frozen before EVAL-3 generation.

## EVAL-3 reporting

Both the frozen old baseline and the frozen feedback-router candidate are
scored in the same one-time EVAL-3 pass. Report per-family F1, cluster-bootstrap
95% intervals, pooled F1, clean FPR, drift/noise expert F1, router accuracy,
feedback acceptance diagnostics and replay false-positive decomposition. A
point estimate below 0.80 remains below target even if its confidence interval
crosses 0.80.
