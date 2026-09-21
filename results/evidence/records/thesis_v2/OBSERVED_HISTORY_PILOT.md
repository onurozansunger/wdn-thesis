# Received-observation consistency pilot

## Hypothesis and architectural change

The previous shared-residual-history pilot did not help replay: mixture F1
changed from 0.7173 to 0.6954 on a held-source TRAIN diagnostic. Its negative
result and frozen artifacts remain unchanged.

This new experiment supplies the same five tree experts and router with
changes between the **actually received** current and past pressures. It does
not use the simulator's original clean readings or uncorrupted replay source.
If a previous reading was itself corrupted, the detector sees that corrupted
reading, exactly as an operational detector would.

The 116 original features are retained. For each lag 1–12, 24, and 48 hours,
the new common bank adds signed pressure change, absolute pressure change,
and an availability flag: **42 new features, 158 total**. Every expert receives
this bank alongside its original profile. The rejected 56 residual-history
features are not added. Existing current reference residuals remain available.

Changes are scaled by the frozen TRAIN reference's noise/error scale and
rounded to 0.1-sigma bins before reaching the model. Nearby values share bins;
there is no exact floating-point equality flag or hand-written replay rule.
The horizon is common to every attack family. Missing history stays missing,
hour offsets are matched by timestamp, and no future observation is read.
Original observations before the cached feature warm-up are valid past context.

## Paired protocol

- L-Town, existing source-held TRAIN fold 0, training seed 701.
- All positives and the same 60,000 sampled negatives as the existing control;
  the same population weights and HGB hyperparameters.
- Reuse the completed 116-feature control instead of retraining it. Verify its
  code, data, and artifact hashes before comparison.
- Even held-source TRAIN scenario IDs select the operating point. Odd IDs
  supply the diagnostic report. These odd scenarios were inspected in the
  earlier pilot, so this is **exploratory development**, not new confirmation.
- Identical selection objective: maximise pooled F1, then worst-family F1,
  then lower clean FPR, on the predeclared budget grid at or below 0.005.
- No canonical split, attack strength, data distribution, or missingness change.
  Both missing probabilities remain 0.50. No EVAL or test set is opened.

Only the five-expert mixture branch is evaluated. Seasonal drift/noise
specialists, warning, feedback, verifier, and deployed outputs stay unchanged.
The pilot's absolute results must not be confused with full-system independent
confirmation, where the prior replay result was 0.6757.

## Predeclared sensitivity check

In addition to the unchanged primary experiment, recompute the **new feature
path only** after rounding received pressures to 0.01 m and 0.05 m. Apply the
same fitted model and primary-selected thresholds; do not retrain or retune.

This checks dependence of the added evidence on numerical precision. These
resolutions are sensitivity settings, not claims about a particular deployed
sensor's specifications. The old 116 features are not recomputed, so this is
**not an end-to-end sensor-quantisation robustness test**. It does not alter or
replace the primary benchmark results.

## Outputs and reproduction

Output directory:
`runs/operational/early_warning_multiseed_v1/observed_history_pilot_v1/ltown/fold_0_seed_701/`.

The frozen protocol includes code/input/control hashes and generator checks.
The report records pooled and per-family F1, precision/recall, clean FPR, and
expert AUPRC on each family's rows plus clean episodes. There is one candidate
training run, not a hyperparameter search. A completed run is not restarted.

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 MPLCONFIGDIR=/tmp/wdn-mpl-cache \
/opt/miniconda3/bin/python -u \
thesis_v2/experiments/early_warning/screen_observed_history.py
```

Before training, 13 focused tests passed, covering timestamp gaps, missingness,
future and other-sensor independence, input reordering, quantisation, invalid
scales, duplicate times, and the unchanged five-expert training objective.

No promotion follows automatically. A worthwhile result needs additional
source-held checks and full-system calibration with other-family protection,
followed by frozen independent confirmation before any new success claim.

## Completed results — 5 September 2026

The candidate completed in 398 seconds. All 22 focused tests passed, including
an additional adapter test for source/scenario isolation and label independence.
The change is retained as a development candidate, **not deployed**.

| TRAIN diagnostic, mixture branch | Original control | Observation history | Change |
|---|---:|---:|---:|
| Pooled F1 | 0.8914 | 0.8938 | +0.0025 |
| Random F1 | 0.9964 | 0.9977 | +0.0014 |
| Replay F1 | 0.7173 | 0.7432 | +0.0259 |
| Drift F1 | 0.7917 | 0.7884 | -0.0033 |
| Noise F1 | 0.7936 | 0.7938 | +0.0003 |
| Targeted F1 | 0.9917 | 0.9921 | +0.0004 |
| Clean FPR | 0.000292 | 0.000271 | -0.000021 |

Replay mixture precision/recall increased from 0.8676/0.6114 to 0.8913/0.6373.
This means five additional true detections and three fewer false positives
within replay events. Its family-plus-clean AUPRC rose from 0.5861 to 0.6394.

The standalone replay expert increased from F1 0.7492 to 0.7826 and AUPRC
0.5930 to 0.6544. Its recall is still only 0.6528. It is not a substitute for
the full detector: that expert's pooled F1 across all families fell from
0.8851 to 0.8718. All expert-by-family metrics are retained in `summary.json`.

| New-feature-path sensitivity, frozen model and thresholds | Pooled F1 | Replay mixture F1 | Replay expert F1 |
|---|---:|---:|---:|
| Primary received readings | 0.8938 | 0.7432 | 0.7826 |
| Readings rounded to 0.01 m for added features | 0.8937 | 0.7432 | 0.7826 |
| Readings rounded to 0.05 m for added features | 0.8925 | 0.7410 | 0.7778 |

The measured gain therefore persists in this limited feature-path sensitivity
check. This does not establish robustness to every sensor precision, replay
implementation, or network, nor end-to-end quantisation robustness.

### Sample-size limits and next gate

There are only **two replay events in two scenarios**, comprising 193 positive
sensor-hours, in this diagnostic subset. Individual event F1 rose from
0.6863 to 0.7150 and from 0.7680 to 0.7903. Improvement is present in both, but
two events and one model/source fold are insufficient for a generalised claim.
These scenarios were already inspected during the earlier feature pilot.
`diagnostic_support.json` records the per-family event counts and paired event
confusion counts.

This is a useful but modest gain, not the requested performance jump. The
mixture remains 0.0568 below 0.80 on this TRAIN diagnostic. The standalone
replay expert's 0.7826 does not mean the deployed system has reached that score.
The full-system independent replay result remains **0.6757**, unchanged.

Next gate: repeat the fixed recipe across other source-held TRAIN folds before
any full-system promotion. If the gain persists, fit on full TRAIN and assess
the complete mixture/seasonal/feedback/verifier combination on calibration,
protecting other families and the clean-FPR budget. Freeze the resulting recipe
before fresh independent confirmation. Do not tune repeatedly on locked EVAL.
