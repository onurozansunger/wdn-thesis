# Stage 1 results: a declared decision latency clears 0.80 for both weak families

Date: 2026-09-02. Scope: **expanded TRAIN generator-held OOF only.** Calibration,
development validation, the locked test and the six locked EVAL seeds were not
read. Run: `runs/operational/delayed_decision_head_v1`.

## What was declared before the run

`thesis_v2/ALL_FAMILIES_080_PLAN.md` fixed `delta = 3` hours, the four
candidates and the pass condition before any fitting; the plan's SHA-256 is
recorded in the run's `signature.json`. The decision for hour *t* is finalised
at *t + 3*; nothing reads a label, an event boundary, an attack parameter or
another sensor's future.

## Result

Family-conditioned hindsight F1 over each candidate's score ordering, on the
same 1,298,091 held rows as the promoted seasonal experts.

| Candidate | Drift F1 | Precision | Recall | Noise F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| Frozen expert, delta = 0 | 0.7040 | 0.9130 | 0.5729 | 0.7399 | 0.8374 | 0.6628 |
| **Forward maximum, delta = 3** | **0.8106** | 0.9344 | 0.7157 | **0.8121** | 0.8852 | 0.7501 |
| Delayed-decision head | 0.7989 | 0.9253 | 0.7028 | 0.8263 | 0.8855 | 0.7746 |
| Head blended with forward maximum | 0.8124 | 0.9078 | 0.7351 | 0.8390 | 0.8613 | 0.8177 |

Gate status (clean-FPR allowance, post-event budget and worst-scenario AP all
measured against the promoted seasonal control):

| Candidate | Drift | Noise | Clean FPR (drift / noise) | Post-event FP (drift / noise) |
|---|---|---|---|---|
| Frozen | below 0.80 | below 0.80 | 0.00089 / 0.00097 | 1 / 36 |
| **Forward maximum** | **pass** | **pass** | 0.00090 / 0.00095 | **0 / 21** |
| Head | below 0.80, worst AP fell | post-event FP 158 | 0.00094 / 0.00097 | 1 / 158 |
| Head blend | pass | post-event FP 119 | 0.00095 / 0.00097 | 1 / 119 |

**Stage 1 passes on its cheapest candidate.** Under the declared three-hour
latency, taking the forward maximum of the *existing, unmodified* expert score
lifts drift from 0.7040 to 0.8106 and noise from 0.7399 to 0.8121, while
*improving* both false-alarm figures: post-event false positives fall from 1 to
0 for drift and from 36 to 21 for noise, and clean FPR is unchanged to four
decimals. Early-hour recall, the quantity the whole campaign was stuck on,
rises from 0.0859 to 0.3320 for drift and from 0.3155 to 0.5146 for noise.

No new model was needed. The plan's Stage 2 (drift onset regression) and Stage 3
(noise variance latch) are **not** triggered.

## What the learned heads showed

The delayed-decision head — the seasonal bank plus 59 forward-window features,
same fixed LightGBM recipe, no Optuna — is *worse* than the forward maximum on
drift (0.7989) and better on noise (0.8263), but it buys the noise gain with a
post-event false-positive tail of 158 against a budget of 36. Blending it with
the forward maximum reaches the best raw numbers of the campaign (drift 0.8124,
noise 0.8390) and still fails the noise post-event budget at 119.

That is the same failure mode every earlier latch attempt hit: the head learns
to keep asserting an attack after the event ends. The forward maximum does not,
because it can only extend an alarm by three hours.

A cheaper score-level stage was refuted separately: LightGBM over the frozen
scores' causal history and bounded future is below the frozen score itself at
zero latency (drift 0.6226, noise 0.6669) and never catches the forward maximum
(drift 0.7254, noise 0.6885 at delta = 3). See
`outputs/delayed_decision_screen.json`.

## Limits

- These are **hindsight optima over each score ordering on TRAIN**, the same
  convention as the 0.7040 / 0.7399 control. They are not deployable thresholds,
  not validation, and not test results. Stage 4 selects one operating point on
  calibration; Stage 5 evaluates once on the locked EVAL seeds.
- Only drift and noise were screened here. The plan's replay floor (>= 0.82) and
  the strong families have not yet been measured under the latency.
- `delta = 6` would give more (drift 0.8556, noise 0.8333) and `delta = 4` more
  again than three. **The declared value stays 3.** It was fixed before the run
  and is not revised upward after seeing the result.
- The gain is a property of the declared latency, not of a better detector. Every
  table from here on reports `delta = 0` beside `delta = 3`, and the campaign's
  strictly-online claim remains drift 0.7040 / noise 0.7399.

## A measurement correction

An earlier version of the score-level screen shifted along the sensor series by
*position* rather than by timestep. At a 50% missing rate most series are gappy,
so a positional shift refuses any neighbour whose intervening hours are absent,
and it understated the forward-maximum gain by about 0.04 F1 (it reported drift
0.7450 and noise 0.7831 at delta = 3). Neighbour matching is now by timestep in
`src/wdn/delayed_decision_features.py`, and
`tests/test_delayed_decision_features.py` pins the gap behaviour.

## Reproduction

- `python3 thesis_v2/experiments/screen_delayed_decision_head.py`
  -> `runs/operational/delayed_decision_head_v1`
- `python3 thesis_v2/experiments/screen_delayed_decision.py`
  -> `outputs/delayed_decision_screen.json`
- `python3 thesis_v2/experiments/diagnose_detectability_ceiling.py`
  -> `outputs/detectability_ceiling.json`

Next action recorded by the run: `stage_4_joint_calibration`.
