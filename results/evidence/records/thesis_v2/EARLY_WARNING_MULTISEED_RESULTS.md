# Early warning, pooled F1, L-Town and repeated runs — results

**Campaign:** `early_warning_multiseed_v1`
**Protocol:** `thesis_v2/EARLY_WARNING_MULTISEED_PROTOCOL.md` (preregistered 4 Sep 2026)
**Date of this report:** 4 September 2026
**Interpreter:** `/opt/miniconda3/bin/python` — Python 3.13.5, NumPy 2.3.1,
scikit-learn 1.8.0, LightGBM 4.7.0, WNTR 1.4.0, PyTorch 2.11.0

> **Completion status.** Stages A–D are measured and complete. **Stage E is
> complete for Modena and running for L-Town.** The headline result is
> confirmed: on six fresh generator seeds that no model or operating point ever
> saw, the candidate reaches pooled F1 **0.8672 ± 0.0026** against the deployed
> baseline's **0.6863 ± 0.0069**, with the paired difference positive in all
> thirty (training seed x data seed) cells. Sections 2–5 remain
> development-surface measurements and are labelled as such; section 7.1 is the
> confirmation. On L-Town the deployed baseline recipe turned out to have **no
> feasible operating point**, which is itself a reported result (section 7.2).

The original locked test (`splits.json` test scenarios `[0, 1, 8, 20]`) was not
opened. EVAL-1 (`4811`–`9811`), EVAL-2 (`20811`–`25811`) and EVAL-3
(`30811`–`35811`) were not re-scored; their recorded reports were read only to
explain past failures.

---

## 1. What was asked, and what the evidence says so far

| Objective | Status |
|---|---|
| O1 — reduce clean-period false alarms without losing family coverage | **Met and confirmed on Modena.** Fresh evaluation: pooled F1 0.6863 → **0.8672**, false alarms −84.2%, every family gate held, paired gain positive on all 30 cells. |
| O2 — a preliminary drift/noise warning before three hours | **Met for noise, not met for drift.** Noise: 90% of events warned with the correct family by onset +1 h. Drift: 13.6% by +1 h, 31.8% by +2 h. |
| O3 — run the same method on L-Town | **Partly answered, negatively so far.** The method runs, but the deployed baseline recipe has no feasible operating point on L-Town. Calibration under the declared fallback is running; no L-Town detection result exists yet. |
| O4 — five paired training seeds, not a best run | **Met on Modena.** Five training seeds, every individual run, paired differences and a cluster-aware uncertainty statement are reported in section 7.1. L-Town pending. |

The single most useful finding is Stage A's, and it contradicts the concern the
handoff raised: the pooled-F1 deficit does **not** live in the protected general
branch. It lives almost entirely in the noise specialist.

---

## 2. Stage A — where the false alarms actually come from

Artifacts: `runs/operational/early_warning_multiseed_v1/stage_a_audit/`
(`false_alarm_audit.json`, two CSVs), figures
`thesis_v2/outputs/early_warning_multiseed/stage_a_false_alarm_audit.png` and
`…_covariates.png`.

**Surface:** 99 calibration scenarios, 2,074,561 observed pressure endpoints,
11,936 positives. The audit reproduces the deployed decision exactly — its
pooled F1 of 0.677156 matches `feedback_router_veto_v1/selection_frozen.json`
to six decimals, which is the check that the reconstruction is the real
deployed rule and not an approximation of it.

**Caveat carried on every number in this section:** the models never saw these
rows, but the deployed thresholds were *chosen* on them, so these false-positive
counts are optimistically biased.

### 2.1 Branch attribution

| Branch | Raises a false alarm | Raises it **alone** |
|---|---:|---:|
| General mixture | 586 | **139** |
| Drift specialist | 2,033 | 1,025 |
| Noise specialist | 7,519 | **6,164** |

Total false alarms 8,705, partitioned exactly (the audit asserts the partition).
Overlaps: drift+noise 930, general+noise 369, general+drift 22, all three 56.

**70.8% of all false alarms are raised by the noise specialist and by nothing
else.** The protected general branch is responsible for 1.6% on its own. The
handoff's contingency — "if most clean FP originate in the protected general
branch, quantify the limit of specialist-only suppression" — does not apply.
Specialist-only suppression is not merely permissible here, it is the whole
problem.

### 2.2 Where they fall

- **By period:** 7,721 of 8,705 (88.7%) fall in clean periods. The recorded
  EVAL-3 figure was 87.9%, so calibration and the locked EVAL agree on the
  shape of the problem.
- **Post-event tails:** 18.8% of all false alarms fall in the six hours after
  an event closes. Tail rates are highest after drift (2.05%) and noise (1.96%)
  events, lowest after replay (0.35%).
- **Duration:** 3,880 false alarms (44.6%) sit in persistent runs of four or
  more alarmed endpoints in one sensor series; 3,185 (36.6%) in runs of 2–3;
  only 1,640 (18.8%) are isolated. Of 6,155 alarm runs, 3,253 are entirely
  false, with a median length of 1 endpoint and a 90th percentile of 4.
- **Concentration:** none. All 272 sensors and all 99 scenarios produce false
  alarms; the worst 10% of sensors carry only 30.8% of them. There is no small
  set of bad sensors to exclude, and excluding one would violate the metric
  scope anyway.

### 2.3 Which covariates predict a false alarm

False-alarm rate over negatives, lowest to highest decile:

| Covariate | Range | Ratio |
|---|---|---:|
| `seq_seen` (length of causal history) | 0.00021 → 0.00499 | 26× |
| `abs_residual` | 0.00214 → 0.01804 | 8.5× |
| hour of day | 0.00165 → 0.00869 | 5.3× |
| `last_gap` (hours since last observation) | 0.0 → 0.00444 | large, but the lowest bin is degenerate |
| `normal_error_scale` | 0.00318 → 0.00581 | 1.8× |
| `reference_support` | 0.00513 → 0.00367 | 1.4×, and **decreasing** |

Reference support is nearly flat and points the *wrong* way: false alarms are
not caused by thin reference support. The strong signals are history length,
residual size and time of day — a diurnal structure the deployed detector does
not model explicitly. This is what made a learned verifier the obvious Stage C
candidate rather than a support-based filter.

### 2.4 The existing replay guard

On calibration the deployed router/feedback veto marks 117,720 rows, removes 432
alarms, and among those removes 12 true alarms and only **38 clean-period false
alarms**. The recorded EVAL-3 figure was 81 clean-period removals out of 664
replay-period removals. The guard does what it was built for — replay-period
precision — and, exactly as the handoff warned, strengthening it cannot close
the pooled-F1 gap.

### 2.5 TRAIN out-of-fold, specialists only

**Provenance, checked rather than assumed:** each of the four
generator-source-held folds in `expanded_train_weak_experts_v1/fold_*/` fitted
its own normal reference, so these scores are full-pipeline out-of-fold, not
fixed-reference conditional. There is **no** matching out-of-fold artifact for
the general mixture on that corpus, so this surface attributes nothing to the
general branch.

Specialists alone, thresholds re-derived on this surface by the deployed
quantile procedure at the deployed budget shares: 5,662 false alarms, of which
4,992 (88.2%) in clean periods — the same 88% clean share as calibration and
EVAL-3. Drift family F1 0.786, noise 0.832. Persistent runs again dominate
(2,376 of 5,662).

---

## 3. Stage B — the causal early-warning head

Artifacts: `runs/operational/early_warning_multiseed_v1/early_head_v1/`
(`bundle.joblib`, `summary.json`, `train_oof_predictions.npz`), figure
`thesis_v2/outputs/early_warning_multiseed/stage_b_early_warning.png`.
Code: `src/wdn/early_warning.py`, `src/wdn/early_warning_metrics.py`.

**Why a new input path was needed.** The deployed evidence router consumes
`maxpool_*`, `delayed_*` and `final_*` scores, every one of which reads
observations inside `(t, t+3]`. It cannot be relabelled an early router. The new
head reads only the 37 causal bank channels plus the two *causal* specialist
scores, aggregated by mean/std/max per graph-time hour, with past-only
differences at lags 1, 2 and 3 hours matched by timestep rather than by row
position. 160 features, 9,548 graph-time rows, a single small gradient-boosted
tree. No neural search: a standalone GRU already failed the prior OOF screen.

**Declared interpretation, because the user did not specify one:** this is a
**network-level family warning** at graph-time resolution, followed by
specialist sensor localisation and confirmation. It is not sensor localisation
and no localisation accuracy is claimed from it.

**Surface:** 62 TRAIN scenarios, four generator-source-held folds; the head
predicting a source never saw that source. 42 drift/noise events — a small
sample, and the reason no confidence interval is quoted as if it were tight.

### 3.1 Warning performance on the onset clock

Fractions are over **all** events, with misses in the denominator.

| | drift (22 events) | noise (20 events) |
|---|---:|---:|
| any warning by onset +0 h | 0.000 | 0.750 |
| any warning by onset +1 h | 0.136 | 0.900 |
| any warning by onset +2 h | 0.318 | 0.950 |
| **correct family** by onset +1 h | 0.045 | **0.900** |
| **correct family** by onset +2 h | 0.182 | **0.950** |
| ever warned | 0.909 | 1.000 |
| ever warned, correct family | 0.909 | 1.000 |
| first-warning delay, median (warned events) | 3.0 h | 0.0 h |
| first-warning delay, mean | 2.8 h | 0.4 h |
| events censored as never warned | 2 | 0 |

**Noise meets the supervisor's request.** 90% of noise events carry a
correct-family warning within one hour of the true onset, and the median
first-warning delay is zero hours — the warning arrives in the onset hour
itself.

**Drift does not, and this is a structural negative result, not a tuning
failure.** The benchmark ramps drift over 6–18 hours (`drift_ramp_hours`), so at
onset the injected displacement is near zero by construction. There is nothing
to warn about yet. Drift is warned eventually (90.9% of events, median 3 h), but
"before three hours" is met for only a third of drift events. Reporting this as
a success would require changing the deadline or the benchmark, and neither is
permitted.

### 3.2 False warnings and abstention

Over clean network-hours: 50 false-warning hours out of 8,395 (0.60%). Excluding
a six-hour post-event tail: 30 out of 7,874 (0.38%). Warning episodes: 154,
covering 1,109 warned graph-hours.

**Abstention never activated.** The threshold was chosen on TRAIN out-of-fold as
the smallest value in a frozen grid whose clean-hour false-warning rate stayed
within the 5% budget; the budget was already met at the grid's lowest value
(0.30), and with five classes the top probability essentially always exceeds
0.30. The abstain column of the confusion matrix is zero everywhere. The
mechanism exists and is tested, but it did no work on this surface, and it
should not be described as if it did.

### 3.3 Graph-time confusion (rows are the true state)

| true \ predicted | normal | other mechanism | likely drift | likely noise | abstain |
|---|---:|---:|---:|---:|---:|
| clean (8,395) | 8,345 | 13 | 14 | 23 | 0 |
| abrupt (440) | 0 | 418 | 13 | 9 | 0 |
| replay (260) | 5 | 252 | 0 | 3 | 0 |
| drift (202) | 60 | 33 | 92 | 17 | 0 |
| noise (251) | 29 | 13 | 25 | 184 | 0 |

Replay and abrupt are represented as competing explanations and are recovered as
such (418/440 and 252/260); neither is folded into drift or noise. Drift is the
weak row, as section 3.1 predicts: 60 of 202 drift hours are called normal.

---

## 4. Stage C — the bounded three-candidate screen

Artifacts: `runs/operational/early_warning_multiseed_v1/verifier_screen_v1/`
(`selection_frozen.json`, `bundle.joblib`), figure
`thesis_v2/outputs/early_warning_multiseed/stage_c_candidate_screen.png`.
Code: `src/wdn/alarm_verifier.py`.

Candidates were frozen in the protocol before any was fitted. The verifiers were
fitted on TRAIN out-of-fold only; selection ran on calibration, optimising
**pooled F1** subject to the frozen gates (drift/noise ≥ 0.80, random/replay/
targeted ≥ 0.90, clean FPR ≤ 0.005, no family worse than the paired baseline by
more than 0.01).

| | pooled F1 | FP | clean-period FP | clean FPR | random | replay | drift | noise | targeted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (deployed) | 0.6772 | 8,705 | 7,721 | 0.00425 | 0.9592 | 0.9319 | 0.8348 | 0.8172 | 0.9457 |
| **C1** threshold-only | 0.6836 | 8,468 | 7,434 | 0.00410 | 0.9575 | 0.9298 | 0.8422 | 0.8108 | 0.9460 |
| **C2** verifier | 0.8032 | 3,328 | 2,937 | 0.00162 | 0.9710 | 0.9369 | 0.8386 | 0.8084 | 0.9652 |
| **C3** verifier + early signal | **0.8551** | **1,787** | 1,394 | 0.00077 | 0.9746 | 0.9367 | 0.8324 | 0.8087 | 0.9707 |

C3 was selected. All gates held: no family fell more than 0.0085 below the
paired baseline (noise, the largest drop), and random, replay and targeted all
improved.

### 4.1 What each candidate shows

- **C1 is the honest control and it barely moves.** Re-spending the same 0.005
  clean-FPR budget across branches buys +0.006 pooled F1. The problem is not
  where the thresholds sit.
- **C2 is most of the gain.** A per-branch verifier over causal and
  bounded-future evidence removes 5,377 false alarms (62%) while giving up 322
  true positives, for +0.126 pooled F1.
- **C3 adds a further +0.052** by feeding the early-warning trajectory into the
  verifier — the graph-time class probabilities at `t`, `t−1` and `t−2`, the
  confidence, and how many consecutive hours the warning has persisted. This is
  the campaign's clearest architectural result: **the early-warning head is not
  only a supervisor deliverable, it measurably improves the confirmed decision.**

### 4.2 What this result is not

It is a **selection** on calibration: 5,508 operating points were searched on
the same 99 scenarios the number is quoted on. The verifier and early head never
saw those rows, but the operating point did. The honest reading is that C3 is
the candidate worth confirming, not that pooled F1 is 0.855. Stage E's fresh
corpora are the confirmation, and until they report, **0.8551 must not be quoted
as a result.**

Two further caveats:

- The verifier was trained where specialist scores are out-of-fold and applied
  where they come from the deployed 110-scenario fit. That shift makes C3
  conservative if anything, but it is a shift.
- The verifier is strictly one-directional (unit-tested): it can only remove a
  candidate specialist alarm, never create one, and it is never applied to the
  general mixture. The existing replay guard is retained unchanged and composed
  once; the verifier is not a second replay guard.

### 4.3 Warning-to-confirmation behaviour, and what the warning does not do

Artifact: `runs/operational/early_warning_multiseed_v1/warning_to_confirmation_v1/warning_to_confirmation.json`.
Calibration, selected candidate rule, all timestamps written down per event.

The two clocks are kept separate: a warning is stamped at the graph-time hour it
is issued, while the decision for hour `t` only becomes available at `t + 3`.

| | |
|---|---:|
| drift/noise events traced | 66 |
| events that got a warning | 66 |
| events that got a confirmation | 66 |
| **events warned before the confirmation was available** | **57 (86.4%)** |
| median warning lead over confirmation | **2.0 h** |
| mean warning lead | 1.94 h |

Warning episodes: 242 in total, median 7 hours long. 229 were followed by a
confirmation inside the episode plus the three-hour latency; **13 closed as
cancellations with no confirmation**. 202 episodes overlap a real event and 40
sit entirely on clean hours — those 40 are the operator-visible cost of the
warning channel, and they are reported here rather than folded into the
confirmed-alarm numbers.

**Early localisation is not provided, and here is the evidence for saying so.**
The head is network-level by declaration. As a separate sensor-level metric, the
*causal* specialist recall at the same deadlines, thresholded at the same 0.5%
clean budget:

| | onset +0 h | onset +1 h | onset +2 h |
|---|---:|---:|---:|
| drift | 0.030 | 0.125 | 0.232 |
| noise | 0.333 | 0.374 | 0.410 |

Noise reaches 37% of attacked endpoints at onset +1 h, against 90% of *events*
warned at network level. That gap is the point: aggregating across sensors is
what makes an early warning possible at all, and network-level family accuracy
must never be quoted as localisation accuracy.

---

## 5. Stage D — L-Town pilot

Artifact: `runs/operational/early_warning_multiseed_v1/ltown_pilot_v1/pilot_report.json`.
Pilot corpus: `data/thesis_v2/ew_ltown_ltown_pilot_seed91811` (10 scenarios,
seed 91811, generated for this campaign; the old `v2_ltown`, `hard_ltown`,
`rec_hard_ltown` and episode datasets were **not** reused).

### 5.1 Correctness

| Check | Result |
|---|---|
| Sensors | 785 (Modena: 272) |
| Reference groups | 197 / 196 / 196 / 196 |
| Realised pressure missing rate | 0.5007 (configured 0.50) |
| Minimum normal observations per sensor | 697 (reference requires 2) |
| Sensors below the reference minimum | 0 |
| Contiguous hourly time for seasonal history | yes |
| **Target-group blindness probe** | perturbing one sensor by +25 m moved its own prediction by **exactly 0.0**, and its whole group by 0.0, while other groups moved by up to 0.092 m |

The blindness probe is the important one: it is a behavioural check that the
785-sensor reference still excludes a sensor's complete group from its own
prediction, rather than an assumption inherited from Modena.

### 5.2 Resources, measured

| Quantity | Measured on 10 L-Town scenarios |
|---|---:|
| Generation | 3.1 s, 56 MB for 10 scenarios |
| Normal reference fit (1,511 normal rows, 785 sensors, rank 16) | 6.1 s |
| Feature bank (116 columns) | 34.7 s, i.e. 3.47 s/scenario |
| Rows per scenario | 60,401 (Modena: 20,942 — **2.88×**) |
| Compressed cache | 19.7 MB/scenario |
| Peak resident memory | 3.06 GB |

Extrapolated to the planned budget (L-Town TRAIN 120, calibration 96, evaluation
144 scenarios): 7.2M / 5.8M / 8.7M rows, 2.20 / 1.76 / 2.64 GB of cache, and 7 /
6 / 8 minutes of feature building. Disk free at the time of measurement: 165 GB.
**L-Town is not resource-limited on this machine**, so no smaller budget needs to
be requested. The reference fit is superlinear in rows and will be re-measured on
the first full L-Town TRAIN corpus rather than scaled from the pilot.

### 5.3 The full L-Town corpora and their own reference

Generated for this campaign under the frozen benchmark, seeds reserved and
collision-checked before generation:

| Purpose | Seeds | Corpora × scenarios |
|---|---|---|
| L-Town TRAIN | 60811–64811 | 5 × 24 = **120** |
| L-Town calibration | 70811–73811 | 4 × 24 = **96** |
| L-Town fresh evaluation (reserved, not yet generated) | 50811–55811 | 6 × 24 = 144 |

The **L-Town normal reference was fitted on L-Town TRAIN only**: 18,020 normal
rows over 785 sensors, rank 16, four target-excluded groups of 197/196/196/196,
median noise scale 0.101 m. It took 179 s at a 2.69 GB peak — comfortably inside
budget, and measured rather than extrapolated as section 5.2 promised. Modena's
reference file was never opened on this path.

The pilot's extrapolation has since been checked against the real thing, on both
corpora: it predicted 7,248,108 TRAIN rows and 5,798,486 calibration rows, and
the built caches hold **7,253,913** and **5,803,001** — 0.08% out in both cases.
The TRAIN bank took 772 s to build, the calibration bank 446 s.

### 5.4 Budget comparability

L-Town uses five 24-scenario TRAIN corpora (120) and four 24-scenario
calibration corpora (96), against Modena's 110 and 99. Whole corpora are
assigned entirely to TRAIN or to calibration, exactly as on Modena; no partial
split was fabricated to hit an exact count. The small difference (120 vs 110,
96 vs 99) is reported rather than engineered away.

A separate point that will matter when L-Town results arrive: identical physical
perturbations do not imply identical standardised difficulty on a different
network. Any Modena/L-Town gap will be reported as a difficulty difference, not
corrected by adjusting attacks.

---

## 6. Verification

`tests/test_early_warning.py` (22 tests) and
`tests/test_early_warning_campaign.py` (17 tests) — **39 new tests, all
passing**. Full suite: **115 passing** (76 pre-existing + 39 new).

What is actually asserted:

- Perturbing every observation after `t` leaves every early-warning feature at
  hours ≤ `t` bitwise unchanged — and the test also asserts the later hours *did*
  move, so it cannot pass vacuously.
- No `maxpool_*`, `delayed_*` or `final_*` channel is reachable from the early
  head's input list.
- Perturbing observations after `t+3` leaves the confirmed score at `t`
  unchanged; forward pooling does not cross a scenario boundary.
- Lags are matched by hour, not by row position: hour 5 does not adopt hour 1 as
  its lag-1 predecessor when hour 4 is missing, and a missing hour flips
  `has_lag1` to zero instead of shifting the window.
- Early features are invariant to relabelling every row's label and family; no
  feature name encodes a label, onset, event, family or target identity.
- The verifier never creates an alarm, an accepting verifier reproduces the
  baseline decision exactly, and every general-mixture alarm survives the
  strictest possible verifier.
- Missed events stay in the denominator as censored delays; a wrong-family
  warning is not counted as correct; an abstention is not a warning; clean-hour
  denominators exclude open events; episodes group only consecutive hours.
- The feature cache refuses an incompatible cached configuration instead of
  overwriting it, and `contiguous_groups` rejects a split graph-time block.

The campaign-level file additionally asserts, against the frozen artifacts
themselves: that TRAIN and calibration scenario identities are disjoint and that
neither touches the locked test or validation scenarios; that every generated
corpus fixes both missing probabilities at 0.50 and the rest of the observation
process; that the distribution check actually rejects a changed
`attack_fraction` or missing rate; that no L-Town corpus points at Modena's
reference and that the L-Town reference was fitted on L-Town TRAIN; that the
audit's branch attribution partitions the false alarms exactly and every
reported F1 reproduces from its own recorded TP/FP/FN; that the Stage A audit
reproduces the recorded deployed operating point to nine decimals; that the
selected candidate really is the best gate-passing pooled F1; that the warning
metrics use event denominators and that confirmation timestamps are the decision
hour plus the declared latency; and that every campaign report still records
`test_evaluated: false` for the locked corpora.

---

## 7. Stage E — Modena confirmed on fresh data; L-Town still running

### 7.1 Modena: complete

Five training seeds (701–705) x six fresh generator seeds (40811–45811) x 24
scenarios = **144 scenarios and ~3,015,700 observed pressure endpoints per
training seed**, the same size as the recorded EVAL-3 corpus. These seeds were
reserved and collision-checked before generation, and every seed's operating
point was frozen in `operating_points.json` before any evaluation corpus was
scored.

Artifacts: `runs/operational/early_warning_multiseed_v1/stage_e_modena/seed/70*/`
(`oof_scores.npz`, `base_bundle.joblib`, `delayed_bundle.joblib`,
`heads_bundle.joblib`, `operating_points.json`, `evaluation_report.json`).
Figure: `thesis_v2/outputs/early_warning_multiseed/stage_e_seed_variability.png`.

**Pooled sensor-endpoint F1, mean over the six fresh data seeds, then over the
five training seeds:**

| Arm | Pooled F1 | sd across training seeds | False positives | Clean FPR |
|---|---:|---:|---:|---:|
| baseline (deployed guarded hybrid) | 0.6863 | 0.0069 | 12,544 | 0.00426 |
| threshold-only (C1) | 0.6998 | 0.0038 | 11,801 | 0.00400 |
| **candidate (C3)** | **0.8672** | **0.0026** | **1,982** | **0.00057** |

Per training seed, candidate: 0.8684, 0.8709, 0.8643, 0.8672, 0.8653.

**Paired candidate − baseline: +0.1809, sd 0.0080 across training seeds.**
Across all thirty (training seed x data seed) cells the paired difference is
positive every time, ranging **+0.1410 to +0.2053**. There is no seed at which
the candidate loses.

Counts summed over all thirty cells: true positives 79,809 → 76,532 (−4.1%),
false positives 62,722 → **9,910 (−84.2%)**, false negatives 10,156 → 13,433.
The candidate buys an 84% cut in false alarms for a 4% cut in true positives.

**Family F1, mean over all thirty cells:**

| | random | replay | drift | noise | targeted |
|---|---:|---:|---:|---:|---:|
| baseline | 0.9513 | 0.9030 | 0.8333 | 0.8506 | 0.9605 |
| candidate | 0.9542 | **0.9280** | 0.8262 | 0.8455 | 0.9674 |
| difference | +0.0029 | **+0.0250** | **−0.0071** | **−0.0051** | +0.0069 |

**Where the cost falls, and how consistently.** The two losses are drift and
noise, and they are systematic rather than noise: drift is worse in 22 of the 30
cells and noise in 25 of 30. Both averages sit inside the preregistered 0.01
tolerance, and per training seed (averaging the six data seeds) the drift loss
ranges −0.0009 to −0.0093 and the noise loss −0.0004 to −0.0086, so no training
seed breaches it either. Replay is the clear gain, +0.0250, worse in only 1 cell
of 30.

**At the level of a single cell the picture is less clean, and the gates were
never defined there.** The preregistered floors are *selection* gates, applied
to calibration, not per-data-seed evaluation gates. Reported honestly, the worst
individual cells are:

| Family | Worst single-cell difference | Lowest single-cell candidate F1 | Floor |
|---|---:|---:|---:|
| drift | −0.0253 | **0.7689** | 0.80 |
| noise | −0.0222 | **0.7844** | 0.80 |
| replay | −0.0003 | **0.8993** | 0.90 |
| random | −0.0154 | 0.9366 | 0.90 |
| targeted | −0.0089 | 0.9394 | 0.90 |

So on individual generator seeds the candidate does dip below the drift, noise
and replay floors, and on some cells the drift and noise losses exceed the 0.01
tolerance. The means pass; every cell does not. Anyone quoting "all gates held"
should say it of the per-network means, which is the level at which the gates
were preregistered.

**Uncertainty, cluster-aware as the protocol requires.** Variation across the
six *data* seeds is about three times the variation across the five *training*
seeds (baseline sd 0.0213 vs 0.0069; candidate 0.0120 vs 0.0026). Five training
seeds x six data seeds are therefore **not** thirty independent corpora, and the
0.0080 quoted for the paired difference is the spread across training seeds, not
a standard error over thirty samples.

**This confirms the Stage C selection.** Calibration said 0.8551; fresh,
never-before-scored corpora say 0.8672. The verifier and the early-warning head
were fitted on TRAIN out-of-fold, the operating points were frozen on
calibration, and neither ever saw these six generator seeds.

### 7.2 L-Town: the deployed baseline recipe is infeasible there

The L-Town arm completed `folds`, `oof`, `fit` and `heads` for all five training
seeds, then **failed at calibration**: no operating point in the budgeted-OR
share grid satisfies the deployed recipe's own constraints on L-Town
(clean FPR ≤ 0.005 *and* replay ≥ 0.82 *and* random ≥ 0.90 *and* targeted ≥ 0.90).

This is a result about the network, not a bug: the same architecture, the same
generator settings and the same calibration procedure produce a constraint set
that Modena can satisfy and L-Town cannot. Section 5.4's warning was that equal
physical perturbations need not mean equal standardised difficulty; this is that
warning coming true.

Under the dated amendment in the protocol's section 11 — declared **before any
L-Town evaluation corpus was generated or scored**, and applied to every arm
identically — the runner now:

1. records `constraint_frontier`: for each constraint, the best value reachable
   anywhere in the grid, the best reachable within the clean-FPR budget, and
   which constraints are unreachable;
2. falls back, for baseline *and* candidates alike, to maximising each arm's own
   objective subject to the **0.005 clean-FPR budget alone**, with the deployed
   family floors reported rather than enforced;
3. keeps the **relative** protection unchanged — no candidate may leave a family
   more than 0.01 below its paired baseline.

Modena's results are untouched and keep the original enforced gates. Every
L-Town number will be labelled as produced under this fallback.

**Status: running.** Relaunched after the fix; it resumed from the cached folds,
OOF scores, fits and heads and is now in `calibrate`. Its fresh evaluation seeds
(50811–55811) have not been generated, so no L-Town detection number exists yet.

### Exact next commands

Modena is complete and needs nothing. For L-Town:

```bash
/opt/miniconda3/bin/python thesis_v2/experiments/early_warning/run_campaign.py --network ltown --stage all
```

Log: `runs/operational/early_warning_multiseed_v1/stage_e_ltown.log`.
Live status: `runs/operational/early_warning_multiseed_v1/stage_e_ltown/status.json`.

To redraw every figure from whatever is finished:

```bash
/opt/miniconda3/bin/python thesis_v2/experiments/early_warning/make_figures.py
```

Every stage is resumable, skips completed work, and refuses an incompatible
cached configuration rather than overwriting it. The Modena run demonstrated
this in practice: it crashed mid-campaign on a missing `early` flag, was
relaunched by its supervisor script, skipped the six completed folds and
finished.

### Corrections made during Stage E

Recorded here rather than hidden. The first three were found and fixed before
the stage that consumes them ran, and were validated end to end on a
three-source slice of fold 0 (17 distinct event ids across three sources, 8,390
veto rows):

1. **Missing `early` flag.** The expert weighting upweights the opening three
   hours of each event; the campaign's feature caches do not store that flag. It
   is now derived from the event ledger inside the runner (`add_early_flag`) —
   supervised-training metadata, never a feature.
2. **Baseline missing its guard.** The baseline arm is the *guarded* hybrid, so
   the router and feedback heads are now refitted per training seed to match that
   seed's experts instead of being borrowed from the recorded deployment. The
   guard's `(margin, cutoff) = (0.10, 0.10)` rule stays fixed at the deployed
   value, so the recipe is repeated rather than re-selected.
3. **Colliding event ids.** Each corpus directory numbers its events from zero,
   so concatenating two generator seeds merged event 0 of one with event 0 of the
   other. `training_weights` equalises risk per `(family, event)` group, so that
   merge would have quietly reweighted the training set. Event ids are now made
   unique by source at load time (`globalise_events`). Nothing in Stages A–C
   reads event ids, so none of those results is affected.
4. **Infeasible baseline aborted the run.** `stage_calibrate` raised instead of
   recording the infeasibility, which would have discarded four completed L-Town
   stages. It now measures the constraint frontier and continues under the
   declared fallback (section 7.2).

## 8. Honest summary

### Stage F addendum — general router recalibration on L-Town

A post-Stage-E experiment changed the fusion mechanism without adding an
expert or introducing a replay-specific inference rule. Symmetric temperature
and uniform shrinkage were selected for the existing router on calibration,
then frozen before six new L-Town generator seeds were created. On 30 paired
fresh comparisons, replay F1 increased from 0.6461 to 0.6757 and pooled F1 from
0.8799 to 0.8841; both improved in all 30 pairs. Random, drift, noise and
targeted mean F1 changed by -0.0008, -0.0012, -0.0016 and -0.0016 respectively,
all within the predeclared 0.01 protection tolerance. Clean FPR decreased from
0.000595 to 0.000583. Replay nevertheless remained below 0.80 in every pair;
the router-only change is a robust refinement, not a solution to L-Town replay.
See `thesis_v2/ROUTER_RECALIBRATION_RESULTS.md` for the protocol and limitations.

**Established, on development surfaces:**

1. The pooled-F1 deficit is a noise-specialist precision problem, not a general-
   branch problem. 70.8% of false alarms come from the noise branch alone.
2. A strictly causal, network-level early-warning head warns on 90% of noise
   events with the correct family within one hour of true onset, at a 0.60%
   clean-hour false-warning rate. On calibration it warns **before** the
   confirmed decision is available for 57 of 66 drift/noise events, with a
   median lead of 2 hours.
3. The same early-warning signal, fed into a one-directional alarm verifier,
   raises calibration pooled F1 from 0.6772 to 0.8551 — and this **confirms on
   fresh Modena data**: 0.6863 → 0.8672 across five training seeds and six
   unseen generator seeds, false alarms down 84.2%, paired gain positive in all
   thirty cells.
4. The whole pipeline runs correctly on L-Town, including target-group
   blindness, and is not resource-limited there. Its 120-scenario TRAIN and
   96-scenario calibration corpora exist and its own reference is fitted.

**Failed, and reported as failed:**

5. Early warning for **drift** does not meet the three-hour request: 13.6% of
   drift events by onset +1 h, 31.8% by +2 h. The benchmark ramps drift over
   6–18 hours, so the signal does not exist yet at onset. This is a property of
   the attack model, not a tuning gap, and it was not repaired by changing the
   deadline.
6. Abstention, though implemented and tested, never activated on the development
   surface and did no work.
7. Early **localisation** is not delivered. Sensor-level causal recall at onset
   +1 h is 0.125 for drift and 0.374 for noise, far below the network-level
   event warning rate, and 40 of 242 warning episodes sit entirely on clean
   hours.
8. Threshold-only recalibration (C1) is nearly worthless: +0.006 pooled F1 on
   calibration, +0.014 on fresh data. The problem was never where the thresholds
   sat.
9. **The deployed recipe does not calibrate on L-Town at all** under its own
   recorded constraints. That is a genuine negative transfer result about the
   recipe, reported rather than engineered away.

**Untested:**

10. Nothing about L-Town's detection performance. Its calibration under the
    declared fallback is still running and its fresh evaluation seeds
    (50811–55811) have not been generated.
11. Whether the candidate's Modena gain transfers to a network whose deployed
    baseline recipe is not even feasible. The Modena result says nothing about
    this either way.
12. The early-warning head's *own* behaviour on fresh Modena data. Section 7.1
    scores the confirmed decision; the onset-clock warning metrics in section 3
    are still TRAIN out-of-fold, and section 4.3's warning-to-confirmation trace
    is still calibration.
13. The detector remains pressure-only. Flow missingness stayed fixed at 0.50 and
    no flow detection result is claimed.
