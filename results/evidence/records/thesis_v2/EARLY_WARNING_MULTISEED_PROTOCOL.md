# Early warning, pooled F1, L-Town and repeated runs — preregistered protocol

**Campaign:** `early_warning_multiseed_v1`
**Opened:** 4 September 2026
**Status of this document:** preregistration. Sections 1–9 are frozen before any
candidate is selected. Anything added after a selection or an evaluation is
recorded in section 11 as a dated amendment, never as a silent edit.

This protocol implements `thesis_v2/CLAUDE_HANDOFF_EARLY_WARNING_MULTISEED.md`.
Where the handoff and this document disagree, the handoff wins and the
difference is written down in section 11.

---

## 1. Objectives, in the order they are allowed to be pursued

| # | Objective | Success is measured as |
|---|---|---|
| O1 | Reduce clean-period false alarms without losing family coverage | calibration pooled sensor-endpoint F1, subject to the section 6 gates |
| O2 | Issue a preliminary drift/noise warning before three hours have elapsed | warning recall/precision by true-onset +1 h and +2 h (section 5) |
| O3 | Run the same method on Modena **and** L-Town | per-network results reported separately, never pooled into one average |
| O4 | Report five paired training seeds, not a best run | mean, standard deviation, every individual run, and paired differences |

The thesis contribution is the architecture — specialised evidence, routing,
feedback, and the accuracy/latency/false-alarm trade-off. **No objective may be
pursued by changing the benchmark, the metric scope, or the evaluation
definition.** "Confirmed" is an operational decision status. It is not a claim
that the decision is correct.

## 2. Environment, as verified rather than as inherited

The handoff names `/opt/miniconda3/envs/ozanbabapython312/bin/python`. That
interpreter **does not have `wntr` installed** and cannot run the generator.

The interpreter for this campaign is:

```
/opt/miniconda3/bin/python
```

Verified 4 September 2026: Python 3.13.5, NumPy 2.3.1, SciPy 1.18.0,
pandas 2.3.2, scikit-learn 1.8.0, LightGBM 4.7.0, PyTorch 2.11.0, WNTR 1.4.0,
NetworkX 3.6.1, joblib 1.5.3, Optuna 4.9.0, `wdn` importable from
`/Users/ozanbabac5/wdn_thesis/src/wdn`. This matches `thesis_v2/ENVIRONMENT.md`
exactly. Baseline test suite: **76 passed** (`python -m pytest tests -q`).

Host: macOS 26.6.2 arm64, 14 cores, 48 GiB RAM, 165 GiB free disk. No GPU, no
cloud, no external upload. Existing `runs/` occupies 8.3 GiB and `data/` 2.2 GiB.

## 3. Frozen benchmark

Generator: `src/wdn/operational_data.py` (`OperationalConfig`,
`corrupt_operational`). Settings are exactly the block in handoff section 6 and
are enforced by `OperationalConfig.validate()`:

```
duration_hours 168, timestep_minutes 60,
missing_rate_pressure 0.50, missing_rate_flow 0.50,
pressure_noise_sigma_m 0.10, flow_noise_sigma_m3s 0.0001,
attack_fraction 0.05, clean_gap_hours [48, 96],
attack_duration_hours [6, 18], pressure_bias_m [0.5, 2.0],
flow_bias_m3s [0.0005, 0.002], drift_ramp_hours [6, 18],
replay_lag_hours [2, 6], injected_noise_factor [3, 8],
demand_variation 0.2, demand_pattern_amplitude 1.0
```

Only `profile`, `network_inp`, `output_dir`, `seed` and `num_scenarios` may
differ between configurations of this campaign. Any other differing field is a
distribution change and aborts the run (`distribution_changes` /
`ALLOWED_CONFIG_CHANGES` in `src/wdn/expanded_train_data.py`).

Realised missing fractions fluctuate around 0.50 (the Modena seed-811 manifest
records 0.50015 pressure, 0.49975 flow). The **configured probability** is
checked; an exact realised fraction is never demanded or engineered.

Family vocabulary of the generator: `random`, `replay`, `stealthy` (reported as
**drift**), `noise`, `targeted`. Random and targeted share the abrupt mechanism.

### 3.1 What is off limits

- The original locked test (`splits.json` test scenarios `[0, 1, 8, 20]`) is
  never opened, scored, tuned on, or altered.
- EVAL-1 (`4811`–`9811`), EVAL-2 (`20811`–`25811`) and EVAL-3
  (`30811`–`35811`) are research history. Saved reports may be read to explain
  failures. They are **not** fresh confirmation for this design, and no new
  fitting or selection may touch them.
- Inference features may never read simulator clean values, target identities,
  attack parameters, event onset or closure, family labels, or future
  observations beyond the declared horizon. Labels and event metadata are for
  supervised training, evaluation, and diagnostic plot annotation only.
- The detector stays **pressure-only**. Flow missingness stays fixed at 0.50; no
  flow detection result will be claimed.
- Attack difficulty, delay definitions, metric scope and evaluation gates are
  never changed to manufacture a better number.

## 4. The system this campaign extends

The final operational detector is **not** the old end-to-end GraphSAGE-GRU
cascade, and the old four-trial GNN Optuna search
(`src/wdn/run_expert_redesign.py::run`) is **not** revived.

Current deployed pipeline, traced through
`thesis_v2/experiments/evaluate_feedback_router_eval3.py`:

1. **Reference.** Rank-16 low-rank normal pressure reference
   (`BlindPressureReference`) with four fixed target-excluded sensor groups,
   wrapped by `RobustBlindReference`. Fitted on noisy *observed* normal TRAIN
   readings; a sensor's whole target group is excluded from its own prediction.
   Deployed reference remains the 62-scenario fit
   (`runs/operational/seasonal_family_deployment_v2/full/reference.joblib`).
2. **Feature bank.** 109 residual/dynamic/sequential columns + 7 daily seasonal
   columns = **116**, built by `latency_deployment.specialist_bank`.
3. **General branch.** `ResidualExpertMixture` — a soft tree mixture of five
   internal experts (general, abrupt, replay, drift, noise) with its own
   internal sensor-level router, distinct from the external evidence router.
   Refitted on 110 TRAIN scenarios (`expanded_detector_v1`).
4. **Specialists.** Drift = full seasonal LightGBM blended in logit space with
   the auxiliary drift tree, 90/10. Noise = fast seasonal profile blended with
   the full seasonal tree, 95/5 (`latency_deployment.specialist_scores`).
5. **Delayed decision, Δ = 3 h.** Each specialist blends a learned delayed head
   (`delayed_head_full_v1`) with a bounded forward maximum
   (`latency_deployment.forward_max`), 50/50 in logit space.
6. **External evidence router + feedback.** `FeedbackRouterBundle`
   (`feedback_router_v2`) aggregates 37 observable local features and 8
   specialist scores by mean/std/max per graph-time group; classes are
   clean/abrupt/replay/drift/noise. Two local heads assess drift/noise evidence
   consistency.
7. **Veto.** Specialist `k` is vetoed only when
   `q_replay > q_k + 0.10 AND feedback_k < 0.10`
   (`evidence_feedback.specialist_veto_masks`, margin and cutoff frozen in
   `feedback_router_veto_v1/selection_frozen.json`).
8. **Final decision.**
   `general_alarm OR (drift_alarm AND NOT veto_drift) OR (noise_alarm AND NOT veto_noise)`.

Recorded thresholds — mixture `0.9846252659864263`, drift `0.9147738694864245`,
noise `0.8947208008795838` — **describe the recorded model only**. They are not
constants to be imposed on another network or another refit. Every repeated fit
re-runs the same calibration *procedure* on its own calibration corpus.

"Feedback" here means a local consistency check. It is not human feedback, not
online parameter updating, and not an iterative physics solver.

### 4.1 The causality problem that blocks a rename

The external router consumes `delayed_drift`, `delayed_noise`, `maxpool_drift`,
`maxpool_noise`, `final_drift`, `final_noise` — all of which read observations
in `(t, t+3]`. **The current router therefore cannot be relabelled an early
router.** Stage B builds a strictly causal input path instead (section 5.1).

### 4.2 Data splits, held fixed across every training seed

| Split | Scenarios | Sources |
|---|---:|---|
| TRAIN | 110 | seed 811 scenarios `[2,5,7,9,12,14,15,16,17,18,19,21,22,23]` + seeds 1811, 2811, 3811, 10811, 11811 (24 each) |
| Calibration | 99 | seed 811 scenarios `[3,4,13]` + seeds 12811, 13811, 14811, 15811 (24 each) |
| Validation | 3 | seed 811 scenarios `[6,10,11]` |
| Locked test | 4 | seed 811 scenarios `[0,1,8,20]` — never opened |

Membership is frozen. Training seeds change stochastic fitting, never split
membership.

## 5. Metric definitions, frozen before any measurement

### 5.0 Two clocks, never conflated

- **Decision clock.** The decision for measurement hour `t` is finalised at
  `t + 3 h` and may read observations in `[t, t+3]`.
- **Warning clock.** The supervisor-requested early warning is measured relative
  to the **true attack onset**, which is available to the evaluator only.

Three hours after the first detected warning is *not* three hours after attack
onset. Reports state which clock every number uses.

### 5.1 Early warning — strictly causal

At runtime timestamp `t`, the warning uses only data observed at or before `t`:
causal seasonal/residual bank columns and causal specialist scores
(`frozen_drift`, `frozen_noise` — the un-pooled, un-delayed scores). No
`maxpool_*`, `delayed_*` or `final_*` column enters the early head. No event
onset or family is given to the detector at inference.

Output classes: `normal`, `likely_drift`, `likely_noise`, `other_mechanism`,
plus an **abstain** state when evidence is insufficient. Replay and abrupt
mechanisms are represented as competing explanations; they are never folded into
drift or noise.

**Declared interpretation, stated prominently because the user did not
specify it:** the early warning is a **network-level family warning** at
graph-time resolution, followed by specialist sensor localisation and
confirmation. Any sensor-level early evidence is reported *separately* and is
never presented as localisation accuracy.

Reported at onset `+1 h` and `+2 h` (and at first available observation):

1. Fraction of all drift/noise events warned by each deadline.
2. Fraction of all events warned **with the correct family** by each deadline.
3. Full confusion matrix, **including missed warnings and abstentions** — not
   accuracy conditional on already-detected events.
4. First-warning delay distribution, with missed events retained as
   misses/censored outcomes rather than dropped.
5. Warning false-alarm rate over clean network-hours, plus episode counts under
   the fixed grouping rule of section 5.4.
6. Early localisation, if provided, as a separate sensor-level metric.

Warning false alarms are **always** reported alongside confirmed-alarm metrics.
Reporting only confirmed alarms is prohibited.

### 5.2 Confirmed decisions

Strict sensor-endpoint metrics on all eligible observed pressure endpoints:

- pooled F1, precision, recall, TP/FP/FN counts,
- all five family F1 values (`family_scores` in `src/wdn/latency_deployment.py`:
  a family's false positives are counted inside that family's rows),
- standalone drift and noise expert metrics,
- clean FPR (family 0 rows) and all-negative FPR,
- actual decision timestamps, and warning→confirmation / warning→cancellation
  behaviour.

**Pooled F1 is not the mean of the five family F1 values.** Family F1 excludes
long clean periods; pooled F1 includes every eligible endpoint. Alarm counts are
sensor-time endpoints, not distinct incidents.

Prohibited: event-level point adjustment; dropping unconfirmed endpoints, hard
early hours, unsupported sensors or clean periods. An abstention on an attacked
endpoint is a **miss**, unless a selective-prediction task is separately
declared.

### 5.3 The arithmetic of the pooled-F1 target

Recorded EVAL-3 (144 scenarios, 6 seeds, 3,015,169 observed pressure endpoints):
TP 14,517, FN 2,062, FP 13,469 — of which 1,633 fall inside attack-family
periods and 11,836 outside them. Pooled precision 0.518724, recall 0.875626,
pooled F1 0.651498, clean FPR 0.00446157.

At unchanged TP/FN, pooled F1 0.75 needs FP ≤ 7,616 and 0.80 needs FP ≤ 5,196.
**These are arithmetic identities, not predictions and not promises.**

Because 11,836 of 13,469 false positives (87.9%) lie outside attack-family
periods, strengthening the replay veto alone cannot close the gap: the deployed
guard removed 664 replay-period false positives but only 81 clean-period ones.

### 5.4 Grouping rule for episode counts

A warning episode is a maximal run of consecutive graph-time hours, within one
`(source, scenario)`, in which the warning is non-`normal`. A confirmed-alarm
episode is defined identically on the confirmed decision, per sensor series.
Grouping/debouncing may be reported as operator workload, but **does not by
itself demonstrate an improvement in strict endpoint pooled F1**, and the strict
endpoint number is always reported next to it.

### 5.5 Uncertainty

Means and standard deviations across the five training seeds, every individual
run, and paired candidate−baseline differences. Uncertainty uses a cluster-aware
scheme over generator seeds and scenarios (`bootstrap` in
`thesis_v2/experiments/evaluate_cross_mechanism_deployment.py`). 5 training
seeds × 6 data seeds are **not** treated as 30 independent corpora. Per-network
results stay prominent; a weak L-Town result is never hidden in a cross-network
average.

## 6. Selection gates, frozen before candidate selection

Selection happens on **calibration only**, using TRAIN out-of-fold predictions
for any fitted verifier. The objective is **calibration pooled F1**, subject to
all of the following, evaluated against the paired baseline of the same seed:

| Gate | Threshold |
|---|---|
| drift family F1 | ≥ 0.80 |
| noise family F1 | ≥ 0.80 |
| random family F1 | ≥ 0.90 |
| replay family F1 | ≥ 0.90 |
| targeted family F1 | ≥ 0.90 |
| clean FPR | ≤ 0.005 |
| any family deterioration vs paired baseline | ≤ 0.01 |

These are **selection targets, not predictions of attainable performance**,
particularly on L-Town. The replay ≥ 0.90 gate is stricter than the 0.82 used in
`cross_mechanism_deployment_v1`; that is deliberate and is frozen here.

**If no candidate passes, the negative result is recorded and the gates are not
relaxed afterwards.** Fresh evaluation is not opened to search for a better
score.

## 7. Stages

### Stage A — false-alarm audit (before choosing what to change)

Surfaces: TRAIN out-of-fold and calibration only.

Decompose false alarms by:
general / drift / noise branch; unique versus overlapping contribution; sensor
and scenario; reference support; missing-data gap structure; daily phase;
temporal duration — distinguishing isolated alarms, persistent runs and
post-event tails.

**Provenance is stated, not assumed.** Verified during inspection:
`runs/operational/expanded_train_weak_experts_v1/fold_*/reference.joblib` shows
the four generator-source-held folds each fit **their own** normal reference, so
the specialist TRAIN OOF is full-pipeline OOF over 62 scenarios. The general
mixture has **no** matching OOF artifact on that corpus; where a general-branch
OOF score is needed it is either produced under the same source-held folds or
the diagnostic is explicitly labelled *fixed-reference conditional*, never
silently called full-pipeline OOF.

Calibration is genuinely held out from fitting, but the deployed thresholds were
*chosen* on calibration; calibration false-positive counts at those thresholds
are therefore optimistically biased, and this is stated wherever they appear.

The audit is delivered **before** any decision about which expert to change. If
most clean false positives originate in the protected general branch, the audit
quantifies the ceiling on specialist-only suppression and proposes a
general-branch change **openly**, rather than covertly vetoing that branch.

### Stage B — causal early warning

First candidate: a small tree or regularised linear classifier over existing
causal features and causal specialist scores. No new large neural search. A
standalone GRU already failed the prior OOF screen
(`thesis_v2/CAUSAL_SEQUENCE_EXPERT_RESULTS.md`); adding one is not assumed to
help. `thesis_v2/CAUSAL_DEADLINE_LOCALIZER_PROTOCOL.md` receives external onset
and family and allows ten checkpoints — **it is not a valid continuous
early-warning baseline** and is not used as one.

Training uses early-phase positives and hard normal examples. The three-hour
specialist confirmation is kept initially; earlier confirmation is optional and
only under a frozen rule with sufficient evidence.

**The early head is advisory.** It never becomes a mandatory gate: an absent
warning must not disable later specialist detection, and the general mixture's
immediate alarm path stays protected in the initial candidates. Warning,
revision and final-decision timestamps are stored explicitly.

### Stage C — bounded pooled-F1 screen (at most three candidates)

On a single Modena pilot training seed:

- **C1** Stricter branch-wise calibration — threshold-only control.
- **C2** Learned specialist alarm verifier over causal/delayed evidence,
  reference reliability, support and temporal consistency.
- **C3** C2 augmented with the early-warning trajectory/consistency signal.

Verifier training uses out-of-fold predictions; hard-negative mining is
restricted to TRAIN. The model is small and bounded — not an open search. A
clean/no-attack outcome is supported, and noise alarms are not prolonged by
unnecessary persistent memory. The existing replay guard is either retained or
replaced by a clearly documented alternative; **both are never applied at once**.

### Stage D — L-Town smoke test and architecture freeze

New operational TRAIN/calibration/evaluation corpora are created for L-Town.
`data/v2_ltown`, `hard_ltown`, `rec_hard_ltown` and the old episode datasets are
**not** interchangeable with this benchmark and are not reused.

A network-specific normal reference and network-specific experts are fitted on
L-Town TRAIN and calibrated on L-Town calibration. This tests **the same method
on a second network** — it is not zero-shot Modena→L-Town transfer, and Modena's
reference is never loaded for L-Town.

Audited before evaluation: sensor roster, masks, target-group exclusion,
seasonal history, eligible endpoints, peak RAM, wall time, disk. L-Town has 782
junctions against Modena's 268 (≈2.9×), so row counts and memory scale
accordingly and memory-efficient feature construction is tested explicitly.

Scenario budgets match Modena (110 TRAIN, 99 calibration) where feasible, using
the same whole-scenario assignment rule — no fabricated split to hit a count.
**If safe L-Town sizing requires a material change, the pilot estimate is
reported and agreement is obtained before full runs.** Equal physical
perturbations need not mean equal standardised difficulty; that difference is
reported, never equalised by adjusting attacks.

The common architectural recipe is chosen from permitted development data and
frozen before new evaluation. The model is not redesigned after seeing L-Town
evaluation scores, and L-Town starts from the same architecture rather than a
bespoke exception.

### Stage E — repeated fitting and fresh evaluation

- **5 training seeds × 2 networks × 2 variants (baseline, candidate) = 20 main
  conditions.**
- Baseline = the current **guarded hybrid** recipe (section 4) — not the
  obsolete GNN cascade and not the unguarded OR.
- Because the candidate only adds a decision layer, the baseline and candidate
  of a seed pair **share the exact fitted base experts and reference**: ten base
  fits plus extension fits, not twenty redundant copies.
- A training run reruns the relevant stochastic fitting stages. Re-scoring one
  checkpoint five times is not five training runs. What varies, what is
  deterministic and what is frozen is documented; artificial randomness is never
  injected merely to produce a nonzero standard deviation.
- TRAIN/calibration membership is fixed across training seeds; the same
  calibration algorithm runs separately per run; the best seed is never chosen.
- Target: **6 fresh evaluation generator seeds per network, 24 scenarios each
  (144 per network)**, subject to the L-Town resource gate.
- All five paired fitted variants, their calibration rules and their hashes are
  frozen **before** the fresh evaluation is scored. Each evaluation corpus
  scores the prespecified paired runs; the first seed's scores are never
  inspected and used to adapt the rest.

## 8. Seed reservation

Reserved **before** generation, checked for collisions against every existing
config and manifest — not merely against filenames.

Excluded because they already exist or are locked: `811`, `1811`, `2811`,
`3811`, `10811`, `11811`, `12811`, `13811`, `14811`, `15811`, `4811`–`9811`,
`20811`–`25811`, `30811`–`35811`.

Reservation for this campaign (recorded in
`runs/operational/early_warning_multiseed_v1/seed_manifest.json`, which is the
authoritative copy):

| Purpose | Network | Seeds |
|---|---|---|
| Fresh evaluation | Modena | `40811, 41811, 42811, 43811, 44811, 45811` |
| Fresh evaluation | L-Town | `50811, 51811, 52811, 53811, 54811, 55811` |
| L-Town TRAIN | L-Town | `60811, 61811, 62811, 63811, 64811` |
| L-Town calibration | L-Town | `70811, 71811, 72811, 73811` |
| L-Town pilot / smoke | L-Town | `90811` |

Training seeds for the repeated fits (model-fitting randomness, *not* data
generation): `701, 702, 703, 704, 705`.

## 9. Verification requirements

Implemented as tests alongside the existing suite. At minimum:

1. Changing observations after `t` cannot change the early warning at `t`.
2. Changing observations after `t+3` cannot change the final decision for `t`.
3. Missing placeholder values cannot alter predictions; timestep gaps cannot
   turn a row shift into an hour shift.
4. No feature accepts inference labels, onsets, families or target identities;
   target-group blindness of the reference remains intact.
5. General alarms survive the specialist guard, and an absent early warning does
   not disable a later specialist.
6. Network/source/scenario caches and OOF folds cannot mix rows, cannot fit a
   preprocessor on held data, and cannot reuse Modena's reference for L-Town.
7. TRAIN, calibration and fresh-evaluation identities are disjoint; generator
   configuration checks reject wrong missingness or a wrong schema.
8. Saved and reloaded predictions reproduce; recorded counts independently
   reproduce the reported F1; warning and final metrics use their declared
   denominators.
9. Resume skips completed stages and **refuses** an incompatible cached
   configuration instead of overwriting it.

Implementation quality: atomic writes, per-stage status files, config hashes,
useful error messages, and saved artifacts inspected without retraining wherever
possible. A failed fit is diagnosed and repaired, never silently retried forever.

## 10. Deliverables and stopping conditions

1. This protocol plus complete seed/data/model manifests.
2. False-alarm audit JSON/CSV and explanatory plots.
3. Network-agnostic early-warning and verifier implementation, with tests and
   reproducible run commands.
4. Resumable campaign runner with training/evaluation separation.
5. Per-run and per-network raw metrics, confusion counts, thresholds, model
   hashes and timings.
6. `thesis_v2/EARLY_WARNING_MULTISEED_RESULTS.md` — English results report with
   figures under `thesis_v2/outputs/early_warning_multiseed/`.
7. A short Turkish user summary: what improved, what failed, what is untested.

**Stopping conditions.** If the bounded screen fails, the diagnosis is finished
and the negative result is reported; fresh evaluation is not opened to hunt for
a better score. If L-Town memory or runtime is impractical, a measured estimate
is reported and a smaller budget is requested. Existing runs, reports, bundles
and the thesis draft are preserved; nothing is reset, cleaned or overwritten.
The graduation PDF and the presentations are not rewritten as part of this work.

**Done means measured and reproducible results, not submitted training
commands.** A still-running campaign is left with a resumable status file and
the exact next command, clearly labelled as incomplete.

## 11. Amendments

*(Dated entries only. Nothing above section 11 is edited after a selection or an
evaluation.)*

- **2026-09-04** — Protocol opened. Interpreter corrected from the handoff's
  `ozanbabapython312` env to `/opt/miniconda3/bin/python` (section 2), because
  the named env cannot import `wntr`.

- **2026-09-04, after Modena's fresh evaluation, before any L-Town evaluation
  corpus was generated or scored.** On L-Town calibration the **deployed
  baseline recipe has no feasible operating point**: its own recorded
  constraints (clean FPR ≤ 0.005 *and* replay ≥ 0.82 *and* random ≥ 0.90 *and*
  targeted ≥ 0.90) cannot be satisfied anywhere in the budgeted-OR share grid.
  Without a baseline there is no paired comparison, so the L-Town arm would
  deliver nothing at all.

  This amendment therefore declares, for **L-Town only** and for **every arm
  identically**:

  1. The infeasibility is measured and reported as a result about the network —
     `constraint_frontier` records, for each constraint, the best value reachable
     anywhere in the grid and the best value reachable within the clean-FPR
     budget, plus which constraints are unreachable.
  2. Both the baseline and the candidates then fall back to the *same* rule:
     maximise the arm's own objective subject to the **0.005 clean-FPR budget
     alone**. The deployed family floors are reported, never enforced.
  3. The **relative** protection is kept unchanged: no candidate may leave any
     family more than 0.01 below its paired baseline. A candidate still cannot
     win by sacrificing a family.

  What this is not: it is **not** a relaxation of the candidate's gates after
  seeing a candidate's score. The absolute floors were calibrated on Modena and
  are a property of the deployed recipe, not of this campaign's hypothesis; the
  failing arm is the *baseline*. Modena's results are unaffected and keep the
  original enforced gates. Every L-Town number will be labelled as produced
  under this fallback, and the enforced-gate infeasibility will be reported
  next to it rather than omitted.
