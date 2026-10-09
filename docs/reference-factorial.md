# E11: target-group exclusion and iterative reweighting

This supplement documents a paired 2 × 2 factorial study of two operations in
the final pressure reference: exclusion of the target sensor group and iterative
residual reweighting (IRLS). The downstream detector is refitted for each altered
condition. The original reference representation, normalization, architecture,
branch-budget allocation and gate hyperparameters are held fixed.

The study adds to the original [E1–E9 submission snapshot](https://github.com/onurozansunger/wdn-thesis/tree/49036da8016097f2ebb627b3b58cea1a07432a47)
and the separate [E10 operating-point and component analysis](https://github.com/onurozansunger/wdn-thesis/blob/300f7274d1158a4e1929f7f97028ed54f975baec/docs/evidence-supplement.md).
It does not replace their frozen results. Cite E11 using the commit containing
this guide and its evidence files.

## Design and evidence

The [design](../results/reference_factorial_20261008/records/runs/operational/reference_factorial_20261008/protocol_design.json)
and [frozen protocol](../results/reference_factorial_20261008/records/runs/operational/reference_factorial_20261008/protocol_frozen.json)
specify both networks, model seeds 701–703 and the same six previously used
evaluation sources per network. The protocol SHA-256 is
`474141759c5d624e56e9be85c9378ed7db0e42d94fdfdd90b145367753de56fd`.

| Condition | Target-group exclusion | IRLS |
|---|---|---|
| `e1r1` | Enabled | Enabled |
| `e1r0` | Enabled | Disabled |
| `e0r1` | Disabled | Enabled |
| `e0r0` | Disabled | Disabled |

`e1r1` uses the retained final models. The other three conditions require 18
new downstream system fits: three conditions × two networks × three model
seeds. The reporting contract requires 24 calibration records and all 144
condition/model-seed/evaluation-source cells. Each calibration record fixes
four policy choices, giving 576 selected evaluation records.

| Evidence | Purpose |
|---|---|
| [Reference and feature parity](../results/reference_factorial_20261008/records/runs/operational/reference_factorial_20261008/control_parity_approved.json) | Thirteen representative TRAIN cases covering both full references and all eleven source-held references |
| [Control score parity](../results/reference_factorial_20261008/records/runs/operational/reference_factorial_20261008/control_score_parity.json) | The new scoring adapter checked against retained E10 calibration scores, one source per network |
| [Supplemental schema inputs](../results/reference_factorial_20261008/records/runs/operational/reference_factorial_20261008/supplemental_schema_inputs.json) | Ordered feature names and used split allowlists checked against the frozen TRAIN/calibration manifests before altered-condition fitting |
| [Evaluation freeze](../results/reference_factorial_20261008/records/runs/operational/reference_factorial_20261008/evaluation_frozen.json) | Hashes of all 24 calibration records and their fitted model files |
| [Calibration and evaluation records](../results/reference_factorial_20261008/records/runs/operational/reference_factorial_20261008/results) | Complete threshold paths, counts, selected policies and score provenance |
| [Summary](../results/reference_factorial_20261008/records/runs/operational/reference_factorial_20261008/report/summary.json) | Condition summaries, paired factorial effects and report-file hashes |
| [Cell table](../results/reference_factorial_20261008/records/runs/operational/reference_factorial_20261008/report/cells.csv) | Every selected cell/policy result and its confusion counts |
| [Condition summary](../results/reference_factorial_20261008/records/runs/operational/reference_factorial_20261008/report/condition_summary.csv) and [source metrics](../results/reference_factorial_20261008/records/runs/operational/reference_factorial_20261008/report/source_metrics.csv) | Condition means and source-level variation |
| [Effect summary](../results/reference_factorial_20261008/records/runs/operational/reference_factorial_20261008/report/effect_summary.csv) and [source effects](../results/reference_factorial_20261008/records/runs/operational/reference_factorial_20261008/report/source_effects.csv) | Paired simple effects, main effects and interaction |
| [Publication manifest](../results/reference_factorial_20261008/manifest.json) | Original-to-published paths, byte counts and SHA-256 hashes |

## Results under the fixed calibration policies

The complete report contains all 144 evaluation cells and 576 selected
cell/policy records. The primary policy maximizes pooled calibration F1 under
a 0.05% clean-FPR cap. The table reports evaluation means over three model
seeds within each of six shared sources, then equally over sources. These
are recalibrated factorial results, separate from the original ten-seed
selected-system results.

| Network | Condition | Pooled F1 | Replay F1 | Clean FPR (%) |
|---|---|---:|---:|---:|
| Modena | `e0r0` | 0.8168 | 0.6395 | 0.0170 |
| Modena | `e0r1` | 0.8752 | 0.9425 | 0.0380 |
| Modena | `e1r0` | 0.8104 | 0.6282 | 0.0179 |
| Modena | `e1r1` | 0.8709 | 0.9335 | 0.0378 |
| L-Town | `e0r0` | 0.9011 | 0.6801 | 0.0177 |
| L-Town | `e0r1` | 0.9152 | 0.6989 | 0.0179 |
| L-Town | `e1r0` | 0.9017 | 0.6878 | 0.0234 |
| L-Town | `e1r1` | 0.9129 | 0.6876 | 0.0191 |

The paired primary-policy effects below are enabled minus disabled; the
interaction is the difference between the two simple effects. Entries are
mean ± descriptive SD across six source means. F1 effects are absolute;
clean-FPR effects are percentage points. The SDs are not standard errors or
significance tests.

| Network | Contrast | Pooled F1 difference | Replay F1 difference | Clean FPR difference (pp) |
|---|---|---:|---:|---:|
| Modena | Exclusion, IRLS on | -0.0042 ± 0.0046 | -0.0091 ± 0.0113 | -0.0002 ± 0.0051 |
| Modena | Exclusion, IRLS off | -0.0064 ± 0.0062 | -0.0113 ± 0.0148 | +0.0009 ± 0.0010 |
| Modena | IRLS, exclusion on | +0.0606 ± 0.0150 | +0.3052 ± 0.0299 | +0.0199 ± 0.0043 |
| Modena | IRLS, exclusion off | +0.0584 ± 0.0110 | +0.3030 ± 0.0249 | +0.0210 ± 0.0069 |
| Modena | Interaction | +0.0022 ± 0.0065 | +0.0022 ± 0.0200 | -0.0011 ± 0.0053 |
| L-Town | Exclusion, IRLS on | -0.0023 ± 0.0011 | -0.0113 ± 0.0036 | +0.0012 ± 0.0005 |
| L-Town | Exclusion, IRLS off | +0.0006 ± 0.0020 | +0.0077 ± 0.0042 | +0.0057 ± 0.0008 |
| L-Town | IRLS, exclusion on | +0.0112 ± 0.0013 | -0.0002 ± 0.0070 | -0.0042 ± 0.0009 |
| L-Town | IRLS, exclusion off | +0.0141 ± 0.0019 | +0.0188 ± 0.0070 | +0.0002 ± 0.0004 |
| L-Town | Interaction | -0.0029 ± 0.0024 | -0.0189 ± 0.0049 | -0.0045 ± 0.0009 |

IRLS improves mean pooled F1 at both exclusion settings in both networks.
In Modena, its replay gains accompany higher clean FPR. In L-Town, enabling
IRLS with exclusion improves pooled F1 and lowers clean FPR, while mean
replay F1 changes little; without exclusion, replay improves. The negative
L-Town replay interaction describes this dependence on exclusion. Exclusion
itself does not consistently improve the reported metrics: its primary
pooled and replay effects are negative in Modena, while their signs in
L-Town depend on whether IRLS is enabled. These contrasts concern downstream
refitting within the fixed representation and calibration recipe.

Relaxing the cap from 0.05% to 0.50% under the pooled objective leaves all
reported condition means unchanged. The balanced objective changes the
trade-off. For L-Town `e1r1`, switching to balanced calibration at the 0.05%
cap raises replay F1 from 0.6876 to 0.7293, while pooled F1 falls from
0.9129 to 0.9015 and clean FPR rises from 0.0191% to 0.0462%. At the same balanced
policy, `e0r1` reaches replay F1 0.7381 with pooled F1 0.9043 and clean FPR 0.0453%.
Relaxing the balanced cap further gives `e1r1` replay F1 0.7303, pooled F1
0.8961 and clean FPR 0.0535%; the other three conditions have slightly lower
mean replay F1 than under their stricter balanced policy.

For Modena `e1r1`, mean cellwise worst-family F1 rises from
0.8036 under the primary policy to 0.8126 and 0.8218 under the two balanced
policies. The corresponding pooled F1 values are 0.8709, 0.8693 and 0.8501;
clean FPR rises from 0.0378% to 0.0503% and 0.0888%. With IRLS disabled,
balanced calibration instead raises Modena replay F1 while lowering pooled,
drift and noise F1 relative to the pooled objective. A calibration cap does
not guarantee the same achieved evaluation FPR. The complete report retains
all four policies and all seven paired contrasts; none replaces the original
selected system or its research-goal assessment.

## What changes

Every condition uses the saved full or source-held TRAIN anchor's mean,
hydraulic standardization, rank-16 basis, four sensor groups, ridge parameter,
inner whitening scale and outer residual scale. It also retains three views,
the original view RNG and median aggregation. Neither scale is recalibrated
for a condition.

With exclusion enabled, a group's readings cannot enter the latent fit used
to predict that group. With exclusion disabled, observed target-group readings
are added to the corresponding views. Original non-target membership and
fallback decisions are preserved. The reported support feature remains the
fraction of observed non-target sensors in every condition. Prediction and
view spread are recomputed.

With IRLS enabled, the reference performs the original six ridge solves and
leverage-adjusted residual updates. With IRLS disabled, it performs one solve
with binary observation weights. This comparison isolates iterative
reweighting within the retained reference: whitening, the shared scales,
three-view median and other robustness mechanisms remain present.

The Modena full anchor retains its original 62-scenario TRAIN population;
downstream Modena fitting uses 110 TRAIN scenarios. L-Town's full anchor uses
its 120 TRAIN scenarios. Each source-held anchor retains its original fitting
population and excludes its held source. Reference fitting used received
pressure observations from family-0 rows, without simulator clean values.

The altered conditions rebuild full and source-held features and refit the
General mixture, causal and delayed specialists, early-warning head, external
router and feedback, and branch verifiers. L-Town also refits the selected
full-history General mixture. Model-seed mappings, sampling algorithms,
weights, feature order and temporal windows remain fixed. Score-dependent
hard-negative samples can change under the same sampling algorithm; this is
part of the downstream response to the altered reference.

The [reference implementation](../results/reference_factorial_20261008/code/src/wdn/models/reference_factorial.py)
and [pipeline](../results/reference_factorial_20261008/code/thesis_v2/experiments/reference_factorial/pipeline.py)
record these interventions. The two parity records establish exact equality
for the representative retained-control cases that they list. They do not
claim an exhaustive repeat of every original feature or score array.

## Calibration and paired effects

The primary policy maximizes pooled calibration F1 under a clean-FPR cap of
0.0005 (0.05%). Secondary policies use the balanced objective at that cap and
both objectives at 0.005 (0.5%). The balanced objective maximizes the minimum
of the five family F1 scores. Tie-breaking is fixed in the protocol.

Each condition follows the same 82-point multiplier path: zero plus 81
logarithmically spaced values from 0.01 to 100, including one. The multiplier
scales the original model seed's nominal General, Drift and Noise budgets.
Thresholds are recomputed from that condition's family-0 calibration scores.
Alarms use strict score exceedance; verifier acceptance uses `>=` its cutoff.
The zero-budget point has zero clean calibration alarms, although it can
still detect attacks above the maximum clean score.

Original seed-specific verifier acceptance cutoffs, L-Town router temperature
and shrinkage, and the external veto are fixed across conditions. Verifier
candidate prethresholds and early-warning abstention are instead learned from
each altered condition's TRAIN-OOF predictions using the original algorithms.
The verifier returns 1 outside its learned candidate region. No family
protection or promotion constraints are used in this factorial calibration.

All calibration choices must be frozen before new factorial evaluation
summaries are computed. Common calibration caps do not imply equal achieved
evaluation FPR. The path is a declared sensitivity analysis; it is not an
exhaustive frontier or an AUROC/AUPRC estimate for the hybrid.

For each metric, contrasts are calculated within matched model-seed/source
cells before averaging. For example, the exclusion effect with IRLS enabled
is `M(e1r1) − M(e0r1)`, and the interaction is
`M(e1r1) − M(e1r0) − M(e0r1) + M(e0r0)`. Three model-seed contrasts are
averaged within each source, followed by equal averaging across six sources.
The protocol specifies all seven contrasts, including both simple effects
for each factor and the two averaged main effects.

F1 effects are absolute differences. FPR values in the JSON and CSV files are
rates; multiply a rate by 100 for percent and a difference by 100 for
percentage points. Source SD is descriptive variation across six source
means, not a standard error based on 18 independent datasets. Worst-family
F1 is computed within each cell before averaging.

## Verification and reproduction

From the public repository root, run the standard-library evidence checker:

```bash
python3 results/reference_factorial_20261008/verify.py
```

Its scope is the supplied files, frozen identity chains, count arithmetic,
declared calibration selection and paired report aggregation. Saved counts
and path reports allow these checks without loading models. They do not
allow an independent reconstruction of predictions or clean-score quantiles
when their underlying arrays are absent.

The publication manifest distinguishes unchanged archived files from new
publication documentation and verification code. The `code/` directory
preserves all 122 Python files listed in the frozen protocol at their original
relative paths. Of these, 116 match the earlier public snapshot and six are
new factorial modules. The records also include corpus manifests, generation
configurations, event metadata, original rule records and score sidecars.
Supporting feature-name and split JSONs are included for path reconstruction.
They were not separately listed in the protocol's input-hash list. The
supplemental schema record verifies that the ordered 109 base names plus seven
seasonal names equal the feature schema in all four frozen TRAIN/calibration
manifests, and that source 811's used TRAIN and calibration scenario lists
equal the corresponding frozen manifest entries. That comparison was recorded
after protocol freeze but before any altered-condition fitting. It does not
retroactively amend the protocol.

Raw pickle corpora, fitted model and reference joblib files, feature caches,
OOF arrays and per-endpoint NPZ scores are outside this publication's compact
evidence set. Their recorded fingerprints connect the published records to
the research archive. Fingerprint consistency is not verification of omitted
bytes. The repository alone therefore supports evidence verification, not
complete retraining or inference reproduction.

Inspecting the exact archived run and performing a fresh rerun require
different output preparation. To inspect the archived run, retain its frozen
protocol, evaluation freeze and result records as published, with their
recorded hashes. The public evidence checker does not alter them.

For a fresh rerun with the external research archive, reconstruct a separate
workspace using the manifest's original code and input paths. Verify the
original input hashes and the supplemental schema hashes before running.
Restore the original E10 control score arrays and models, full/fold reference
anchors, feature caches required by the parity checks, raw corpora and schema
JSONs. In the new E11 output namespace, initially restore only
`protocol_design.json` and the retained `control_score_parity.json` record.
Do not copy the archived E11 `protocol_frozen.json`, `evaluation_frozen.json`,
results, generated caches or new models into that destination.

The control parity check creates a new record, including execution details,
so its hash can differ on a fresh run. The freeze step must record that new
identity. Copying completed E11 outputs into the same namespace would trigger
the immutable-record checks or cause completed stages to be reused. Preserve
the archived run separately for comparison.

The code snapshot must be restored to its original paths; its scripts are
not intended to run directly from the publication's nested `code/` directory.
The frozen runtime records Python 3.13.5, NumPy 2.3.1, SciPy 1.18.0,
scikit-learn 1.8.0, LightGBM 4.7.0 and joblib 1.5.3. Check the restored runtime
against that record. Reference feature construction explicitly uses two
BLAS threads. The executable entry points in the isolated workspace are:

```bash
python thesis_v2/experiments/reference_factorial/control_parity.py
python thesis_v2/experiments/reference_factorial/experiment.py --stage freeze
python thesis_v2/experiments/reference_factorial/experiment.py --stage all
python thesis_v2/experiments/reference_factorial/report.py
```

The frozen driver uses two seed workers and processes altered conditions
sequentially. It removes only newly generated, reproducible TRAIN/fold caches
after their models and calibration records are verified. Cleanup records
preserve those cache hashes. The report refuses missing evaluation cells and
checks all predefined primary and secondary policies.

The recorded run also used external scheduling helpers to reduce elapsed
time. The [training scheduling record](../results/reference_factorial_20261008/records/runs/operational/reference_factorial_20261008/execution_parallelization_20261009.json)
records up to three seed workers, with no more than two simultaneous L-Town
fits. The [evaluation scheduling record](../results/reference_factorial_20261008/records/runs/operational/reference_factorial_20261008/execution_evaluation_parallelization_20261009.json)
records six independent source workers after all 24 calibration selections
were frozen. Each record embeds its helper source and hash. The helpers
preserve the frozen numerical code, estimator settings and BLAS configuration;
each evaluation cell has one writer, and shared feature preparation finishes
before scoring begins. The commands above retain the original, slower
scheduling and do not require these helpers.

## Limits

This is a retrospective factorial on previously used evaluation sources and
three retained model seeds. It supplies conditional evidence within the
selected representation and architecture. The original branch allocations
and gate hyperparameters were selected for `e1r1` and may favor it; the study
does not compare four independently optimized architectures.

IRLS removal leaves the other normalization and aggregation mechanisms
intact. The shared TRAIN representation also retains target sensors during
basis fitting; the exclusion factor concerns the current snapshot's latent
fit. This study does not isolate the early-warning contribution, use a fresh
locked test or establish performance on real operational attacks.

The retained stacking procedure holds sources out separately at successive
layers. Later held-source fits reuse the assembled base OOF predictions;
they do not retrain every upstream component within a fully nested split.
The factorial preserves this training procedure in every condition.
