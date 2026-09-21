# A different performance direction: blind normal reference

## Decision

Do not rely on a larger temporal GNN or a longer Optuna search as the main
route to the requested improvement. A separate, measured prototype gives a
substantial development improvement by changing the reference signal and
expert inputs. This is a candidate direction, not a validated final model.

The current GNN reconstruction can see the same sensor reading that its
anomaly head judges. That permits copying some corruption into the reference;
this is a structural risk, not a proven explanation of the entire old gap.
The new reference predicts a group of sensors using only other groups. Even
robust reweighting cannot access the target group. Predictions are compared
with the actual observed readings to obtain causal residual histories.

The exclusion principle is related to J-invariant prediction in
[Noise2Self (Batson & Royer, ICML 2019)](https://proceedings.mlr.press/v97/batson19a.html).
This is our pressure-reference adaptation, not an implementation of their
image model or a claim that their denoising guarantees prove attack detection.

## What is implemented

- A low-rank pressure reference learned exclusively from **noisy observed
  pressure in normal training episodes**. No simulator clean pressure or
  flow targets are used, even for fitting this reference.
- Four fixed sensor groups, inferred independently of values and labels.
  For each group, masked robust factor fitting uses only the other groups.
- Normalised residual magnitude, slope, variance, persistence and delayed
  reference agreement over 4/8/16-hour histories, with observation support.
- A single fixed tree classifier to isolate representation quality.
- A separate mixture of five tree experts: general, abrupt bias, replay,
  drift and noise. Random/targeted attacks share the same bias mechanism and
  therefore share the abrupt expert; their report labels remain separate.
- A learned **sensor-level** soft router that receives features only at
  inference. Training labels supervise the router but never select experts
  during calibration/validation inference.

This is a tree-based mixture, **not** a completed integration into the temporal
GNN. It currently handles pressure detection; it does not replace the full
pressure/flow reconstruction pipeline. No claim of neural MoE improvement or
equal model capacity is made.

## Fixed-data development results

| Method | Overall F1 | Replay F1 | Overall AUPRC | Sensor FPR |
|---|---:|---:|---:|---:|
| Original V2 GNN pilot, 12 epochs | 0.42445 | 0.73239 | 0.35867 | 0.004244 |
| Completed Optuna trial 1 | 0.47754 | 0.71233 | 0.46249 | 0.002797 |
| Final best of four GNN Optuna trials (trial 3) | 0.49393 | 0.70270 | 0.46344 | 0.002235 |
| Rank-4 blind reference + single tree | 0.75892 | 0.85246 | 0.76611 | 0.000257 |
| Rank-8 blind reference + single tree | 0.67813 | 0.88889 | 0.6798 | 0.001913 |
| Rank-16 blind reference + single tree | 0.78661 | 0.88889 | 0.7920 | 0.000257 |
| Rank-16 blind reference + learned expert mixture | **0.80374** | **0.89231** | **0.79990** | **0.000466** |
| Same tree experts, uniform mixture | 0.77304 | 0.90625 | See saved summary | See saved summary |

Three reference ranks (4, 8, 16) were explored on development data; rank 16
was then used for the expert prototype. These are model choices, not new
independent evaluations. The fixed tree recipe uses 120 boosting iterations,
15 leaves, learning rate 0.08 and no validation-based early stopping.
The original four-trial GNN search is now complete, with all four expert
audits finished. Trial 1 was the completed comparator when this prototype
was first recorded; trial 3 is the final winner of the original search.

Every score above uses exactly the same calibration/validation sensor labels,
family order and 16-step endpoints as the V2 GNN pilot. Pressure and flow
missing probabilities stay at 0.50. No attack amplitude, prevalence, scenario
split or physical parameter was changed. Each detector selects one global
threshold on calibration using the original overall/replay objective. The
locked test is not used or scored.

The learned mixture has 301 TP / 29 FP / 118 FN, overall precision 0.91212,
recall 0.71838, and a remaining overall-F1 gap of 0.09626 to 0.90. Its replay
score uses only 35 positive readings (29 TP / 1 FP / 6 FN); this is a small,
correlated development sample and not evidence of field robustness.

## Expert quality is not solved

Mixture F1 by family: random **0.9697**, replay **0.8923**, drift **0.2222**,
noise **0.2985**, targeted **0.9275**. The aggregate gain is largely on abrupt
and replay attacks. Weak gradual drift and noise remain major limitations.

| Expert | Random AUPRC | Replay AUPRC | Drift AUPRC | Noise AUPRC | Targeted AUPRC |
|---|---:|---:|---:|---:|---:|
| General | 1.000 | 0.888 | 0.451 | 0.662 | 0.969 |
| Abrupt | 0.998 | 0.849 | 0.299 | 0.438 | 0.983 |
| Replay | 0.993 | **0.943** | 0.275 | 0.312 | 0.963 |
| Drift | 0.999 | 0.654 | 0.417 | 0.343 | 0.870 |
| Noise | 0.932 | 0.712 | 0.429 | 0.603 | 0.571 |

General still outperforms the dedicated drift/noise experts on their own
families. The global mixture threshold is 0.99, consistent with uncalibrated
scores from balanced specialist objectives; numerical scores must not be
interpreted as calibrated attack probabilities. The old threshold search is
coarse, so calibration and weak-family recall deserve a controlled follow-up.

## Verification and next work

27 tests pass. New tests change every value in a target group and verify that
its reference predictions do not change, verify missing-placeholder invariance,
empty-reference behaviour, and causal temporal features. Saved-estimator reload
predictions were checked against saved outputs. Validation labels and family
order exactly match the original GNN audit arrays.

The next architecture decision should preserve this blind reference and its
strong single-classifier control, then strengthen/calibrate the drift/noise
experts. A neural expert integration should be judged against this control,
not assumed to improve it. Before making final claims: confirm on independent
training/data seeds, quantify event-level false alarms and delay, and test
sensitivity to sensor placement/model mismatch. Keep the locked test unopened
until the method and selection protocol are fixed.

## Artifacts

- `src/wdn/models/blind_reference.py`
- `src/wdn/probe_blind_reference.py`
- `src/wdn/probe_residual_experts.py`
- `runs/operational/blind_reference_probe_v1`
- `runs/operational/blind_reference_probe_rank8`
- `runs/operational/blind_reference_probe_rank16`
- `runs/operational/blind_residual_experts_v1/summary.json`

All are separate from the old datasets, model checkpoints and completed Optuna
study. The original search was not restarted or modified for these probes.

## Subsequent weak-family follow-up

The original results above are preserved. See [FAMILY_BALANCE_RESULTS.md](FAMILY_BALANCE_RESULTS.md)
for the later causal-context and calibration experiments on the same data.
One family-balanced operating point raises drift/noise F1 to 0.508/0.575,
but lowers overall F1 to 0.732 and increases false positives from 29 to 178.
An aggregate-priority point reaches overall F1 0.818 while drift/noise remain
0.379/0.438. A separate calibration-only score-fusion search did not resolve
this tradeoff. None of these results establishes comprehensive attack coverage.
