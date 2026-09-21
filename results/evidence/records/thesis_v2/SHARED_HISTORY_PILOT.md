# Shared causal history: a controlled architecture pilot

## Why this experiment

The independently confirmed router recalibration improved L-Town replay F1
from 0.6461 to 0.6757 and pooled F1 from 0.8799 to 0.8841. This was a real but
modest improvement, not the requested jump to 0.80. Forcing greater false-alarm
budget use had already failed a calibration pilot. Neither result establishes
that the replay expert itself, rather than routing or final fusion, is the main
bottleneck.

The next controlled comparison changes the information supplied to the existing
five experts and their router. It adds no expert, no family-specific inference
rule, and no additional decision latency.

## What changes

The original 116 features remain intact. An additional 56 features preserve
past evidence at 1, 2, 3, 4, 5, 6, 8, and 12 hours:

- residual, sequential innovation, dynamic innovation, lag advantage, and
  reference support at each past hour;
- the change from that hour's residual to the current residual;
- an explicit availability mask for that hour.

Every existing expert receives these common features; their original feature
profiles otherwise stay unchanged. The router receives the same augmented bank.
The number of estimators, their HGB hyperparameters, training seed, selected
training rows, and population weights are identical between the two arms.

These are trajectories of existing residual evidence, **not** the six separate
raw lag-comparison scores discarded by the original feature extractor, nor a
match against unavailable clean readings. Missing observations remain missing;
the code does not count the last observed row as the previous hour. No future
observations, labels, event boundaries, or clean targets enter these features.

## Scope and safeguards

The pilot uses L-Town's existing source-held TRAIN fold 0 and model seed 701.
The fold reference was already fitted without the held-out generator source.
All four remaining TRAIN sources supply the training population; as in the
original estimator, every positive and 60,000 sampled negatives enter fitting,
with weights restoring the original clean population size.

Within the held-out **TRAIN** source, even scenario IDs select thresholds and
odd scenario IDs supply the diagnostic report. This is an inner development
partition, not a change to the canonical dataset splits and not locked testing.
Both arms select thresholds by pooled F1, then worst-family F1, under clean
false-alarm budgets no greater than 0.005. The same predeclared grid applies to
all score streams. Diagnostic labels do not select thresholds.

The frozen 50% pressure/flow missingness, attack strengths, generator data, and
canonical splits do not change. No EVAL or test set is read. Cached endpoints
start at hour 15, so earlier history is unavailable to this first pilot; it is
not imputed from another series or a clean trace.

## Interpretation and next decision

The report contains mixture/general/expert F1, precision, recall, clean FPR,
family-plus-clean AUPRC, and mean routing weights on attack positives. It can
show whether added history improves evidence and whether the learned mixture
loses information visible in a standalone expert.

This experiment evaluates the **five-expert mixture branch only**, not the
complete deployed warning, delayed-specialist, feedback, and verification
pipeline. The separate seasonal specialists and all deployed artifacts stay
unchanged. Its absolute F1 must not be compared directly to the 0.6757 replay
score of the full system on independent confirmation data.

A promising result would justify more source-held folds, full-system
calibration with protected-family checks, and only then a frozen independent
confirmation. A small or negative result is retained as an ablation; it does
not justify opening EVAL repeatedly. A second plausible direction is to train
the existing router against out-of-fold detection usefulness rather than
assuming the named family expert is always the best detector. That is a
separate intervention and has not been implemented in this pilot.

## Reproduction

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 MPLCONFIGDIR=/tmp/wdn-mpl-cache \
/opt/miniconda3/bin/python -u \
thesis_v2/experiments/early_warning/screen_shared_history.py
```

Output: `runs/operational/early_warning_multiseed_v1/shared_history_pilot_v1/ltown/fold_0_seed_701/`.
The protocol records code/input hashes; unchanged completed models are reused,
and a completed pilot is not retrained. Tests extend
`tests/test_delayed_decision_features.py` with causal, missingness, series
isolation, input-order, and original-training-objective checks.

## Completed pilot: 5 September 2026

The paired run completed in 288 seconds. Eighteen focused tests passed. The
candidate is **not promoted**: the additional residual-history features did not
produce a useful improvement on this diagnostic partition.

| TRAIN diagnostic, mixture branch | Original 116 features | Shared history, 172 features |
|---|---:|---:|
| Pooled F1 | 0.8914 | 0.8902 |
| Random F1 | 0.9964 | 0.9959 |
| Replay F1 | 0.7173 | 0.6954 |
| Drift F1 | 0.7917 | 0.7923 |
| Noise F1 | 0.7936 | 0.7966 |
| Targeted F1 | 0.9917 | 0.9917 |
| Clean FPR | 0.000292 | 0.000304 |

Standalone replay-expert F1 was 0.7492 / 0.7548, with recall 0.6114 / 0.6062.
Its family-plus-clean AUPRC changed from 0.5930 to 0.5851. A slightly higher
F1 at the chosen operating point therefore does not establish better ranking.
The diagnostic subset contained 193 replay-positive sensor-hours; these are
correlated observations, not 193 independent attack events.

The original mixture missed 75 replay-positive rows. On 11 of these, at least
one standalone expert alarmed at its own frozen global threshold. Across all
193 replay positives, 65 triggered none of those expert thresholds. Thus there
is both fusion loss and shared missed evidence at these operating points.
These counts are not an information-theoretic ceiling or proof that another
router or threshold could not improve recall. Expert thresholds differ from
the mixture threshold, and a simple alarm union would also change false alarms.

Artifacts: `summary.json`, `thresholds_frozen.json`, and
`expert_error_overlap.json` in the output directory above. The deployed
independent-confirmation replay figure remains 0.6757; **it did not fall to
0.6954 or rise to 0.7173**. Those latter values describe this different,
TRAIN-only branch diagnostic.

### Next hypothesis, not yet implemented

Do not scale up this failed feature recipe or assume a larger GRU will fix it.
Earlier GRU experiments already have mixed/negative evidence, documented in
`CAUSAL_SEQUENCE_EXPERT_RESULTS.md` and `EXPANDED_TRAIN_WEAK_EXPERT_RESULTS.md`.

A more direct information change is to preserve multi-lag differences between
**actually received sensor readings**, together with past/current reference
consistency and availability, in the common bank of every expert and router.
The earlier GNN mechanism model contained a continuous `self_lag_distance`
feature, but this 116-column residual tree bank does not. The present pilot
only added histories of residual summaries; that is a different intervention.

The next bounded pilot should compare this observation-consistency bank against
the unchanged 116-feature control on TRAIN folds. Use a common past horizon
for every family; no attack-family metadata, clean source readings, exact-copy
rule, or new expert. Any apparent gain should also survive a predefined sensor
quantisation/rounding sensitivity audit, reported separately from the unchanged
primary benchmark, before scaling to calibration and independent confirmation.
This is a testable candidate, not a prediction that replay will reach 0.80.
