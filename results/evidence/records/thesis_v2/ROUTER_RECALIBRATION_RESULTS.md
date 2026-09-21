# General Router Recalibration Results

## Scope

This experiment tested a general architectural change without adding an expert
or defining a replay-specific inference rule. The existing five-expert mixture,
router, feedback mechanism, drift/noise verifier, attack distribution, splits,
and 0.50 pressure/flow missing probabilities were retained. A single
temperature and uniform-shrinkage transform was applied symmetrically to all
router outputs. Inference never receives the attack-family label.

Hyperparameters and thresholds were selected independently for five training
seeds on L-Town calibration only. Six new generator seeds (100811--105811), 24
scenarios each, were reserved after the rules were frozen. Neither the earlier
Stage-E evaluation seeds nor the original locked test were used for selection.

## Negative preliminary result

Simply redistributing the clean false-alarm budget after feedback was not
feasible. On calibration seed 701 its best worst-family point raised replay F1
only from 0.6454 to 0.6539 while reducing pooled F1 from 0.8882 to 0.7626. This
screen is retained under `final_budget_recalibration_v2` as a negative result.

## Fresh paired confirmation

The unchanged Stage-E candidate and the frozen symmetric-router candidate were
evaluated on the same 30 training-seed/data-seed combinations.

| Metric | Stage E | Recalibrated | Paired change |
|---|---:|---:|---:|
| Pooled F1 | 0.8799 | **0.8841** | **+0.0042** |
| Random F1 | 0.9911 | 0.9903 | -0.0008 |
| Replay F1 | 0.6461 | **0.6757** | **+0.0296** |
| Drift F1 | 0.9081 | 0.9070 | -0.0012 |
| Noise F1 | 0.9270 | 0.9255 | -0.0016 |
| Targeted F1 | 0.9912 | 0.9895 | -0.0016 |
| Clean FPR | 0.000595 | **0.000583** | -0.000012 |

Replay and pooled F1 improved in all 30 paired comparisons. Every protected
family's mean reduction was below the predeclared 0.01 tolerance, and clean FPR
remained far below the fixed 0.005 ceiling. The general change therefore passes
its protection criteria and provides a repeatable but modest improvement.

It does **not** solve L-Town replay: all 30 recalibrated replay results remain
below 0.80 (mean 0.6757, range 0.6077--0.7287). This bounds what router-only
recalibration can recover from the current frozen expert evidence. The result
should be presented as a successful general fusion refinement, not as evidence
that the 0.80 replay objective was reached.

Machine-readable results and the paired table are in
`runs/operational/early_warning_multiseed_v1/router_temperature_confirmation_v1/ltown/`.
