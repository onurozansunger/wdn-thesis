# E10: supplementary analyses and retained flow studies

The original E1–E9 evidence register remains fixed at commit
[`49036da8016097f2ebb627b3b58cea1a07432a47`](https://github.com/onurozansunger/wdn-thesis/tree/49036da8016097f2ebb627b3b58cea1a07432a47).
E10 is a later supplement. When citing it, pin links to the commit that contains
this page rather than attributing the new analysis to the original commit.

The [supplement README](../results/supplementary_20261008/README.md) gives the
evaluation scope and verification command. Its [manifest](../results/supplementary_20261008/manifest.json)
maps all archived source paths to public locations with byte counts and hashes.

## Completed operating-point and component analyses

| Record | Purpose |
|---|---|
| [Frozen policy](../results/supplementary_20261008/analysis/operating_points/protocol.json) | Networks, model seeds, sources, calibration caps, threshold path, objectives and diagnostic variants |
| [Frozen analysis code hashes](../results/supplementary_20261008/analysis/operating_points/analysis_code_frozen.json) | Identify the code fixed for the completed analysis |
| [Extraction driver](../results/supplementary_20261008/analysis/extract_hybrid_scores.py) | Existing-model inference and reproduction checks used to recover branch outputs |
| [Comparison driver](../results/supplementary_20261008/analysis/compare_operating_points.py) | Calibration selection, evaluation and aggregation under the frozen policy |
| [Extraction sidecars](../results/supplementary_20261008/analysis/hybrid_scores) | 210 per-source provenance/check records and 20 calibration-validation records |
| [Per-fit choices and evaluation cells](../results/supplementary_20261008/analysis/operating_points) | 20 calibration choices and 120 completed evaluation reports |
| [Complete summary](../results/supplementary_20261008/analysis/operating_points/complete_summary.json) | Equal-weight cell means, model/source summaries and paired differences |
| [Report and figures](../results/supplementary_20261008/analysis/report) | CSV tables, all declared comparison outcomes and two network plots |

Original-rule reproduction uses the historical records available for each
network. Modena checks exact saved pooled confusion counts and rounded family F1
values; no historical per-endpoint decision vector was retained for a direct
vector comparison. L-Town checks exact family and pooled counts and the retained
specialist union. The new analyses reuse the original evaluation sources.

## Three flow-feature development studies

| Study | Protocol and report | Retained detailed records | Interpretation |
|---|---|---|---|
| Modena weak-family context comparison | [Training protocol](../results/supplementary_20261008/records/thesis_v2/WEAK_FAMILY_TRAINING_PROTOCOL.md), [results](../results/supplementary_20261008/records/thesis_v2/WEAK_FAMILY_RESULTS.md) | [Splits, frozen selection and four trial reports](../results/supplementary_20261008/records/runs/operational/weak_family_v1) | Pressure/flow context was evaluated on a small development population; pressure context was selected on calibration. The reused validation set did not trigger reselection. |
| L-Town flow/fusion replay comparison | [Protocol](../results/supplementary_20261008/records/thesis_v2/experiments/early_warning/FLOW_FUSION_EXPERIMENT.md), [results](../results/supplementary_20261008/records/thesis_v2/outputs/tables/ltown_flow_fusion_experiment.md) | [Flow/fusion development archive](../results/supplementary_20261008/records/runs/operational/early_warning_multiseed_v1/flow_fusion_v1) | The flow candidate passed the source-held TRAIN progression gate but no full-system fit had a calibration operating point satisfying the protection constraints. No fresh confirmation followed. |
| L-Town joint flow/received-trajectory comparison | [Prospective plan](../results/supplementary_20261008/records/thesis_v2/experiments/early_warning/REPLAY_TARGET_075_PLAN.md), [results](../results/supplementary_20261008/records/thesis_v2/outputs/tables/ltown_received_joint_075.md) | [Frozen joint-candidate records](../results/supplementary_20261008/records/runs/operational/early_warning_multiseed_v1/received_joint_075_v1) | Calibration protections passed, but the replay gain missed the declared progression requirement. No fresh confirmation followed. |

The reports distinguish General-only TRAIN diagnostics, full-system calibration
and independent confirmation populations. Flow candidates were not selected for
the final detector. Their rejection does not establish that flow is generally
unhelpful. The unchanged detector inputs are pressures; flow still has an
indirect role in the generator's shared snapshot family codes.

## Audit limits

`verify.py` checks the files supplied here and recomputes summary arithmetic.
Recorded hashes for omitted arrays and models establish consistency among
records, not an independent verification of absent bytes or a rerun of inference.
The eight frozen-rule component variants do not replace a factorial experiment
that varies target-group exclusion and robust fitting independently.
