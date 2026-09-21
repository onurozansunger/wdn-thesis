"""Summarise operational development runs without evaluating any model/test set."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ATTACK_FAMILIES = ("random", "replay", "stealthy", "noise", "targeted")


def main():
    run_root = ROOT / "runs/operational"
    rows = []
    for args_path in sorted(run_root.rglob("args.json")):
        run_dir = args_path.parent
        args = json.loads(args_path.read_text())
        summary_path = run_dir / "summary.json"
        history_path = run_dir / "history.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            v = summary["validation"]
            epoch, state = summary["best_epoch"], "completed"
        elif history_path.exists():
            history = json.loads(history_path.read_text())
            candidates = [e for e in history if e.get("stage") != "expert_warmup"]
            if not candidates:
                continue
            selected = max(candidates, key=lambda e: e["validation"]["target_score"])
            v = selected["validation"]
            epoch, state = selected["epoch"], f"partial ({len(history)}/{args['epochs']})"
        else:
            continue
        replay = v["per_family"].get("replay", {})
        audit_path = run_dir / "expert_audit.json"
        expert_metrics = {}
        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
            if audit["checkpoint_epoch"] != epoch:
                raise ValueError(f"Stale expert audit in {run_dir}")
            for name, metrics in audit["expert_matrix"].items():
                family_metrics = metrics["per_family"][name]
                expert_metrics[name] = {
                    "own_family_auprc": family_metrics["auprc"],
                    "clean_fpr": metrics["per_family"]["clean"]["fpr"],
                    "threshold": metrics["threshold"],
                }
        rows.append({"run": str(run_dir.relative_to(ROOT)), "model": args["model"],
                     "state": state, "best_epoch": epoch, "overall_f1": v["overall"]["f1"],
                     "replay_f1": replay.get("f1"), "auprc": v["overall"]["auprc"],
                     "replay_positives": replay.get("tp", 0)+replay.get("fn", 0),
                     "families": sorted(v["per_family"]),
                     "fpr": v["overall"]["fpr"],
                     "clean_fpr": v["per_family"].get("clean", {}).get("fpr"),
                     "threshold": v["threshold"], "expert_metrics": expert_metrics})
    out = ROOT / "thesis_v2/outputs"
    (out/"operational_results.json").write_text(json.dumps(rows, indent=2))
    lines = ["# Operational MoE development results", "",
        "Detector tables use development validation; the recovery-screen section is explicitly TRAIN-only.",
        "Detector thresholds are chosen on separate calibration scenarios unless a diagnostic is explicitly labelled otherwise.",
        "No scores below establish performance on the locked test or on field data.", "",
        "| Run | State | Best epoch | Overall F1 | Replay F1 | AUPRC | Sensor FPR | Replay positives |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['run']} | {row['state']} | {row['best_epoch']} | "
                     f"{row['overall_f1']:.4f} | {row['replay_f1']:.4f} | "
                     f"{row['auprc']:.4f} | {row['fpr']:.5f} | {row['replay_positives']} |")
    audited = [row for row in rows if row["expert_metrics"]]
    if audited:
        lines += ["", "## Independent expert diagnostics", "",
            "Own-family AUPRC without routing. These scores do not establish that the owner",
            "beats every other expert; the full comparison is in each run's expert_audit.md.", "",
            "| Run | Random | Replay | Drift | Noise | Targeted |",
            "|---|---:|---:|---:|---:|---:|"]
        for row in audited:
            values = [f"{row['expert_metrics'][f]['own_family_auprc']:.4f}" for f in ATTACK_FAMILIES]
            lines.append(f"| {row['run']} | " + " | ".join(values) + " |")
        lines += ["", "Expert clean-episode sensor FPR, using each expert's single global",
            "calibration threshold; these thresholds are not tuned on validation families.", "",
            "| Run | General | Random | Replay | Drift | Noise | Targeted |",
            "|---|---:|---:|---:|---:|---:|---:|"]
        for row in audited:
            values = [f"{row['expert_metrics'][f]['clean_fpr']:.5f}" for f in ("clean", *ATTACK_FAMILIES)]
            lines.append(f"| {row['run']} | " + " | ".join(values) + " |")
    probes = []
    for path in sorted(run_root.glob("blind*/summary.json")):
        report = json.loads(path.read_text())
        if report.get("test_evaluated"):
            raise ValueError("Development probe unexpectedly evaluated test")
        for method, result in report["results"].items():
            probes.append({"run": str(path.parent.relative_to(ROOT)), "method": method,
                "scope": report["status"], "overall_f1": result["overall"]["f1"],
                "replay_f1": result["per_family"]["replay"]["f1"],
                "auprc": result["overall"]["auprc"], "fpr": result["overall"]["fpr"],
                "threshold": result["threshold"], "per_family": result["per_family"]})
    (out/"operational_probes.json").write_text(json.dumps(probes, indent=2))
    if probes:
        lines += ["", "## Blind-reference exploratory prototypes", "",
            "Separate pressure-only prototypes, not changes to the trained temporal GNN.",
            "Same calibration/validation examples and 50% missing; no test evaluation.",
            "Three reference ranks were explored; results are development evidence only.",
            "See BLIND_REFERENCE_PROTOTYPE.md for methods, independent expert matrices and limitations.", "",
            "| Run | Detector | Overall F1 | Replay F1 | AUPRC | Sensor FPR |",
            "|---|---|---:|---:|---:|---:|"]
        for row in probes:
            lines.append(f"| {row['run']} | {row['method']} | {row['overall_f1']:.4f} | "
                         f"{row['replay_f1']:.4f} | {row['auprc']:.4f} | {row['fpr']:.6f} |")
    family_path = out/"family_balance_results.json"
    if family_path.exists():
        family_report = json.loads(family_path.read_text())
        if family_report["test_evaluated"]:
            raise ValueError("Family development probe unexpectedly evaluated test")
        lines += ["", "## Weak-family follow-up on frozen data", "",
            "Causal residual context and family-balanced calibration; separate pressure-only tree probes.",
            "All variants are disclosed, not a validation-selected final winner. The tail-fusion Optuna",
            "winner was selected on calibration but worsened drift versus the dynamic balanced point.",
            "See FAMILY_BALANCE_RESULTS.md for recall, calibration objectives, expert audit and limitations.", "",
            "| Variant | Overall F1 | Random | Replay | Drift | Noise | Targeted | Sensor FP |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in family_report["comparisons"]:
            v = row["validation"]
            values = " | ".join(f"{v['per_family'][f]['f1']:.4f}" for f in ATTACK_FAMILIES)
            lines.append(f"| {row['name']} | {v['overall']['f1']:.4f} | {values} | {v['overall']['fp']} |")
        lines += ["", "Higher weak-family F1 comes with more false positives. Even the balanced point misses",
            "most drift/noise positive readings; F1 above 0.5 is not comprehensive attack coverage.",
            "The original four full GNN Optuna trials were not extended; the separate 40-trial search",
            "only calibrated frozen expert scores and performed no model training."]
        if (ROOT/"thesis_v2/EXPERT_REDESIGN_ANALYSIS.md").exists():
            lines += ["", "Further event-level diagnosis and proposed modifications: EXPERT_REDESIGN_ANALYSIS.md.",
                "Its oracle reference exclusion is diagnostic only, not a deployable detector or a new F1 result."]
    redesign_path = out/"expert_redesign_results.json"
    if redesign_path.exists():
        redesign = json.loads(redesign_path.read_text())
        if redesign["verification"]["test_evaluated"]:
            raise ValueError("Redesign report unexpectedly evaluated test")
        chosen = redesign["selection"]["trial"]
        lines += ["", "## Robust reference and sequential expert redesign", "",
            f"Calibration-selected trial: {chosen}. All trials below are development validation results.",
            "Three inner scenario folds train the combiner from held-out expert scores; no validation tuning or test evaluation.",
            "See EXPERT_REDESIGN_RESULTS.md for the independent expert audit, event metrics, controls and limitations.", "",
            "| Variant | Overall F1 | Random | Replay | Drift | Noise | Targeted | Sensor FP |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in redesign["comparisons"]:
            if "trial" not in row:
                continue
            v = row["validation"]
            values = " | ".join(f"{v['per_family'][f]['f1']:.4f}" for f in ATTACK_FAMILIES)
            lines.append(f"| {row['name']} | {v['overall']['f1']:.4f} | {values} | {v['overall']['fp']} |")
    weak_path = out/"weak_family_results.json"
    if weak_path.exists():
        weak = json.loads(weak_path.read_text())
        if weak["verification"]["test_extracted"]:
            raise ValueError("Weak-family campaign unexpectedly extracted test")
        lines += ["", "## Drift/noise target campaign", "",
            f"Calibration-selected candidate: {weak['selection']['trial']} ({weak['selection']['mode']}).",
            "New bounded campaign; old studies are unchanged. All values are reused development validation, not test results.",
            f"Both drift/noise F1 >= 0.80: {weak['target_both_achieved']}. See WEAK_FAMILY_RESULTS.md for audits and limits.", "",
            "| Variant | Overall F1 | Random | Replay | Drift | Noise | Targeted | Sensor FP |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in weak["comparisons"]:
            v = row["validation"]
            values = " | ".join(f"{v['per_family'][f]['f1']:.4f}" for f in ATTACK_FAMILIES)
            lines.append(f"| {row['name']} | {v['overall']['f1']:.4f} | {values} | {v['overall']['fp']} |")
    recovery_path = out/"drift_noise_recovery_results.json"
    if recovery_path.exists():
        recovery = json.loads(recovery_path.read_text())
        if recovery["validation_evaluated"] or recovery["test_evaluated"]:
            raise ValueError("Recovery screen is restricted to TRAIN diagnostics")
        lines += ["", "## Physical reference and marked change recovery screen", "",
            "TRAIN-only held-scenario specialist AP, not new validation or test F1.",
            f"Decision: {recovery['next_action']}. Optuna trials started: {recovery['optuna_trials']}.",
            "The prior validation scores above remain unchanged; no full-stack improvement is implied.",
            "See DRIFT_NOISE_RECOVERY_RESULTS.md for reference accuracy, early recall, false alarms and prespecified gates.", "",
            "| Component | Drift mean scenario AP | Drift worst AP | Noise mean scenario AP | Noise worst AP |",
            "|---|---:|---:|---:|---:|"]
        for name in ("control", "physics", "state"):
            row = recovery["experts"][name]
            lines.append(f"| {name} | {row['drift']['macro_ap']:.4f} | {row['drift']['worst_ap']:.4f} | "
                         f"{row['noise']['macro_ap']:.4f} | {row['noise']['worst_ap']:.4f} |")
    nuisance_path = out/"normal_nuisance_stage_a_results.json"
    if nuisance_path.exists():
        nuisance_report = json.loads(nuisance_path.read_text())
        nuisance = nuisance_report["summary"]
        audit = nuisance_report["independent_audit"]
        if nuisance["calibration_evaluated"] or nuisance["validation_evaluated"] or nuisance["test_evaluated"]:
            raise ValueError("Normal nuisance Stage A is restricted to TRAIN diagnostics")
        required_checks = (
            "saved_oof_metrics_reproduced_exact",
            "normal_gate_reproduced_exact",
            "evidence_gates_reproduced_exact",
            "artifact_hashes_match",
            "metadata_matches_frozen_control",
            "accessed_train_scenarios_only",
        )
        if not all(audit[name] for name in required_checks):
            raise ValueError("Normal nuisance Stage A independent audit is incomplete")
        normal_b = nuisance["normal_diagnostics"]["B"]["clean"]["scenario_macro"]
        normal_m = nuisance["normal_diagnostics"]["M"]["clean"]["scenario_macro"]
        serial = nuisance_report["posthoc_diagnostics"]["same_sensor_consecutive_hour_residual"]
        lines += ["", "## Conditional normal nuisance Stage A", "",
            "TRAIN-only nested OOF diagnostics; no calibration, validation or test extraction.",
            f"Decision: {nuisance['next_action']}. Surviving families: {len(nuisance['survivors'])}.",
            f"Conditional mean M/B MAE ratio: {nuisance['normal_gate']['macro_mae_ratio']:.4f}; "
            f"scenarios improved: {nuisance['normal_gate']['improved_scenarios']}/14.",
            "Cross-fitted scale corrected overconfidence but did not improve weak-attack evidence; no weak head,",
            "Optuna, physical Stage B, calibration or validation run followed. See NORMAL_NUISANCE_STAGE_A_RESULTS.md.", "",
            "| Arm | Normal macro MAE | Normal macro NLL | abs(z)>3 | Drift macro AP | Noise macro AP |",
            "|---|---:|---:|---:|---:|---:|",
            f"| B | {normal_b['mae_m']:.4f} | {normal_b['nll']:.4f} | {normal_b['fraction_abs_z_gt3']:.4f} | "
            f"{nuisance['evidence_diagnostics']['B']['drift']['macro_ap']:.4f} | "
            f"{nuisance['evidence_diagnostics']['B']['noise']['macro_ap']:.4f} |",
            f"| M | {normal_m['mae_m']:.4f} | {normal_m['nll']:.4f} | {normal_m['fraction_abs_z_gt3']:.4f} | "
            f"{nuisance['evidence_diagnostics']['M']['drift']['macro_ap']:.4f} | "
            f"{nuisance['evidence_diagnostics']['M']['noise']['macro_ap']:.4f} |",
            f"| local control | - | - | - | "
            f"{nuisance['evidence_diagnostics']['local_control']['drift']['macro_ap']:.4f} | "
            f"{nuisance['evidence_diagnostics']['local_control']['noise']['macro_ap']:.4f} |", "",
            f"Post-hoc clean lag-1 residual correlation remained {serial['B']['pooled_pearson_r']:.3f} for B and "
            f"{serial['M']['pooled_pearson_r']:.3f} for M. This was not a gate; it motivates a separate causal latent-AR screen."]
    latent_path = out/"latent_ar_stage_b_results.json"
    if latent_path.exists():
        latent_report = json.loads(latent_path.read_text())
        latent = latent_report["summary"]
        audit = latent_report["independent_audit"]
        if latent["calibration_evaluated"] or latent["validation_evaluated"] or latent["test_evaluated"]:
            raise ValueError("Latent AR Stage B is restricted to TRAIN diagnostics")
        required_checks = ("saved_oof_metrics_reproduced_exact", "normal_gate_reproduced_exact",
            "evidence_gates_reproduced_exact", "lag_diagnostics_reproduced_exact",
            "artifact_hashes_match", "stage_a_metadata_and_baseline_exact",
            "accessed_train_scenarios_only")
        if not all(audit[name] for name in required_checks):
            raise ValueError("Latent AR Stage B independent audit is incomplete")
        bnormal = latent["normal_diagnostics"]["B"]["clean"]["scenario_macro"]
        anormal = latent["normal_diagnostics"]["A"]["clean"]["scenario_macro"]
        lines += ["", "## Causal latent AR Stage B", "",
            "Separate TRAIN-only nested OOF component screen; no calibration, validation or test extraction.",
            f"Decision: {latent['next_action']}. Surviving families: {len(latent['survivors'])}.",
            f"Normal MAE improved in {latent['normal_gate']['improved_scenarios']}/14 scenarios, but interval width "
            f"was {latent['normal_gate']['macro_mean_width95_ratio']:.3f}x B versus the frozen 1.25x limit.",
            "No weak head, fusion or Optuna run followed. See LATENT_AR_STAGE_B_RESULTS.md.", "",
            "| Arm | Normal MAE | Normal NLL | lag-1 r | Drift macro AP | Noise macro AP |",
            "|---|---:|---:|---:|---:|---:|",
            f"| B | {bnormal['mae_m']:.4f} | {bnormal['nll']:.4f} | "
            f"{latent['lag_one_diagnostics']['B']['pooled_pearson_r']:.3f} | "
            f"{latent['evidence_diagnostics']['B']['drift']['macro_ap']:.4f} | "
            f"{latent['evidence_diagnostics']['B']['noise']['macro_ap']:.4f} |",
            f"| A | {anormal['mae_m']:.4f} | {anormal['nll']:.4f} | "
            f"{latent['lag_one_diagnostics']['A']['pooled_pearson_r']:.3f} | "
            f"{latent['evidence_diagnostics']['A']['drift']['macro_ap']:.4f} | "
            f"{latent['evidence_diagnostics']['A']['noise']['macro_ap']:.4f} |",
            f"| local control | - | - | - | "
            f"{latent['evidence_diagnostics']['local_control']['drift']['macro_ap']:.4f} | "
            f"{latent['evidence_diagnostics']['local_control']['noise']['macro_ap']:.4f} |"]
    family_specific_path = out/"family_specific_experts_results.json"
    if family_specific_path.exists():
        family_report = json.loads(family_specific_path.read_text())
        family = family_report["summary"]
        audit = family_report["independent_audit"]
        if family["calibration_evaluated"] or family["validation_evaluated"] or family["test_evaluated"]:
            raise ValueError("Family-specific screen is restricted to TRAIN diagnostics")
        required_checks = ("source_hashes_match", "input_hashes_match",
            "missing_rates_pressure_and_flow_equal_0_50", "prediction_artifact_hashes_match",
            "saved_metrics_and_gates_reproduced_exact", "bundle_reload_predictions_exact",
            "accessed_train_scenarios_only", "no_forbidden_scenario_in_oof")
        if not all(audit[name] for name in required_checks):
            raise ValueError("Family-specific expert independent audit is incomplete")
        lines += ["", "## Separate fast/persistent drift and noise experts", "",
            "TRAIN-only nested scenario-OOF component screen; no calibration, validation or test extraction.",
            f"Decision: {family['next_action']}. Surviving families: {len(family['survivors'])}.",
            "Two family-specific nonlinear heads used event-balanced positives, active-event hard negatives,",
            "clean rows and other-family negatives; their stackers used only inner-OOF scores.",
            "Neither family passed. See FAMILY_SPECIFIC_EXPERTS_RESULTS.md.", "",
            "| Family | Legacy macro AP | Separate expert macro AP | Separate oracle F1 | Early recall @.001 |",
            "|---|---:|---:|---:|---:|",
            f"| drift | {family['diagnostics']['L']['drift']['macro_ap']:.4f} | "
            f"{family['diagnostics']['E']['drift']['macro_ap']:.4f} | "
            f"{family['family_oracle_f1']['drift']['f1']:.4f} | "
            f"{family['diagnostics']['E']['drift']['early_macro_recall']:.4f} |",
            f"| noise | {family['diagnostics']['L']['noise']['macro_ap']:.4f} | "
            f"{family['diagnostics']['E']['noise']['macro_ap']:.4f} | "
            f"{family['family_oracle_f1']['noise']['f1']:.4f} | "
            f"{family['diagnostics']['E']['noise']['early_macro_recall']:.4f} |"]
    incident_path = out/"incident_window_localizer_results.json"
    if incident_path.exists():
        incident = json.loads(incident_path.read_text())
        if not incident["all_checks_pass"] or incident["interpretation"]["test_evaluated"]:
            raise ValueError("Incident-window audit is incomplete or touched test")
        online = incident["online_development_validation_unchanged"]
        train = incident["conditional_incident_window_train_nested"]
        validation = incident["conditional_incident_window_development_validation"]
        lines += ["", "## Conditional incident-window localisation", "",
            "This separate offline stage assumes a correct external incident window and family hypothesis.",
            "It uses future observations and is not an online detector. All saved metrics were independently recalculated; test remains locked.", "",
            "| Evaluation | Drift F1 | Noise F1 |", "|---|---:|---:|",
            f"| Existing online development validation | {online['drift_f1']:.4f} | {online['noise_f1']:.4f} |",
            f"| Incident-window nested TRAIN | {train['drift']['f1']:.4f} | {train['noise']['f1']:.4f} |",
            f"| Incident-window development validation | {validation['drift']['f1']:.4f} | {validation['noise']['f1']:.4f} |",
            "", "The conditional TRAIN target passed, but only drift reached 0.80 on development validation.",
            "Noise did not generalise, so the joint 0.80 target is not achieved. See INCIDENT_WINDOW_LOCALIZER_RESULTS.md."]
    budgeted_path = out/"budgeted_incident_localizer_results.json"
    if budgeted_path.exists():
        budgeted = json.loads(budgeted_path.read_text())
        if not budgeted["all_checks_pass"] or budgeted["claim_limits"]["test_evaluated"]:
            raise ValueError("Budgeted incident localizer audit is incomplete or touched test")
        drift = budgeted["development_validation"]["drift"]
        noise = budgeted["development_validation"]["noise"]
        lines += ["", "## Post-development budgeted incident localizer", "",
            "A later bounded candidate combines equal within-incident ranks from the three-member",
            "cross-fitted noise committee and a physical variance-change statistic. Calibration selected",
            f"top_k={budgeted['selected_top_k']} under the fixed threat-model maximum of 14 pressure sensors.", "",
            "| Family | Development F1 | Precision | Recall |", "|---|---:|---:|---:|",
            f"| drift | {drift['f1']:.4f} | {drift['precision']:.4f} | {drift['recall']:.4f} |",
            f"| noise | {noise['f1']:.4f} | {noise['precision']:.4f} | {noise['recall']:.4f} |",
            "", "Both development targets pass and the saved metrics reproduce exactly. However, this",
            "candidate was designed after repeated use of development validation. It requires an external",
            "window and family hypothesis, uses future observations, and is not independent confirmation or",
            "online performance. Test remains locked. See BUDGETED_INCIDENT_LOCALIZER_RESULTS.md."]
    causal_path = out/"causal_deadline_localizer_results.json"
    if causal_path.exists():
        causal = json.loads(causal_path.read_text())
        if not causal["all_checks_pass"] or causal["claim_limits"]["test_evaluated"]:
            raise ValueError("Causal deadline localizer audit is incomplete or touched test")
        summary = causal["summary"]
        calibration = summary["calibration"]
        validation = summary["reused_development_validation"]
        strict = summary["unchanged_strict_pointwise_online"]
        lines += ["", "## Causal tenth-checkpoint or closure localisation", "",
            "A post-alarm candidate emits one sensor set using only the prefix available at the",
            "earlier of ten hourly checkpoints or incident closure; it never backfills earlier rows.",
            "Both validation incidents closed before ten steps, so these are delayed closure decisions.", "",
            "| Evaluation | Drift sensor-event F1 | Noise sensor-event F1 |", "|---|---:|---:|",
            f"| Calibration | {calibration['drift']['f1']:.4f} | {calibration['noise']['f1']:.4f} |",
            f"| Reused development validation | {validation['drift']['f1']:.4f} | {validation['noise']['f1']:.4f} |",
            f"| Unchanged strict row/hour online | {strict['drift_f1']:.4f} | {strict['noise_f1']:.4f} |", "",
            "The 0.70 goal is reached only for the sensor-event checkpoint metric. The strict online",
            "point metric did not improve. External onset/family are required, development validation",
            "is reused, and test remains locked. See CAUSAL_DEADLINE_LOCALIZER_RESULTS.md."]
    sequence_path = out/"causal_sequence_experts_results.json"
    if sequence_path.exists():
        sequence = json.loads(sequence_path.read_text())
        if not sequence["all_checks_pass"] or sequence["claim_limits"]["test_evaluated"]:
            raise ValueError("Causal sequence expert audit is incomplete or touched test")
        summary = sequence["summary"]
        base = summary["baseline"]
        drift = summary["candidates"][summary["selection"]["drift"]]
        noise = summary["candidates"][summary["selection"]["noise"]]
        fusion = sequence["fusion_screen"]
        lines += ["", "## Causal drift/noise GRU component screen", "",
            "Two unidirectional GRU capacities and four past-only score-memory choices were evaluated",
            "with whole-scenario TRAIN OOF predictions. Both failed promotion and no later split was read.", "",
            "| Family | Frozen macro AP | Best GRU macro AP | Frozen oracle F1 | Best GRU oracle F1 |",
            "|---|---:|---:|---:|---:|",
            f"| drift | {base['diagnostics']['drift']['macro_ap']:.4f} | "
            f"{drift['diagnostics']['drift']['macro_ap']:.4f} | "
            f"{base['family_oracle_f1']['drift']['f1']:.4f} | "
            f"{drift['family_oracle_f1']['drift']['f1']:.4f} |",
            f"| noise | {base['diagnostics']['noise']['macro_ap']:.4f} | "
            f"{noise['diagnostics']['noise']['macro_ap']:.4f} | "
            f"{base['family_oracle_f1']['noise']['f1']:.4f} | "
            f"{noise['family_oracle_f1']['noise']['f1']:.4f} |", "",
            f"The final cross-fitted score fusion reached noise macro AP "
            f"{fusion['results'][fusion['selected']['noise']]['macro_ap']:.4f}, but the gain was below "
            "the frozen meaningful-improvement gate and drift worsened. The online model remains unchanged.",
            "See CAUSAL_SEQUENCE_EXPERT_RESULTS.md."]
    expanded_path = out/"expanded_train_weak_experts_results.json"
    if expanded_path.exists():
        expanded = json.loads(expanded_path.read_text())
        if not expanded["all_checks_pass"] or expanded["claim_limits"]["test_evaluated"]:
            raise ValueError("Expanded TRAIN expert audit is incomplete or touched test")
        base = expanded["base_summary"]
        tuning = expanded["optuna_summary"]
        control = base["candidates"]["mechanism"]
        winner = tuning["winner_report"]
        lines += ["", "## Multi-seed expanded TRAIN weak-family screen", "",
            "Three predeclared generator seeds added 48 TRAIN-only scenarios without changing the",
            "attack distribution or 50% pressure/flow missingness. Four source-held folds and exactly",
            "four Optuna tree trials were independently audited; calibration, validation and test stayed closed.", "",
            "| Family | TRAIN events | Mechanism macro AP | Winner macro AP | Mechanism oracle F1 | Winner oracle F1 |",
            "|---|---:|---:|---:|---:|---:|",
            f"| drift | {base['event_counts']['total']['stealthy']} | "
            f"{control['diagnostics']['drift']['macro_ap']:.4f} | "
            f"{winner['diagnostics']['drift']['macro_ap']:.4f} | "
            f"{control['family_oracle_f1']['drift']['f1']:.4f} | "
            f"{winner['family_oracle_f1']['drift']['f1']:.4f} |",
            f"| noise | {base['event_counts']['total']['noise']} | "
            f"{control['diagnostics']['noise']['macro_ap']:.4f} | "
            f"{winner['diagnostics']['noise']['macro_ap']:.4f} | "
            f"{control['family_oracle_f1']['noise']['f1']:.4f} | "
            f"{winner['family_oracle_f1']['noise']['f1']:.4f} |", "",
            "Noise crossed 0.70 OOF oracle F1, while drift reached 0.6706. Noise still failed the",
            "post-event false-alarm gate, so neither candidate was promoted to calibration. These",
            "threshold-oracle TRAIN scores are not deployable validation results. See",
            "EXPANDED_TRAIN_WEAK_EXPERT_RESULTS.md."]
    lines += ["", "## Interpretation limits", "",
        "- The two initial Optuna smoke trials used an earlier split that omitted drift from validation;",
        "  they test software only and are not comparable with the corrected development pilots.",
        "- Corrected pilot splits contain every attack family in every partition.",
        "- A single expert and a six-expert MoE are not parameter-matched. These are initial controls,",
        "  not evidence that any gain is caused by specialisation.",
        "- All scores use the fixed 50% missing protocol. Physical parameters were not tuned to scores.",
        "- Sensor/noise and episode settings remain provisional assumptions; see OPERATIONAL_MOE_PLAN.md.",
        "- Few replay positives or events require independent-data confirmation before interpreting a 0.50 crossing.",
        "- A partial run has no completion claim; the table uses its best validation checkpoint so far.", ""]
    (out/"operational_results.md").write_text("\n".join(lines))
    print(out/"operational_results.md")


if __name__ == "__main__":
    main()
