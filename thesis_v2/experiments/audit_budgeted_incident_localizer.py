"""Independent metric, budget-selection, and artifact audit for the budgeted localizer."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wdn.run_expert_redesign import sha, write_json


RUN = Path("runs/operational/budgeted_incident_localizer_v1")
BASE = Path("runs/operational/mechanism_redesign_v1/full")
OUTPUT = Path("thesis_v2/outputs/budgeted_incident_localizer_results.json")
REPORT = Path("thesis_v2/BUDGETED_INCIDENT_LOCALIZER_RESULTS.md")


def metrics(labels, decisions):
    labels, decisions = np.asarray(labels, bool), np.asarray(decisions, bool)
    tp = int(np.sum(labels & decisions)); fp = int(np.sum(~labels & decisions))
    fn = int(np.sum(labels & ~decisions))
    return {"f1": float(2*tp/max(2*tp+fp+fn, 1)),
        "precision": float(tp/max(tp+fp, 1)), "recall": float(tp/max(tp+fn, 1)),
        "tp": tp, "fp": fp, "fn": fn}


def reproduce_top_k(base, saved, top_k):
    rows, score = saved["row"].astype(int), saved["score"]
    decision = np.zeros(len(rows), dtype=bool)
    for incident in np.unique(base["event"][rows]):
        local_positions = np.flatnonzero(base["event"][rows] == incident)
        nodes = np.unique(base["node"][rows[local_positions]])
        ranked = []
        for node in nodes:
            positions = local_positions[base["node"][rows[local_positions]] == node]
            if not np.all(score[positions] == score[positions[0]]):
                raise ValueError("Saved fused score must be constant within incident/sensor")
            ranked.append((float(score[positions[0]]), int(node), positions))
        for _, _, positions in sorted(ranked, key=lambda item: (-item[0], item[1]))[:top_k]:
            decision[positions] = True
    return decision


def run():
    summary = json.loads((RUN/"summary.json").read_text())
    selection = json.loads((RUN/"selection_frozen.json").read_text())
    calibration = dict(np.load(RUN/"predictions_calibration.npz"))
    validation = dict(np.load(RUN/"predictions_validation.npz"))
    calibration_base = dict(np.load(BASE/"features_calibration.npz"))
    validation_base = dict(np.load(BASE/"features_validation.npz"))
    choices = {}
    for top_k in range(1, summary["selection"]["maximum_pressure_targets"]+1):
        decision = reproduce_top_k(calibration_base, calibration, top_k)
        choices[str(top_k)] = metrics(calibration["label"], decision)
    selected_k = max(range(1, 15), key=lambda top_k:
        (choices[str(top_k)]["f1"], choices[str(top_k)]["precision"], -top_k))
    validation_decision = reproduce_top_k(validation_base, validation, selected_k)
    validation_metrics = metrics(validation["label"], validation_decision)
    checks = {"selection_file_exact": selection == summary["selection"],
        "calibration_hash_exact": sha(RUN/"predictions_calibration.npz") ==
            summary["prediction_hashes"]["calibration"],
        "validation_hash_exact": sha(RUN/"predictions_validation.npz") ==
            summary["prediction_hashes"]["validation"],
        "calibration_choices_exact": choices == summary["selection"]["calibration_choices"],
        "selected_k_exact": selected_k == summary["selection"]["top_k"],
        "saved_calibration_decision_exact": np.array_equal(
            calibration["decision"], reproduce_top_k(calibration_base, calibration, selected_k)),
        "saved_validation_decision_exact": np.array_equal(
            validation["decision"], validation_decision),
        "noise_validation_metrics_exact": validation_metrics ==
            summary["development_validation"]["noise"],
        "selection_precedes_validation_predictions":
            (RUN/"selection_frozen.json").stat().st_mtime_ns <=
            (RUN/"predictions_validation.npz").stat().st_mtime_ns,
        "test_not_evaluated": summary["test_evaluated"] is False,
        "both_development_targets_pass": summary["target_0_80_both_achieved"] is True,
        "reused_validation_disclosed":
            summary["reused_development_validation_not_independent"] is True}
    result = {"independent_recalculation": True, "checks": checks,
        "all_checks_pass": all(checks.values()), "selected_top_k": selected_k,
        "development_validation": summary["development_validation"],
        "claim_limits": {"reused_development_validation": True,
            "conditional_on_external_window_and_family": True,
            "fixed_maximum_pressure_targets": 14, "test_evaluated": False}}
    if not result["all_checks_pass"]:
        raise ValueError("Budgeted incident localizer audit failed")
    write_json(OUTPUT, result); write_json(RUN/"independent_audit.json", result)
    drift = result["development_validation"]["drift"]
    noise = result["development_validation"]["noise"]
    lines = ["# Budgeted incident localizer results", "",
        "The saved predictions, calibration budget selection, and metrics were independently",
        "recalculated exactly. The locked test split was not read.", "",
        "| Family | Development F1 | Precision | Recall | TP | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| drift | {drift['f1']:.4f} | {drift['precision']:.4f} | {drift['recall']:.4f} | {drift['tp']} | {drift['fp']} | {drift['fn']} |",
        f"| noise | {noise['f1']:.4f} | {noise['precision']:.4f} | {noise['recall']:.4f} | {noise['tp']} | {noise['fp']} | {noise['fn']} |",
        "", f"Calibration selected top_k={selected_k} under the predeclared maximum of 14 pressure targets.",
        "Noise fuses equal within-incident ranks of the three-member OOF expert committee and",
        "the dynamic-innovation consecutive-difference statistic.", "",
        "Both development targets pass, but this is reused development validation obtained after",
        "several diagnostic iterations. It is not independent confirmation. The model also requires",
        "a correct external incident window and family hypothesis and uses future observations.",
        "It must be presented as conditional retrospective localisation, not online detection.", ""]
    REPORT.write_text("\n".join(lines))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
