"""Freeze and evaluate the sparse-budget incident localizer on development data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit

from evaluate_incident_window_localizer import BASE, DATA, metrics, state_arrays
from wdn.models.budgeted_incident import budgeted_noise_rank_score, top_k_sensor_decision
from wdn.run_expert_redesign import sha, write_json


MEMBERS = Path("runs/operational/family_specific_experts_v1")
DRIFT = Path("runs/operational/incident_window_validation_v1")
PROTOCOL = Path("thesis_v2/BUDGETED_INCIDENT_LOCALIZER_PROTOCOL.md")
MAX_TARGETS = 14


def committee_score(arrays, bundles):
    scores = []
    for bundle in bundles:
        component = bundle["noise"]
        heads = component["expert"].predict_heads(arrays["X"])
        scores.append(component["stacker"].predict(heads))
    scores = np.clip(np.column_stack(scores), 1e-6, 1-1e-6)
    return expit(np.log(scores/(1-scores)).mean(axis=1)), scores


def run(output_dir):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    if (output/"summary.json").exists():
        print("Budgeted incident localizer already complete; no recomputation")
        return
    bundles = [joblib.load(MEMBERS/f"outer_{index}/bundle.joblib") for index in range(3)]
    names = json.loads((BASE/"feature_names.json").read_text())
    events = json.loads((DATA/"events.json").read_text())
    calibration = state_arrays("calibration", names, events)
    calibration_point, calibration_members = committee_score(calibration, bundles)
    calibration_score, calibration_rows, calibration_groups = budgeted_noise_rank_score(
        calibration, calibration_point, names)
    calibration_label = calibration["labels"][calibration_rows] > 0
    choices = {}
    for top_k in range(1, MAX_TARGETS+1):
        decision = top_k_sensor_decision(calibration_rows, calibration_groups, top_k)
        choices[str(top_k)] = metrics(calibration_label, decision)
    selected_k = max(range(1, MAX_TARGETS+1), key=lambda top_k:
        (choices[str(top_k)]["f1"], choices[str(top_k)]["precision"], -top_k))
    selection = {"top_k": selected_k, "maximum_pressure_targets": MAX_TARGETS,
        "fixed_equal_rank_weights": {"committee": 0.5, "physical_difference": 0.5},
        "calibration_choices": choices, "frozen_before_validation_prediction": True,
        "protocol_sha256": sha(PROTOCOL)}
    write_json(output/"selection_frozen.json", selection)
    np.savez_compressed(output/"predictions_calibration.npz", score=calibration_score,
        label=calibration_label, row=calibration_rows,
        decision=top_k_sensor_decision(calibration_rows, calibration_groups, selected_k),
        member_point_scores=calibration_members)

    validation = state_arrays("validation", names, events)
    validation_point, validation_members = committee_score(validation, bundles)
    validation_score, validation_rows, validation_groups = budgeted_noise_rank_score(
        validation, validation_point, names)
    validation_label = validation["labels"][validation_rows] > 0
    validation_decision = top_k_sensor_decision(validation_rows, validation_groups, selected_k)
    noise_metrics = metrics(validation_label, validation_decision)
    np.savez_compressed(output/"predictions_validation.npz", score=validation_score,
        label=validation_label, row=validation_rows, decision=validation_decision,
        scenario=validation["scenario"][validation_rows],
        member_point_scores=validation_members)

    drift_summary = json.loads((DRIFT/"summary.json").read_text())
    drift_metrics = drift_summary["validation"]["drift"]["metrics"]
    success = drift_metrics["f1"] >= .8 and noise_metrics["f1"] >= .8
    summary = {"scope": "post-development conditional budgeted incident localisation",
        "selection": selection, "development_validation": {
            "drift": drift_metrics, "noise": noise_metrics},
        "target_0_80_both_achieved": bool(success),
        "reused_development_validation_not_independent": True,
        "external_window_and_family_required": True,
        "calibration_evaluated": True, "validation_evaluated": True,
        "test_evaluated": False, "input_hashes": {
            "drift_summary": sha(DRIFT/"summary.json"),
            "calibration_features": sha(BASE/"full/features_calibration.npz"),
            "validation_features": sha(BASE/"full/features_validation.npz"),
            **{str(MEMBERS/f"outer_{index}/bundle.joblib"):
                sha(MEMBERS/f"outer_{index}/bundle.joblib") for index in range(3)}},
        "prediction_hashes": {"calibration": sha(output/"predictions_calibration.npz"),
            "validation": sha(output/"predictions_validation.npz")}}
    write_json(output/"summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/operational/budgeted_incident_localizer_v1")
    run(parser.parse_args().output_dir)
