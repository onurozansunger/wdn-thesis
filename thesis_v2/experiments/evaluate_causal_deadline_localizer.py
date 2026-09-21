"""Evaluate the frozen past-only event checkpoint localizer."""
from __future__ import annotations

import argparse, json
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit

from evaluate_incident_window_localizer import BASE, DATA, state_arrays
from wdn.models.causal_deadline import causal_deadline_groups, sensor_event_metrics
from wdn.run_expert_redesign import sha, write_json

MODEL = Path("runs/operational/incident_window_validation_v1/model.joblib")
MEMBERS = Path("runs/operational/family_specific_experts_v1")
NOISE_SELECTION = Path("runs/operational/budgeted_incident_localizer_v1/selection_frozen.json")
PROTOCOL = Path("thesis_v2/CAUSAL_DEADLINE_LOCALIZER_PROTOCOL.md")
DEADLINE = 10
RECIPES = {"drift": {"mode": "top2", "weight": .5, "top_k": 12},
           "noise": {"mode": "mean", "weight": .5, "top_k": 13}}


def noise_committee(arrays, bundles):
    members = []
    for bundle in bundles:
        component = bundle["noise"]
        members.append(component["stacker"].predict(
            component["expert"].predict_heads(arrays["X"])))
    scores = np.clip(np.column_stack(members), 1e-6, 1-1e-6)
    return expit(np.log(scores/(1-scores)).mean(axis=1))


def evaluate_split(arrays, model, bundles, names):
    drift_heads = model["drift_expert"].predict_heads(arrays["X"])
    point = {"drift": model["drift_stacker"].predict(drift_heads),
             "noise": noise_committee(arrays, bundles)}
    report, serial = {}, {}
    for family in ("drift", "noise"):
        recipe = RECIPES[family]
        groups = causal_deadline_groups(arrays, point[family], names, family, DEADLINE,
            recipe["mode"], recipe["weight"])
        report[family] = sensor_event_metrics(groups, recipe["top_k"])
        report[family]["incidents"] = {str(event): {
            "incident_steps": group["incident_steps"],
            "decision_steps": group["available_steps"],
            "decision_at_closure": group["available_steps"] == group["incident_steps"]}
            for event, group in groups.items()}
        serial[family] = groups
    return report, serial


def run(output_dir):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    if (output/"summary.json").exists():
        summary = json.loads((output/"summary.json").read_text())
        write_json(Path("thesis_v2/outputs/causal_deadline_localizer_results.json"), summary)
        print("Causal deadline evaluation already complete; no recomputation")
        return
    prior = json.loads(NOISE_SELECTION.read_text())
    if prior["top_k"] != 13 or prior["fixed_equal_rank_weights"] != {
            "committee": .5, "physical_difference": .5}:
        raise ValueError("Previously frozen noise recipe changed")
    selection = {"deadline_steps": DEADLINE, "recipes": RECIPES,
        "deadline_source": "scenario-OOF TRAIN-only causal screen",
        "noise_recipe_source": str(NOISE_SELECTION),
        "frozen_before_current_validation_prediction": True,
        "protocol_sha256": sha(PROTOCOL)}
    write_json(output/"selection_frozen.json", selection)

    names = json.loads((BASE/"feature_names.json").read_text())
    events = json.loads((DATA/"events.json").read_text())
    model = joblib.load(MODEL)
    bundles = [joblib.load(MEMBERS/f"outer_{index}/bundle.joblib") for index in range(3)]
    calibration_report, _ = evaluate_split(state_arrays("calibration", names, events),
        model, bundles, names)
    validation_report, groups = evaluate_split(state_arrays("validation", names, events),
        model, bundles, names)
    # Compact, independently replayable checkpoint decisions.
    flat = {}
    for family, family_groups in groups.items():
        for event, group in family_groups.items():
            sensors = group["sensors"]
            chosen = sorted([sensor for sensor in sensors if sensor["available"]],
                key=lambda sensor: (sensor["fused_rank"], -sensor["node"]), reverse=True)
            selected = {sensor["node"] for sensor in chosen[:RECIPES[family]["top_k"]]}
            prefix = f"{family}_{event}"
            flat[f"{prefix}_node"] = np.asarray([sensor["node"] for sensor in sensors])
            flat[f"{prefix}_score"] = np.asarray([sensor["fused_rank"] for sensor in sensors])
            flat[f"{prefix}_label"] = np.asarray([sensor["target"] for sensor in sensors])
            flat[f"{prefix}_decision"] = np.asarray([sensor["node"] in selected for sensor in sensors])
    np.savez_compressed(output/"predictions_validation.npz", **flat)

    strict = {"scope": "unchanged causal row/hour development validation",
        "overall_f1": .817625458996328, "replay_f1": .8656716417910447,
        "drift_f1": .45901639344262296, "noise_f1": .40540540540540543}
    summary = {"scope": "conditional causal tenth-checkpoint/closure sensor localisation",
        "selection": selection, "calibration": calibration_report,
        "reused_development_validation": validation_report,
        "target_0_70_both_achieved_on_sensor_event_metric": bool(all(
            validation_report[family]["f1"] >= .7 for family in RECIPES)),
        "unchanged_strict_pointwise_online": strict,
        "not_early_continuous_localisation": True,
        "reused_development_validation_not_independent": True,
        "external_alarm_onset_and_family_required": True,
        "retrospective_localizer_unchanged": True,
        "pressure_missing_probability": .5, "flow_missing_probability": .5,
        "calibration_evaluated": True, "validation_evaluated": True,
        "test_evaluated": False,
        "input_hashes": {"model": sha(MODEL), "noise_selection": sha(NOISE_SELECTION),
            "calibration": sha(BASE/"full/features_calibration.npz"),
            "validation": sha(BASE/"full/features_validation.npz"),
            **{str(MEMBERS/f"outer_{i}/bundle.joblib"):
                sha(MEMBERS/f"outer_{i}/bundle.joblib") for i in range(3)}},
        "prediction_sha256": sha(output/"predictions_validation.npz")}
    write_json(output/"summary.json", summary)
    write_json(Path("thesis_v2/outputs/causal_deadline_localizer_results.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/operational/causal_deadline_localizer_v1")
    run(parser.parse_args().output_dir)
