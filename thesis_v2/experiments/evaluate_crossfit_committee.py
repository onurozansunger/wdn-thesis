"""Evaluate an equal log-odds committee of the three saved outer family experts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score

from evaluate_incident_window_localizer import BASE, DATA, best_threshold, metrics, state_arrays
from wdn.models.incident_window import aggregate_incident_scores
from wdn.run_expert_redesign import sha, write_json


MEMBERS = Path("runs/operational/family_specific_experts_v1")
PROTOCOL = Path("thesis_v2/CROSSFIT_COMMITTEE_PROTOCOL.md")
FAMILY_ID = {"drift": 3, "noise": 4}


def committee_scores(arrays, bundles, family):
    member_scores = []
    for bundle in bundles:
        component = bundle[family]
        heads = component["expert"].predict_heads(arrays["X"])
        member_scores.append(component["stacker"].predict(heads))
    member_scores = np.column_stack(member_scores)
    logits = np.log(np.clip(member_scores, 1e-6, 1-1e-6)
                    /np.clip(1-member_scores, 1e-6, 1-1e-6))
    return expit(logits.mean(axis=1)), member_scores


def incident_candidate(arrays, point_score, family, mode):
    selected = arrays["families"] == FAMILY_ID[family]
    score = aggregate_incident_scores(point_score, arrays["event"], arrays["node"],
        arrays["timestep"], selected, mode)[selected]
    return {"score": score, "label": arrays["labels"][selected] > 0,
        "scenario": arrays["scenario"][selected], "row": np.flatnonzero(selected)}


def report(arrays, threshold):
    decision = arrays["score"] > threshold
    by_scenario = {}
    for sid in np.unique(arrays["scenario"]):
        local = arrays["scenario"] == sid
        by_scenario[str(int(sid))] = {"metrics": metrics(arrays["label"][local], decision[local]),
            "ap": float(average_precision_score(arrays["label"][local], arrays["score"][local]))}
    return {"metrics": metrics(arrays["label"], decision),
        "ap": float(average_precision_score(arrays["label"], arrays["score"])),
        "by_scenario": by_scenario}, decision


def run(output_dir):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    if (output/"summary.json").exists():
        print("Cross-fitted committee evaluation already complete; no recomputation")
        return
    bundles = [joblib.load(MEMBERS/f"outer_{index}/bundle.joblib") for index in range(3)]
    names = json.loads((BASE/"feature_names.json").read_text())
    events = json.loads((DATA/"events.json").read_text())
    calibration = state_arrays("calibration", names, events)
    calibration_point, member_calibration = {}, {}
    for family in FAMILY_ID:
        calibration_point[family], member_calibration[family] = committee_scores(
            calibration, bundles, family)
    candidates = {"drift_top2": incident_candidate(calibration,
        calibration_point["drift"], "drift", "top2")}
    for mode in ("logit", "mean"):
        candidates[f"noise_{mode}"] = incident_candidate(calibration,
            calibration_point["noise"], "noise", mode)
    calibration_rows = {}
    for name, arrays in candidates.items():
        threshold, f1 = best_threshold(arrays["score"], arrays["label"])
        calibration_rows[name] = {"threshold": threshold, "f1": f1,
            "ap": float(average_precision_score(arrays["label"], arrays["score"]))}
    noise_name = max(("noise_logit", "noise_mean"), key=lambda name:
        (calibration_rows[name]["f1"], calibration_rows[name]["ap"], name == "noise_logit"))
    selection = {"drift_candidate": "drift_top2", "noise_candidate": noise_name,
        "thresholds": {"drift": calibration_rows["drift_top2"]["threshold"],
                       "noise": calibration_rows[noise_name]["threshold"]},
        "calibration_candidates": calibration_rows,
        "frozen_before_validation_committee_prediction": True,
        "protocol_sha256": sha(PROTOCOL)}
    write_json(output/"selection_frozen.json", selection)

    validation = state_arrays("validation", names, events)
    validation_point, member_validation = {}, {}
    for family in FAMILY_ID:
        validation_point[family], member_validation[family] = committee_scores(
            validation, bundles, family)
    validation_arrays = {
        "drift": incident_candidate(validation, validation_point["drift"], "drift", "top2"),
        "noise": incident_candidate(validation, validation_point["noise"], "noise",
                                    noise_name.removeprefix("noise_"))}
    validation_report, saved = {}, {}
    for family, arrays in validation_arrays.items():
        validation_report[family], decision = report(arrays, selection["thresholds"][family])
        for key, value in arrays.items(): saved[f"{family}_{key}"] = value
        saved[f"{family}_decision"] = decision
    np.savez_compressed(output/"predictions_validation.npz", **saved,
        drift_member_point_scores=member_validation["drift"],
        noise_member_point_scores=member_validation["noise"])
    success = all(validation_report[family]["metrics"]["f1"] >= .8 for family in FAMILY_ID)
    summary = {"scope": "cross-fitted committee; conditional retrospective development validation",
        "selection": selection, "validation": validation_report,
        "target_0_80_both_achieved": bool(success),
        "calibration_evaluated": True, "validation_evaluated": True,
        "test_evaluated": False,
        "member_hashes": {str(MEMBERS/f"outer_{index}/bundle.joblib"):
            sha(MEMBERS/f"outer_{index}/bundle.joblib") for index in range(3)},
        "predictions_sha256": sha(output/"predictions_validation.npz")}
    write_json(output/"summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/operational/crossfit_committee_v1")
    run(parser.parse_args().output_dir)
