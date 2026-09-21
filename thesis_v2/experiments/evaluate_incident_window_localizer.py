"""Fit on TRAIN, calibrate, then evaluate the frozen incident-window localizer once."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score

from wdn.models.family_specific import FamilySpecificExpert, MonotoneFamilyStacker
from wdn.models.incident_window import aggregate_incident_scores
from wdn.run_expert_redesign import load_arrays, sha, write_json
from wdn.screen_family_specific import _with_state


BASE = Path("runs/operational/mechanism_redesign_v1")
OOF = Path("runs/operational/family_specific_experts_v1/oof_predictions.npz")
TRAIN_SCREEN = Path("runs/operational/incident_window_localizer_v1/summary.json")
DATA = Path("data/thesis_v2/operational_modena_seed811")
PROTOCOL = Path("thesis_v2/INCIDENT_WINDOW_VALIDATION_PROTOCOL.md")
FAMILIES = {"drift": {"id": 3, "mode": "top2"},
            "noise": {"id": 4, "mode": "logit"}}


def best_threshold(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels, bool)
    order = np.argsort(-scores, kind="stable")
    score, target = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(score[:-1] != score[1:]), len(score)-1]
    count, tp = ends+1, np.cumsum(target)[ends]
    f1 = 2*tp/np.maximum(count+target.sum(), 1)
    best = int(np.argmax(f1))
    return float(np.nextafter(score[ends[best]], -np.inf)), float(f1[best])


def metrics(labels, decisions):
    labels, decisions = np.asarray(labels, bool), np.asarray(decisions, bool)
    tp = int(np.sum(labels & decisions)); fp = int(np.sum(~labels & decisions))
    fn = int(np.sum(labels & ~decisions))
    return {"f1": float(2*tp/max(2*tp+fp+fn, 1)),
        "precision": float(tp/max(tp+fp, 1)), "recall": float(tp/max(tp+fn, 1)),
        "tp": tp, "fp": fp, "fn": fn}


def state_arrays(split, names, events):
    arrays = load_arrays(BASE/f"full/features_{split}.npz")
    return _with_state(arrays, names, events)[0]


def score_split(arrays, drift_expert, noise_expert, drift_stacker):
    drift_heads = drift_expert.predict_heads(arrays["X"])
    noise_heads = noise_expert.predict_heads(arrays["X"])
    point_scores = {"drift": drift_stacker.predict(drift_heads),
                    "noise": noise_heads[:, 1]}
    output = {}
    for family, recipe in FAMILIES.items():
        selected = arrays["families"] == recipe["id"]
        aggregate = aggregate_incident_scores(point_scores[family], arrays["event"],
            arrays["node"], arrays["timestep"], selected, recipe["mode"])
        output[family] = {"score": aggregate[selected],
            "label": arrays["labels"][selected] > 0,
            "scenario": arrays["scenario"][selected],
            "row": np.flatnonzero(selected)}
    return output


def evaluate(scores, thresholds):
    report = {}
    for family, arrays in scores.items():
        decision = arrays["score"] > thresholds[family]
        scenarios = {}
        for sid in np.unique(arrays["scenario"]):
            local = arrays["scenario"] == sid
            scenarios[str(int(sid))] = {"metrics": metrics(arrays["label"][local], decision[local]),
                "ap": float(average_precision_score(arrays["label"][local], arrays["score"][local]))}
        report[family] = {"metrics": metrics(arrays["label"], decision),
            "ap": float(average_precision_score(arrays["label"], arrays["score"])),
            "by_scenario": scenarios}
    return report


def run(output_dir):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    if (output/"summary.json").exists():
        print("Incident-window development validation already complete; no refit")
        return
    train_screen = json.loads(TRAIN_SCREEN.read_text())
    if not train_screen["passed"] or any(train_screen[key] for key in
            ("calibration_evaluated", "validation_evaluated", "test_evaluated")):
        raise ValueError("TRAIN incident-window gate or split state changed")
    names = json.loads((BASE/"feature_names.json").read_text())
    events = json.loads((DATA/"events.json").read_text())
    oof = load_arrays(OOF)
    drift_stacker = MonotoneFamilyStacker("drift").fit(oof["head_scores"][:, :2], oof)
    train, full_names = _with_state(load_arrays(BASE/"full/features_train.npz"), names, events)
    drift_expert = FamilySpecificExpert(full_names, "drift").fit(train)
    noise_expert = FamilySpecificExpert(full_names, "noise").fit(train)
    bundle = {"drift_expert": drift_expert, "noise_expert": noise_expert,
              "drift_stacker": drift_stacker, "feature_names": full_names}
    joblib.dump(bundle, output/"model.joblib")

    calibration = state_arrays("calibration", names, events)
    calibration_scores = score_split(calibration, drift_expert, noise_expert, drift_stacker)
    thresholds, calibration_report = {}, {}
    for family, arrays in calibration_scores.items():
        thresholds[family], hindsight_f1 = best_threshold(arrays["score"], arrays["label"])
        calibration_report[family] = {"threshold": thresholds[family],
            "metrics": metrics(arrays["label"], arrays["score"] > thresholds[family]),
            "ap": float(average_precision_score(arrays["label"], arrays["score"])),
            "threshold_search_f1": hindsight_f1}
    selection = {"thresholds": thresholds, "calibration": calibration_report,
        "frozen_before_validation_read": True, "protocol_sha256": sha(PROTOCOL)}
    write_json(output/"selection_frozen.json", selection)
    np.savez_compressed(output/"predictions_calibration.npz", **{
        f"{family}_{key}": value for family, arrays in calibration_scores.items()
        for key, value in arrays.items()})

    validation = state_arrays("validation", names, events)
    validation_scores = score_split(validation, drift_expert, noise_expert, drift_stacker)
    validation_report = evaluate(validation_scores, thresholds)
    np.savez_compressed(output/"predictions_validation.npz", **{
        f"{family}_{key}": value for family, arrays in validation_scores.items()
        for key, value in arrays.items()})
    success = all(validation_report[family]["metrics"]["f1"] >= .8 for family in FAMILIES)
    summary = {"scope": "conditional offline incident localisation; reused development validation",
        "selection": selection, "validation": validation_report,
        "target_0_80_both_achieved": bool(success),
        "conditional_on_external_window_and_family": True,
        "calibration_evaluated": True, "validation_evaluated": True,
        "test_evaluated": False, "input_hashes": {"train_screen": sha(TRAIN_SCREEN),
            "train_oof": sha(OOF), "feature_names": sha(BASE/"feature_names.json"),
            "train": sha(BASE/"full/features_train.npz"),
            "calibration": sha(BASE/"full/features_calibration.npz"),
            "validation": sha(BASE/"full/features_validation.npz")}}
    write_json(output/"summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/operational/incident_window_validation_v1")
    run(parser.parse_args().output_dir)
