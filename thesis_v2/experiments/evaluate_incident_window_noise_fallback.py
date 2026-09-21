"""Bounded calibration selection between two pre-identified noise aggregations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score

from wdn.models.family_specific import MonotoneFamilyStacker
from wdn.models.incident_window import aggregate_incident_scores
from wdn.run_expert_redesign import load_arrays, sha, write_json
from evaluate_incident_window_localizer import (
    BASE, DATA, OOF, best_threshold, metrics, state_arrays)


PRIMARY = Path("runs/operational/incident_window_validation_v1")
PROTOCOL = Path("thesis_v2/INCIDENT_WINDOW_NOISE_FALLBACK_PROTOCOL.md")
MODES = ("logit", "mean")


def candidate(arrays, expert, stacker, mode):
    point = stacker.predict(expert.predict_heads(arrays["X"]))
    selected = arrays["families"] == 4
    score = aggregate_incident_scores(point, arrays["event"], arrays["node"],
        arrays["timestep"], selected, mode)[selected]
    return {"score": score, "label": arrays["labels"][selected] > 0,
            "scenario": arrays["scenario"][selected]}


def run(output_dir):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    if (output/"summary.json").exists():
        print("Noise incident-window fallback already complete; no recomputation")
        return
    primary = json.loads((PRIMARY/"summary.json").read_text())
    if (primary["target_0_80_both_achieved"] or not primary["validation_evaluated"]
            or primary["test_evaluated"]):
        raise ValueError("Primary development result does not permit the frozen fallback")
    bundle = joblib.load(PRIMARY/"model.joblib")
    oof = load_arrays(OOF)
    stacker = MonotoneFamilyStacker("noise").fit(oof["head_scores"][:, 2:4], oof)
    names = json.loads((BASE/"feature_names.json").read_text())
    events = json.loads((DATA/"events.json").read_text())
    calibration = state_arrays("calibration", names, events)
    calibration_rows = {}
    for mode in MODES:
        arrays = candidate(calibration, bundle["noise_expert"], stacker, mode)
        threshold, search_f1 = best_threshold(arrays["score"], arrays["label"])
        calibration_rows[mode] = {"threshold": threshold, "f1": search_f1,
            "ap": float(average_precision_score(arrays["label"], arrays["score"]))}
    selected_mode = max(MODES, key=lambda mode: (calibration_rows[mode]["f1"],
                                                  calibration_rows[mode]["ap"], -MODES.index(mode)))
    selection = {"mode": selected_mode, "calibration": calibration_rows,
        "threshold": calibration_rows[selected_mode]["threshold"],
        "frozen_before_fallback_validation_prediction": True,
        "protocol_sha256": sha(PROTOCOL)}
    write_json(output/"selection_frozen.json", selection)

    validation = state_arrays("validation", names, events)
    arrays = candidate(validation, bundle["noise_expert"], stacker, selected_mode)
    decision = arrays["score"] > selection["threshold"]
    by_scenario = {}
    for sid in np.unique(arrays["scenario"]):
        local = arrays["scenario"] == sid
        by_scenario[str(int(sid))] = {"metrics": metrics(arrays["label"][local], decision[local]),
            "ap": float(average_precision_score(arrays["label"][local], arrays["score"][local]))}
    validation_report = {"metrics": metrics(arrays["label"], decision),
        "ap": float(average_precision_score(arrays["label"], arrays["score"])),
        "by_scenario": by_scenario}
    np.savez_compressed(output/"predictions_validation.npz", **arrays, decision=decision)
    summary = {"scope": "second-pass development; conditional incident-window noise localisation",
        "selection": selection, "validation": validation_report,
        "noise_0_80_achieved": bool(validation_report["metrics"]["f1"] >= .8),
        "calibration_evaluated": True, "validation_evaluated": True,
        "test_evaluated": False, "primary_summary_sha256": sha(PRIMARY/"summary.json"),
        "validation_predictions_sha256": sha(output/"predictions_validation.npz")}
    write_json(output/"summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/operational/incident_window_noise_fallback_v1")
    run(parser.parse_args().output_dir)
