"""Calibration-only selection of a post-development family-balanced OR rule."""
from __future__ import annotations

import argparse
import fcntl
import itertools
import json
from pathlib import Path

import numpy as np

from fit_evaluate_seasonal_family_experts import family_threshold
from wdn.operational_calibration import calibrate_threshold
from wdn.run_expert_redesign import load_arrays, read_json, sha, write_json
from wdn.train_operational_moe import summarise


SOURCE = Path("runs/operational/seasonal_family_deployment_v2")
OLD = Path("runs/operational/blind_residual_experts_v1")
PROTOCOL = Path("thesis_v2/SEASONAL_BALANCED_UNION_PROTOCOL.md")
BUDGETS = tuple(i / 10000 for i in range(5, 51, 5))
FAMILIES = ("random", "replay", "stealthy", "noise", "targeted")


def points(scores, arrays):
    result = {"old": {}, "drift": {}, "noise": {}}
    for budget in BUDGETS:
        result["old"][budget] = calibrate_threshold(scores["old"], arrays["labels"],
            arrays["families"], objective="worst_f1", max_fpr=budget,
            min_replay_f1=.5)
        result["drift"][budget] = family_threshold(scores["specialists"][:, 0], arrays, 3, budget)
        result["noise"][budget] = family_threshold(scores["specialists"][:, 1], arrays, 4, budget)
    return result


def decision(scores, selected, fitted):
    return ((scores["old"] > fitted["old"][selected["old_budget"]]["threshold"])
        | (scores["specialists"][:, 0]
           > fitted["drift"][selected["drift_budget"]]["threshold"])
        | (scores["specialists"][:, 1]
           > fitted["noise"][selected["noise_budget"]]["threshold"]))


def run(output_dir):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    lock = (output / "campaign.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Balanced-union evaluation is already active") from error
    signature = {"protocol_sha256": sha(PROTOCOL),
        "source_audit_sha256": sha(SOURCE / "independent_audit.json"),
        "source_selection_sha256": sha(SOURCE / "selection_frozen.json"),
        "source_calibration_predictions_sha256": sha(SOURCE / "predictions_calibration.npz"),
        "old_calibration_predictions_sha256": sha(OLD / "predictions_calibration.npz"),
        "source_sha256": sha(Path(__file__)), "budgets": list(BUDGETS),
        "post_validation_development_iteration": True, "test_evaluated": False}
    if (output / "signature.json").exists() and read_json(output / "signature.json") != signature:
        raise ValueError("Balanced-union source/input signature changed")
    write_json(output / "signature.json", signature)
    if (output / "summary.json").exists():
        print("Balanced-union evaluation already complete", flush=True); return
    if not read_json(SOURCE / "independent_audit.json")["all_checks_pass"]:
        raise RuntimeError("Source deployment audit did not pass")

    cal = load_arrays(SOURCE / "full/features_calibration.npz")
    cal_scores = {"specialists": load_arrays(
        SOURCE / "predictions_calibration.npz")["specialists"],
        "old": load_arrays(OLD / "predictions_calibration.npz")["mixture"]}
    fitted = points(cal_scores, cal)
    candidates = []
    for old_budget, drift_budget, noise_budget in itertools.product(BUDGETS, repeat=3):
        selected = {"old_budget": old_budget, "drift_budget": drift_budget,
                    "noise_budget": noise_budget}
        predicted = decision(cal_scores, selected, fitted)
        report = summarise(predicted.astype(float), cal["labels"], cal["families"], .5)
        rows = report["per_family"]
        if (report["overall"]["fpr"] > .005 or rows["clean"]["fpr"] > .005
                or rows["replay"]["f1"] < .5):
            continue
        drift, noise = rows["stealthy"]["f1"], rows["noise"]["f1"]
        objective = (min(drift, noise), (drift + noise) / 2,
                     rows["replay"]["f1"], report["overall"]["f1"],
                     -report["overall"]["fpr"])
        candidates.append((objective, selected, report))
    if not candidates:
        raise RuntimeError("No feasible balanced-union calibration point")
    objective, selected, calibration = max(candidates, key=lambda row: row[0])
    selection = {"budgets": selected,
        "thresholds": {key: fitted[key][selected[f"{key}_budget"]]["threshold"]
                       for key in ("old", "drift", "noise")},
        "objective": list(objective), "calibration": calibration,
        "feasible_candidates": len(candidates),
        "selected_before_loading_validation_predictions": True,
        "post_validation_development_iteration": True, "test_evaluated": False}
    write_json(output / "selection_frozen.json", selection)
    np.savez_compressed(output / "predictions_calibration.npz",
        decision=decision(cal_scores, selected, fitted), labels=cal["labels"],
        families=cal["families"], scenario=cal["scenario"])

    # Selection is now immutable. This is the first validation-prediction load in this script.
    val = load_arrays(SOURCE / "full/features_validation.npz")
    val_scores = {"specialists": load_arrays(
        SOURCE / "predictions_validation.npz")["specialists"],
        "old": load_arrays(OLD / "predictions_validation.npz")["mixture"]}
    predicted = decision(val_scores, selected, fitted)
    validation = summarise(predicted.astype(float), val["labels"], val["families"], .5)
    np.savez_compressed(output / "predictions_validation.npz", decision=predicted,
        labels=val["labels"], families=val["families"], scenario=val["scenario"],
        timestep=val["timestep"], node=val["node"])
    family_f1 = {name: validation["per_family"][name]["f1"] for name in FAMILIES}
    result = {"status": "completed", "selection": selection,
        "development_validation": validation, "family_f1": family_f1,
        "all_attack_families_at_least_0_70": min(family_f1.values()) >= .70,
        "drift_and_noise_at_least_0_70": min(family_f1["stealthy"], family_f1["noise"]) >= .70,
        "overall_f1": validation["overall"]["f1"],
        "claim_limits": {"post_validation_development_iteration": True,
            "validation_is_reused_and_not_independent": True,
            "binary_rule_has_no_ranking_auprc_claim": True, "test_evaluated": False},
        "hashes": {"selection": sha(output / "selection_frozen.json"),
            "calibration_predictions": sha(output / "predictions_calibration.npz"),
            "validation_predictions": sha(output / "predictions_validation.npz")},
        "test_evaluated": False}
    write_json(output / "summary.json", result)
    print(json.dumps({"budgets": selected, "overall_f1": result["overall_f1"],
        "family_f1": family_f1,
        "all_attack_families_at_least_0_70": result["all_attack_families_at_least_0_70"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/operational/seasonal_balanced_union_v1")
    run(parser.parse_args().output_dir)
