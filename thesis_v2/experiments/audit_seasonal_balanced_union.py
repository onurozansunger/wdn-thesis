"""Independent reconstruction audit for the family-balanced union rule."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from evaluate_seasonal_balanced_union import BUDGETS, FAMILIES, decision, points
from wdn.run_expert_redesign import load_arrays, read_json, sha, write_json
from wdn.train_operational_moe import summarise


RUN = Path("runs/operational/seasonal_balanced_union_v1")
SOURCE = Path("runs/operational/seasonal_family_deployment_v2")
OLD = Path("runs/operational/blind_residual_experts_v1")
OUTPUT = Path("thesis_v2/outputs/seasonal_balanced_union_results.json")


def run():
    summary, signature = read_json(RUN / "summary.json"), read_json(RUN / "signature.json")
    saved_selection = read_json(RUN / "selection_frozen.json")
    checks = {}
    checks["run_completed"] = summary["status"] == "completed"
    checks["source_deployment_audit_passed"] = (
        read_json(SOURCE / "independent_audit.json")["all_checks_pass"]
        and signature["source_audit_sha256"] == sha(SOURCE / "independent_audit.json"))
    checks["source_and_input_hashes_match"] = (
        signature["protocol_sha256"] == sha(Path("thesis_v2/SEASONAL_BALANCED_UNION_PROTOCOL.md"))
        and signature["source_selection_sha256"] == sha(SOURCE / "selection_frozen.json")
        and signature["source_calibration_predictions_sha256"] == sha(
            SOURCE / "predictions_calibration.npz")
        and signature["old_calibration_predictions_sha256"] == sha(
            OLD / "predictions_calibration.npz")
        and signature["source_sha256"] == sha(Path(
            "thesis_v2/experiments/evaluate_seasonal_balanced_union.py")))
    checks["post_development_and_test_limits_explicit"] = (
        summary["claim_limits"]["post_validation_development_iteration"]
        and summary["claim_limits"]["validation_is_reused_and_not_independent"]
        and not summary["test_evaluated"] and not signature["test_evaluated"])
    checks["selection_written_before_output_validation_predictions"] = (
        (RUN / "selection_frozen.json").stat().st_mtime_ns
        <= (RUN / "predictions_validation.npz").stat().st_mtime_ns)

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
    objective, selected, calibration = max(candidates, key=lambda row: row[0])
    expected_selection = {"budgets": selected,
        "thresholds": {key: fitted[key][selected[f"{key}_budget"]]["threshold"]
                       for key in ("old", "drift", "noise")},
        "objective": list(objective), "calibration": calibration,
        "feasible_candidates": len(candidates),
        "selected_before_loading_validation_predictions": True,
        "post_validation_development_iteration": True, "test_evaluated": False}
    checks["all_calibration_candidates_and_selection_reproduced_exact"] = (
        expected_selection == saved_selection)
    saved_cal = load_arrays(RUN / "predictions_calibration.npz")
    checks["calibration_decision_reproduced_exact"] = np.array_equal(
        decision(cal_scores, selected, fitted), saved_cal["decision"])

    val = load_arrays(SOURCE / "full/features_validation.npz")
    val_scores = {"specialists": load_arrays(
        SOURCE / "predictions_validation.npz")["specialists"],
        "old": load_arrays(OLD / "predictions_validation.npz")["mixture"]}
    predicted = decision(val_scores, selected, fitted)
    report = summarise(predicted.astype(float), val["labels"], val["families"], .5)
    saved_val = load_arrays(RUN / "predictions_validation.npz")
    checks["validation_decision_and_metrics_reproduced_exact"] = (
        np.array_equal(predicted, saved_val["decision"])
        and report == summary["development_validation"])
    family_f1 = {name: report["per_family"][name]["f1"] for name in FAMILIES}
    checks["all_five_attack_family_f1_values_at_least_0_70"] = (
        min(family_f1.values()) >= .70
        and family_f1 == summary["family_f1"]
        and summary["all_attack_families_at_least_0_70"])
    checks["saved_output_hashes_match"] = all(
        summary["hashes"][key] == sha(path) for key, path in {
            "selection": RUN / "selection_frozen.json",
            "calibration_predictions": RUN / "predictions_calibration.npz",
            "validation_predictions": RUN / "predictions_validation.npz"}.items())
    checks["validation_contains_only_declared_three_scenarios"] = set(
        saved_val["scenario"]) == {6, 10, 11}

    result = {"all_checks_pass": bool(all(checks.values())), **checks,
        "summary": summary, "claim_limits": summary["claim_limits"]}
    write_json(RUN / "independent_audit.json", result)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True); write_json(OUTPUT, result)
    print(json.dumps({key: value for key, value in result.items()
                      if isinstance(value, bool)}, indent=2))


if __name__ == "__main__":
    run()
