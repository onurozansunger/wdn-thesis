"""Independent reconstruction audit for the seasonal deployment candidate."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from fit_evaluate_seasonal_family_experts import (
    PENALTIES, combined_score, family_report, family_threshold, logit,
    specialist_scores)
from wdn.run_expert_redesign import (
    load_arrays, operating_point, read_json, score_report, sha, write_json)


RUN = Path("runs/operational/seasonal_family_deployment_v2")
SOURCE_RUN = Path("runs/operational/seasonal_family_deployment_v1")
SCREEN = Path("runs/operational/seasonal_family_experts_v3")
BASE = Path("runs/operational/expanded_train_weak_experts_v1")
OLD = Path("runs/operational/blind_residual_experts_v1")
SPLITS = Path("runs/operational/blind_reference_probe_rank16/splits.json")
OUTPUT = Path("thesis_v2/outputs/seasonal_family_deployment_results.json")


def run():
    summary = read_json(RUN / "summary.json")
    signature = read_json(RUN / "signature.json")
    selection = read_json(RUN / "selection_frozen.json")
    reuse = read_json(RUN / "artifact_reuse.json")
    splits = read_json(SPLITS)
    checks = {}
    checks["run_completed"] = summary["status"] == "completed"
    checks["train_screen_independent_audit_passed"] = (
        read_json(SCREEN / "independent_audit.json")["all_checks_pass"]
        and signature["screen_audit_sha256"] == sha(SCREEN / "independent_audit.json"))
    checks["source_hashes_match"] = all(
        sha(Path(path)) == digest for path, digest in signature["source_sha256"].items())
    checks["input_hashes_match"] = (
        signature["protocol_sha256"] == sha(Path("thesis_v2/SEASONAL_FAMILY_DEPLOYMENT_PROTOCOL.md"))
        and signature["screen_summary_sha256"] == sha(SCREEN / "summary.json")
        and signature["base_signature_sha256"] == sha(BASE / "signature.json")
        and signature["splits_sha256"] == sha(SPLITS)
        and signature["old_calibration_predictions_sha256"] == sha(
            OLD / "predictions_calibration.npz"))
    checks["test_never_evaluated"] = (
        not summary["test_evaluated"] and not signature["test_evaluated"]
        and not summary["claim_limits"]["test_evaluated"])

    expected_reuse = all(sha(RUN / relative) == digest
                         for relative, digest in reuse["artifacts"].items())
    source_signature = read_json(SOURCE_RUN / "signature.json")
    changed = {path for path in source_signature["source_sha256"]
               if source_signature["source_sha256"][path] != signature["source_sha256"][path]}
    checks["precalibration_artifact_reuse_is_explicit_and_hashed"] = (
        reuse["copied_before_calibration_selection"] and expected_reuse
        and changed == {str(Path(
            "thesis_v2/experiments/fit_evaluate_seasonal_family_experts.py").resolve())})
    checks["selection_frozen_before_validation_artifacts"] = (
        (RUN / "selection_frozen.json").stat().st_mtime_ns
        <= (RUN / "full/features_validation.npz").stat().st_mtime_ns)

    train = load_arrays(RUN / "full/features_train.npz")
    cal = load_arrays(RUN / "full/features_calibration.npz")
    val = load_arrays(RUN / "full/features_validation.npz")
    original_train_global = {811000 + sid for sid in splits["train"]}
    protected_global = {811000 + sid for key in ("calibration", "validation", "test")
                        for sid in splits[key]}
    checks["all_62_train_scenarios_and_no_protected_original"] = (
        len(set(train["scenario"])) == 62
        and original_train_global <= set(train["scenario"])
        and set(train["scenario"]).isdisjoint(protected_global))
    checks["calibration_validation_scope_exact_and_test_absent"] = (
        set(cal["scenario"]) == set(splits["calibration"])
        and set(val["scenario"]) == set(splits["validation"])
        and set(cal["scenario"]).isdisjoint(splits["test"])
        and set(val["scenario"]).isdisjoint(splits["test"]))
    data_audit = read_json(BASE / "data_audit.json")
    checks["pressure_and_flow_missing_probabilities_remain_half"] = all(
        row["pressure_missing_probability"] == row["flow_missing_probability"] == .5
        for row in data_audit["config_audit"].values())

    bundle = joblib.load(RUN / "full/bundle.joblib")
    cal_specialists = specialist_scores(bundle, cal["X"],
        load_arrays(RUN / "full/seasonal_calibration.npz")["X"])
    val_specialists = specialist_scores(bundle, val["X"],
        load_arrays(RUN / "full/seasonal_validation.npz")["X"])
    saved_cal = load_arrays(RUN / "predictions_calibration.npz")
    saved_val = load_arrays(RUN / "predictions_validation.npz")
    checks["saved_bundle_reloads_specialist_predictions_exact"] = (
        np.array_equal(cal_specialists, saved_cal["specialists"])
        and np.array_equal(val_specialists, saved_val["specialists"]))

    points = {"drift": family_threshold(cal_specialists[:, 0], cal, 3),
              "noise": family_threshold(cal_specialists[:, 1], cal, 4)}
    old_cal = load_arrays(OLD / "predictions_calibration.npz")
    old_val = load_arrays(OLD / "predictions_validation.npz")
    old_point = operating_point(old_cal["mixture"], cal)
    centers = {"old": logit(old_point["threshold"]),
        "drift": logit(points["drift"]["threshold"]),
        "noise": logit(points["noise"]["threshold"])}
    names = ["old_mixture"] + [f"seasonal_max_{penalty:g}" for penalty in PENALTIES]
    candidates = {}
    for name in names:
        score = combined_score(name, old_cal["mixture"], cal_specialists, centers)
        try:
            point = operating_point(score, cal)
        except ValueError:
            continue
        candidates[name] = {"point": point,
            "specialist_penalty": None if name == "old_mixture" else float(name.rsplit("_", 1)[1])}
    winner = max(candidates, key=lambda name: (
        candidates[name]["point"]["calibration"]["worst_family_f1"],
        candidates[name]["point"]["calibration"]["macro_f1"],
        candidates[name]["point"]["calibration"]["overall_f1"],
        -candidates[name]["point"]["calibration"]["fpr"], -names.index(name)))
    checks["calibration_thresholds_candidates_and_winner_reproduced_exact"] = (
        points == selection["family_thresholds"]
        and old_point == selection["old_mixture_point"]
        and candidates == selection["candidates"]
        and winner == selection["candidate"])

    selected_val = combined_score(winner, old_val["mixture"], val_specialists, centers)
    checks["selected_validation_detector_score_reproduced_exact"] = np.array_equal(
        selected_val, saved_val["detector"])
    specialist_validation = {
        "drift": family_report(val_specialists[:, 0], val, 3, points["drift"]),
        "noise": family_report(val_specialists[:, 1], val, 4, points["noise"])}
    specialist_calibration = {
        "drift": family_report(cal_specialists[:, 0], cal, 3, points["drift"]),
        "noise": family_report(cal_specialists[:, 1], cal, 4, points["noise"])}
    selected_report = score_report(selected_val, val, candidates[winner]["point"])
    old_report = score_report(old_val["mixture"], val, old_point)
    checks["saved_calibration_and_validation_metrics_reproduced_exact"] = (
        specialist_calibration == summary["specialists"]["calibration"]
        and specialist_validation == summary["specialists"]["development_validation"]
        and selected_report == summary["whole_detector"]["selected"]
        and old_report == summary["whole_detector"]["old_mixture_recalibrated"])
    checks["saved_output_hashes_match"] = all(
        summary["hashes"][key] == sha(path) for key, path in {
            "bundle": RUN / "full/bundle.joblib",
            "selection": RUN / "selection_frozen.json",
            "calibration_predictions": RUN / "predictions_calibration.npz",
            "validation_predictions": RUN / "predictions_validation.npz"}.items())

    result = {"all_checks_pass": bool(all(checks.values())), **checks,
        "summary": summary, "claim_limits": summary["claim_limits"]}
    write_json(RUN / "independent_audit.json", result)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT, result)
    print(json.dumps({key: value for key, value in result.items()
                      if isinstance(value, bool)}, indent=2))


if __name__ == "__main__":
    run()
