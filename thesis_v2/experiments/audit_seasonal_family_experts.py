"""Independent artifact and metric audit for frozen seasonal family experts."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from wdn.expanded_train_data import ExpandedTrainData, scenario_uid
from wdn.run_expert_redesign import load_arrays, read_json, sha, write_json
from wdn.screen_family_specific import _best_f1
from wdn.screen_recovery import expert_diagnostics
from wdn.seasonal_pressure_features import NAMES, endpoint_seasonal_features


RUN = Path("runs/operational/seasonal_family_experts_v3")
BASE = Path("runs/operational/expanded_train_weak_experts_v1")
TUNING = Path("runs/operational/expanded_train_optuna_v1")
PROTOCOL = Path("thesis_v2/SEASONAL_FAMILY_EXPERT_PROTOCOL.md")
ORIGINAL = Path("data/thesis_v2/operational_modena_seed811")
EXPANSIONS = tuple(Path(f"data/thesis_v2/operational_train_expansion_seed{seed}")
                   for seed in (1811, 2811, 3811))
SPLITS = Path("runs/operational/blind_reference_probe_rank16/splits.json")
OUTPUT = Path("thesis_v2/outputs/seasonal_family_experts_results.json")
META_KEYS = ("labels", "families", "event", "scenario", "source", "timestep", "node", "early")


def logit_blend(first, second, first_weight):
    def logit(value):
        value = np.clip(value, 1e-6, 1 - 1e-6)
        return np.log(value / (1 - value))
    linear = first_weight * logit(first) + (1 - first_weight) * logit(second)
    return 1 / (1 + np.exp(-np.clip(linear, -30, 30)))


def oracle(scores, arrays, family_id):
    selected = arrays["families"] == family_id
    return _best_f1(scores[selected], arrays["labels"][selected])


def run():
    summary, signature = read_json(RUN / "summary.json"), read_json(RUN / "signature.json")
    base_summary = read_json(BASE / "summary.json")
    base_audit = read_json(BASE / "independent_audit.json")
    events = read_json(BASE / "events_train_global.json")
    base_names = read_json(BASE / "feature_names.json")
    seasonal_names = read_json(RUN / "seasonal_feature_names.json")
    splits = read_json(SPLITS)
    data = ExpandedTrainData(ORIGINAL, EXPANSIONS, splits)
    checks = {}
    checks["run_completed"] = summary["status"] == "completed"
    checks["base_independent_audit_passed"] = base_audit["all_checks_pass"]
    checks["protocol_hash_matches"] = signature["protocol_sha256"] == sha(PROTOCOL)
    checks["source_hashes_match"] = all(
        sha(Path(path)) == digest for path, digest in signature["source_sha256"].items())
    checks["input_hashes_match"] = (
        signature["base_summary_sha256"] == sha(BASE / "summary.json")
        and signature["base_signature_sha256"] == sha(BASE / "signature.json")
        and signature["tuned_trial_2_summary_sha256"] == sha(TUNING / "trial_0002/summary.json")
        and signature["splits_sha256"] == sha(SPLITS))
    checks["fixed_blends_match_protocol"] = (
        signature["drift_seasonal_weight"] == .90
        and signature["noise_fast_weight"] == .95)
    checks["no_protected_split_evaluated"] = not any(
        summary[key] or signature[key] for key in
        ("calibration_evaluated", "validation_evaluated", "test_evaluated"))
    checks["seasonal_schema_exact"] = seasonal_names == list(NAMES)

    protected = {scenario_uid(811, sid) for split in ("calibration", "validation", "test")
                 for sid in splits[split]}
    metadata_parts, prediction_parts = [], []
    seasonal_exact = scope_exact = reload_exact = True
    for outer, held_ids in enumerate(data.source_folds):
        train = load_arrays(BASE / f"fold_{outer}/features_train.npz")
        held = load_arrays(BASE / f"fold_{outer}/features_held_out.npz")
        scope_exact &= (set(held["scenario"]) == set(held_ids)
                        and set(train["scenario"]).isdisjoint(set(held["scenario"]))
                        and set(held["scenario"]).isdisjoint(protected))
        for split, arrays in (("train", train), ("held_out", held)):
            fresh, fresh_names = endpoint_seasonal_features(arrays, data.scenario)
            saved = load_arrays(RUN / f"fold_{outer}/seasonal_{split}.npz")["X"]
            seasonal_exact &= fresh_names == seasonal_names and np.array_equal(fresh, saved)
        held_seasonal = load_arrays(RUN / f"fold_{outer}/seasonal_held_out.npz")["X"]
        held_X = np.column_stack((held["X"], held_seasonal))
        bundle = joblib.load(RUN / f"fold_{outer}/bundle.joblib")
        fresh_prediction = bundle.predict(held_X)
        saved_prediction = load_arrays(RUN / f"fold_{outer}/predictions.npz")
        reload_exact &= all(np.array_equal(fresh_prediction[key], saved_prediction[key])
                            for key in ("drift", "noise_full", "noise_fast"))
        metadata_parts.append({key: held[key] for key in META_KEYS})
        prediction_parts.append(saved_prediction)
    checks["four_source_folds_disjoint_complete_and_train_only"] = bool(
        scope_exact and len(data.source_folds) == 4
        and len(set().union(*map(set, data.source_folds))) == 62)
    checks["causal_seasonal_features_reproduced_exact"] = bool(seasonal_exact)
    checks["saved_models_reload_predictions_exact"] = bool(reload_exact)

    oof = {key: np.concatenate([part[key] for part in metadata_parts]) for key in META_KEYS}
    seasonal_drift = np.concatenate([part["drift"] for part in prediction_parts])
    noise_full = np.concatenate([part["noise_full"] for part in prediction_parts])
    noise_fast = np.concatenate([part["noise_fast"] for part in prediction_parts])
    tuned_drift = np.concatenate([load_arrays(
        TUNING / f"trial_0002/fold_{outer}_predictions.npz")["scores"][:, 0]
        for outer in range(4)])
    scores = np.column_stack((logit_blend(seasonal_drift, tuned_drift, .90),
                              logit_blend(noise_fast, noise_full, .95)))
    saved_oof = load_arrays(RUN / "oof_predictions.npz")
    checks["fixed_blends_and_oof_reproduced_exact"] = (
        np.array_equal(scores, saved_oof["scores"])
        and all(np.array_equal(oof[key], saved_oof[key]) for key in META_KEYS))
    checks["oof_hash_matches"] = summary["oof_sha256"] == sha(RUN / "oof_predictions.npz")

    diagnostics = expert_diagnostics(oof, scores, events)
    f1 = {"drift": oracle(scores[:, 0], oof, 3), "noise": oracle(scores[:, 1], oof, 4)}
    checks["saved_metrics_reproduced_exact"] = (
        diagnostics == summary["diagnostics"] and f1 == summary["family_oracle_f1"])
    gate_exact = True
    for family in ("drift", "noise"):
        row, control = diagnostics[family], base_summary["candidates"]["mechanism"]["diagnostics"][family]
        f1_row = f1[family]
        control_f1 = base_summary["candidates"]["mechanism"]["family_oracle_f1"][family]
        expected = {
            "macro_ap_improves": row["macro_ap"] > control["macro_ap"],
            "oracle_f1_improves": f1_row["f1"] > control_f1["f1"],
            "oracle_f1_at_least_0_70": f1_row["f1"] >= .70,
            "clean_fpr_within_allowance": row["curve_clean_fpr"] <= control["curve_clean_fpr"] + .00025,
            "post_event_fp_within_allowance": row["curve_post_event_fp"] <= control["curve_post_event_fp"] + 5,
            "worst_ap_preserved": row["worst_ap"] >= control["worst_ap"] - .03}
        gate_exact &= expected == summary["promotion_gates"][family]["checks"]
        gate_exact &= summary["promotion_gates"][family]["passed"] == all(expected.values())
    checks["promotion_gates_reproduced_exact"] = bool(gate_exact)
    checks["both_families_clear_frozen_gate"] = (
        summary["both_families_promoted"]
        and all(summary["promotion_gates"][family]["passed"] for family in ("drift", "noise")))

    result = {"all_checks_pass": bool(all(checks.values())), **checks,
        "summary": summary,
        "claim_limits": {
            "architecture_selected_after_train_oof_inspection": True,
            "oracle_thresholds_not_deployable": True,
            "expanded_data_is_train_only": True,
            "calibration_evaluated": False,
            "validation_evaluated": False,
            "test_evaluated": False}}
    write_json(RUN / "independent_audit.json", result)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT, result)
    print(json.dumps({key: value for key, value in result.items()
                      if isinstance(value, bool)}, indent=2))


if __name__ == "__main__":
    run()
