"""Cross-fitted TRAIN-only residual fusion after the standalone GRU screen."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from wdn.models.family_specific import _balanced_weights
from wdn.run_expert_redesign import load_arrays, read_json, sha, write_json
from wdn.screen_family_specific import _best_f1

BASE = Path("runs/operational/mechanism_redesign_v1")
FAMILY = Path("runs/operational/family_specific_experts_v1")
SEQUENCE = Path("runs/operational/causal_sequence_experts_v1")
PROTOCOL = Path("thesis_v2/CAUSAL_SEQUENCE_FUSION_PROTOCOL.md")
PHYSICAL = {
    3: ("dynamic_innovation", "dynamic_cusum_positive", "dynamic_cusum_negative",
        "drift_ramp_strength_3", "drift_ramp_strength_5", "drift_ramp_strength_8",
        "drift_mean_strength_5", "drift_consistency_8"),
    4: ("dynamic_abs_innovation", "dynamic_past_std_8", "dynamic_innovation_rms_8",
        "noise_state_2.0", "noise_state_4.0", "noise_energy_3", "noise_energy_5",
        "noise_energy_8")}


def run(output_path):
    output_path = Path(output_path)
    if output_path.exists():
        print("Causal sequence fusion screen already complete; no refit"); return
    names = read_json(BASE/"feature_names.json")
    folds = read_json(FAMILY/"outer_folds.json")
    parts = [load_arrays(path) for path in sorted(BASE.glob("fold_*/features_held_out.npz"))]
    arrays = {key: np.concatenate([part[key] for part in parts]) for key in
        ("X", "labels", "families", "event", "scenario", "timestep", "node")}
    saved = load_arrays(SEQUENCE/"oof_predictions.npz"); arrays["early"] = saved["early"]
    baseline = load_arrays(FAMILY/"oof_predictions.npz")["scores_E"]
    short = load_arrays(SEQUENCE/"short/oof_raw.npz")["scores"]
    long = load_arrays(SEQUENCE/"long/oof_raw.npz")["scores"]
    predictions, results = {}, {}
    for family_id, column, family in ((3, 0, "drift"), (4, 1, "noise")):
        score_inputs = np.column_stack([logit(np.clip(values[:, column], 1e-6, 1-1e-6))
                                        for values in (baseline, short, long)])
        physical = arrays["X"][:, [names.index(name) for name in PHYSICAL[family_id]]]
        methods = {"logistic_scores": score_inputs,
            "logistic_mechanism": np.column_stack((score_inputs, physical)),
            "tree_mechanism": np.column_stack((score_inputs, physical))}
        family_mask = arrays["families"] == family_id
        for method, features in methods.items():
            output = np.zeros(len(arrays["labels"]), dtype=np.float64)
            for outer, held in enumerate(folds):
                fit = ~np.isin(arrays["scenario"], held); predict = ~fit
                subset = {key: value[fit] for key, value in arrays.items()}
                selected, weights = _balanced_weights(subset, family_id, 2., 2500+outer)
                scaler = StandardScaler().fit(features[fit][selected], sample_weight=weights)
                train_x = scaler.transform(features[fit][selected])
                target = ((subset["families"][selected] == family_id)
                          & (subset["labels"][selected] > 0))
                if method == "tree_mechanism":
                    model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7,
                        min_samples_leaf=30, learning_rate=.05, l2_regularization=20.,
                        random_state=2500+outer)
                else:
                    model = LogisticRegression(C=.05, max_iter=1000, random_state=2500+outer)
                model.fit(train_x, target, sample_weight=weights)
                output[predict] = model.predict_proba(scaler.transform(features[predict]))[:, 1]
            by_scenario = {str(int(sid)): float(average_precision_score(
                arrays["labels"][family_mask & (arrays["scenario"] == sid)],
                output[family_mask & (arrays["scenario"] == sid)]))
                for sid in np.unique(arrays["scenario"][family_mask])}
            result = {"macro_ap": float(np.mean(list(by_scenario.values()))),
                "worst_ap": float(min(by_scenario.values())), "by_scenario_ap": by_scenario,
                "family_oracle_f1": _best_f1(output[family_mask], arrays["labels"][family_mask])}
            results[f"{family}_{method}"] = result; predictions[f"{family}_{method}"] = output
    np.savez_compressed(SEQUENCE/"fusion_predictions.npz", **predictions)
    base_summary = read_json(SEQUENCE/"summary.json")
    best = {family: max((key for key in results if key.startswith(family+"_")),
        key=lambda key: (results[key]["macro_ap"], results[key]["worst_ap"]))
        for family in ("drift", "noise")}
    checks = {}
    for family in ("drift", "noise"):
        candidate = results[best[family]]
        control = base_summary["baseline"]["diagnostics"][family]
        checks[family] = {"macro_ap_improves": candidate["macro_ap"] > control["macro_ap"],
            "worst_ap_not_lower": candidate["worst_ap"] >= control["worst_ap"],
            "meaningful_macro_gain_0_015": candidate["macro_ap"] >= control["macro_ap"]+.015}
    report = {"scope": "TRAIN-only scenario-cross-fitted residual fusion",
        "protocol_sha256": sha(PROTOCOL), "results": results, "selected": best,
        "checks": checks, "both_families_passed": all(all(row.values()) for row in checks.values()),
        "next_action": "stop_before_calibration", "calibration_evaluated": False,
        "validation_evaluated": False, "test_evaluated": False,
        "input_hashes": {"sequence_summary": sha(SEQUENCE/"summary.json"),
            "sequence_oof": sha(SEQUENCE/"oof_predictions.npz"),
            "baseline_oof": sha(FAMILY/"oof_predictions.npz")},
        "predictions_sha256": sha(SEQUENCE/"fusion_predictions.npz")}
    write_json(output_path, report); print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run(SEQUENCE/"fusion_screen.json")
