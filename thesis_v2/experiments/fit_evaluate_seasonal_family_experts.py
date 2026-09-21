"""Full TRAIN fit, calibration-only selection, and late validation evaluation."""
from __future__ import annotations

import argparse
import fcntl
import gc
import json
from pathlib import Path
import time

import joblib
import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score

from wdn.expanded_train_data import ExpandedTrainData
from wdn.models.seasonal_family import SeasonalFamilyExperts
from wdn.models.tuned_family_tree import TunedFamilyTreeConfig, TunedFamilyTrees
from wdn.run_expert_redesign import (
    CampaignData, load_arrays, operating_point, read_json, score_report, sha, write_json)
from wdn.seasonal_pressure_features import endpoint_seasonal_features
from wdn.train_weak_families import add_early


SCREEN = Path("runs/operational/seasonal_family_experts_v3")
BASE = Path("runs/operational/expanded_train_weak_experts_v1")
OLD = Path("runs/operational/blind_residual_experts_v1")
ORIGINAL = Path("data/thesis_v2/operational_modena_seed811")
EXPANSIONS = tuple(Path(f"data/thesis_v2/operational_train_expansion_seed{seed}")
                   for seed in (1811, 2811, 3811))
SPLITS = Path("runs/operational/blind_reference_probe_rank16/splits.json")
PROTOCOL = Path("thesis_v2/SEASONAL_FAMILY_DEPLOYMENT_PROTOCOL.md")
BASELINE_FEATURES = Path("runs/operational/blind_reference_probe_rank16")
PENALTIES = (0., .5, 1., 1.5, 2.)


def atomic_npz(path, **arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def logit(value):
    value = np.clip(np.asarray(value, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(value / (1 - value))


def logit_blend(first, second, first_weight):
    return expit(np.clip(first_weight * logit(first) +
                         (1 - first_weight) * logit(second), -30, 30))


def family_threshold(scores, arrays, family_id, max_fpr=.005):
    """Calibration-only threshold for a specialist's conditional localization."""
    scores = np.asarray(scores, dtype=float)
    scope = np.asarray(arrays["families"]) == family_id
    labels = np.asarray(arrays["labels"])[scope] > 0
    if not labels.any() or labels.all():
        raise ValueError("Family calibration requires positive and negative active-event rows")
    order = np.argsort(-scores[scope], kind="stable")
    ranked_score, ranked_label = scores[scope][order], labels[order]
    ends = np.r_[np.flatnonzero(ranked_score[:-1] != ranked_score[1:]), len(ranked_score) - 1]
    predicted = ends + 1
    true_positive = np.cumsum(ranked_label)[ends]
    f1 = 2 * true_positive / (labels.sum() + predicted)
    precision = true_positive / predicted
    recall = true_positive / labels.sum()
    thresholds = np.nextafter(ranked_score[ends], np.array(-np.inf, dtype=ranked_score.dtype))
    clean = scores[(np.asarray(arrays["families"]) == 0) & (np.asarray(arrays["labels"]) == 0)]
    normal = scores[np.asarray(arrays["labels"]) == 0]
    clean_sorted, normal_sorted = np.sort(clean), np.sort(normal)
    clean_fp = len(clean) - np.searchsorted(clean_sorted, thresholds, side="right")
    normal_fp = len(normal) - np.searchsorted(normal_sorted, thresholds, side="right")
    clean_fpr, normal_fpr = clean_fp / len(clean), normal_fp / len(normal)
    feasible = np.flatnonzero((clean_fpr <= max_fpr) & (normal_fpr <= max_fpr))
    if not len(feasible):
        raise ValueError("No feasible family calibration threshold")
    best = feasible[np.lexsort((-thresholds[feasible], recall[feasible],
                                precision[feasible], f1[feasible]))[-1]]
    return {"threshold": float(thresholds[best]), "family_id": family_id,
        "objective": "family_conditioned_f1", "max_calibration_fpr": max_fpr,
        "calibration": {"f1": float(f1[best]), "precision": float(precision[best]),
            "recall": float(recall[best]), "clean_fpr": float(clean_fpr[best]),
            "all_negative_fpr": float(normal_fpr[best])}}


def family_report(scores, arrays, family_id, point):
    selected = np.asarray(arrays["families"]) == family_id
    labels = np.asarray(arrays["labels"])[selected] > 0
    decision = np.asarray(scores)[selected] > point["threshold"]
    tp = int(np.sum(decision & labels)); fp = int(np.sum(decision & ~labels))
    fn = int(np.sum(~decision & labels)); tn = int(np.sum(~decision & ~labels))
    clean = (np.asarray(arrays["families"]) == 0) & (np.asarray(arrays["labels"]) == 0)
    normal = np.asarray(arrays["labels"]) == 0
    return {"f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
        "auprc": float(average_precision_score(labels, np.asarray(scores)[selected])),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "clean_fpr": float(np.mean(np.asarray(scores)[clean] > point["threshold"])),
        "all_negative_fpr": float(np.mean(np.asarray(scores)[normal] > point["threshold"])),
        "threshold": point["threshold"], "conditional_on_true_family": True}


def specialist_scores(bundle, base_X, seasonal_X):
    extended = np.column_stack((base_X, seasonal_X))
    seasonal = bundle["seasonal"].predict(extended)
    tuned_drift = bundle["tuned_drift"].predict(base_X)[:, 0]
    return np.column_stack((logit_blend(seasonal["drift"], tuned_drift, .90),
                            logit_blend(seasonal["noise_fast"], seasonal["noise_full"], .95)))


def combined_score(name, old_mixture, specialists, centers):
    if name == "old_mixture":
        return np.asarray(old_mixture)
    penalty = float(name.rsplit("_", 1)[1])
    margins = np.column_stack((logit(old_mixture) - centers["old"],
                               logit(specialists[:, 0]) - centers["drift"] - penalty,
                               logit(specialists[:, 1]) - centers["noise"] - penalty))
    return expit(np.max(margins, axis=1))


def run(output_dir):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    lock = (output / "campaign.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Full seasonal deployment run is already active") from error
    if not read_json(SCREEN / "independent_audit.json")["all_checks_pass"]:
        raise RuntimeError("TRAIN-only seasonal screen did not pass its independent audit")
    signature = {"protocol_sha256": sha(PROTOCOL),
        "screen_audit_sha256": sha(SCREEN / "independent_audit.json"),
        "screen_summary_sha256": sha(SCREEN / "summary.json"),
        "base_signature_sha256": sha(BASE / "signature.json"),
        "splits_sha256": sha(SPLITS),
        "old_calibration_predictions_sha256": sha(OLD / "predictions_calibration.npz"),
        "source_sha256": {str(path): sha(path) for path in (
            Path("src/wdn/seasonal_pressure_features.py"), Path("src/wdn/models/seasonal_family.py"),
            Path("src/wdn/models/tuned_family_tree.py"), Path(__file__))},
        "calibration_evaluated": True, "validation_evaluated": True, "test_evaluated": False}
    if (output / "signature.json").exists() and read_json(output / "signature.json") != signature:
        raise ValueError("Full seasonal source/input signature changed")
    write_json(output / "signature.json", signature)
    if (output / "summary.json").exists():
        print("Full seasonal deployment run already complete; no refit", flush=True); return
    started = time.monotonic()

    def status(phase, **details):
        write_json(output / "status.json", {"phase": phase,
            "elapsed_seconds": time.monotonic() - started,
            "test_evaluated": False, **details})
        print(phase, details, flush=True)

    splits = read_json(SPLITS)
    expanded = ExpandedTrainData(ORIGINAL, EXPANSIONS, splits)
    protected = CampaignData(ORIGINAL)
    full = output / "full"; full.mkdir(exist_ok=True)
    reference_path = full / "reference.joblib"
    if not reference_path.exists():
        status("fitting blind normal reference on all expanded TRAIN", scenarios=62)
        joblib.dump(expanded.reference(expanded.scenario_ids), reference_path)
    reference = joblib.load(reference_path)
    train_path = full / "features_train.npz"
    if not train_path.exists():
        status("building full expanded TRAIN base features")
        train, names = expanded.features(expanded.scenario_ids, reference)
        train = add_early(train, expanded.events)
        atomic_npz(train_path, **train); write_json(output / "feature_names.json", names)
    train, names = load_arrays(train_path), read_json(output / "feature_names.json")
    seasonal_train_path = full / "seasonal_train.npz"
    if not seasonal_train_path.exists():
        status("building full expanded TRAIN seasonal features")
        seasonal_train, seasonal_names = endpoint_seasonal_features(train, expanded.scenario)
        atomic_npz(seasonal_train_path, X=seasonal_train)
        write_json(output / "seasonal_feature_names.json", seasonal_names)
    seasonal_names = read_json(output / "seasonal_feature_names.json")

    bundle_path = full / "bundle.joblib"
    if not bundle_path.exists():
        status("fitting frozen Optuna-trial-2 drift auxiliary on full TRAIN")
        config = TunedFamilyTreeConfig(max_leaf_nodes=23, min_samples_leaf=50,
            learning_rate=.1, l2_regularization=5., max_iter=150, seed=2621)
        tuned = TunedFamilyTrees(names, config).fit(train)
        status("fitting frozen seasonal drift/noise experts on full TRAIN")
        seasonal_X = load_arrays(seasonal_train_path)["X"]
        extended_train = {**train, "X": np.column_stack((train["X"], seasonal_X))}
        seasonal = SeasonalFamilyExperts(names + seasonal_names, seed=4100).fit(extended_train)
        joblib.dump({"seasonal": seasonal, "tuned_drift": tuned,
                     "score_blends": {"drift_seasonal": .90, "noise_fast": .95}}, bundle_path)
        del seasonal_X, extended_train, seasonal, tuned
        gc.collect()
    bundle = joblib.load(bundle_path)

    cal_path = full / "features_calibration.npz"
    if not cal_path.exists():
        status("building original calibration features after TRAIN freeze")
        cal, found_names = protected.features(splits["calibration"], reference)
        if found_names != names:
            raise ValueError("Calibration feature schema differs")
        baseline = load_arrays(BASELINE_FEATURES / "features_calibration.npz")
        for key in ("labels", "families"):
            np.testing.assert_array_equal(cal[key], baseline[key])
        atomic_npz(cal_path, **cal)
        seasonal_cal, found_seasonal = endpoint_seasonal_features(cal, protected.scenario)
        if found_seasonal != seasonal_names:
            raise ValueError("Calibration seasonal schema differs")
        atomic_npz(full / "seasonal_calibration.npz", X=seasonal_cal)
    cal = load_arrays(cal_path)
    cal_specialists = specialist_scores(bundle, cal["X"],
        load_arrays(full / "seasonal_calibration.npz")["X"])
    atomic_npz(output / "predictions_calibration.npz", specialists=cal_specialists)
    points = {"drift": family_threshold(cal_specialists[:, 0], cal, 3),
              "noise": family_threshold(cal_specialists[:, 1], cal, 4)}
    old_cal = load_arrays(OLD / "predictions_calibration.npz")
    np.testing.assert_array_equal(cal["labels"], load_arrays(
        BASELINE_FEATURES / "features_calibration.npz")["labels"])
    old_point = operating_point(old_cal["mixture"], cal)
    centers = {"old": logit(old_point["threshold"]),
        "drift": logit(points["drift"]["threshold"]),
        "noise": logit(points["noise"]["threshold"])}
    candidate_names = ["old_mixture"] + [f"seasonal_max_{penalty:g}" for penalty in PENALTIES]
    candidates = {}
    for name in candidate_names:
        score = combined_score(name, old_cal["mixture"], cal_specialists, centers)
        try:
            point = operating_point(score, cal)
        except ValueError:
            continue
        candidates[name] = {"point": point,
            "specialist_penalty": None if name == "old_mixture" else float(name.rsplit("_", 1)[1])}
    if not candidates:
        raise RuntimeError("No feasible calibration detector candidate")
    winner = max(candidates, key=lambda name: (
        candidates[name]["point"]["calibration"]["worst_family_f1"],
        candidates[name]["point"]["calibration"]["macro_f1"],
        candidates[name]["point"]["calibration"]["overall_f1"],
        -candidates[name]["point"]["calibration"]["fpr"], -candidate_names.index(name)))
    selection = {"candidate": winner, "candidates": candidates,
        "family_thresholds": points, "old_mixture_point": old_point,
        "centers": {key: float(value) for key, value in centers.items()},
        "selected_before_validation_feature_extraction": True,
        "calibration_scenarios": splits["calibration"], "test_evaluated": False}
    write_json(output / "selection_frozen.json", selection)
    status("calibration selection frozen; now building reused validation", candidate=winner)

    val_path = full / "features_validation.npz"
    if not val_path.exists():
        val, found_names = protected.features(splits["validation"], reference)
        if found_names != names:
            raise ValueError("Validation feature schema differs")
        baseline = load_arrays(BASELINE_FEATURES / "features_validation.npz")
        for key in ("labels", "families"):
            np.testing.assert_array_equal(val[key], baseline[key])
        atomic_npz(val_path, **val)
        seasonal_val, found_seasonal = endpoint_seasonal_features(val, protected.scenario)
        if found_seasonal != seasonal_names:
            raise ValueError("Validation seasonal schema differs")
        atomic_npz(full / "seasonal_validation.npz", X=seasonal_val)
    val = load_arrays(val_path)
    val_specialists = specialist_scores(bundle, val["X"],
        load_arrays(full / "seasonal_validation.npz")["X"])
    old_val = load_arrays(OLD / "predictions_validation.npz")
    selected_cal_score = combined_score(winner, old_cal["mixture"], cal_specialists, centers)
    selected_val_score = combined_score(winner, old_val["mixture"], val_specialists, centers)
    point = candidates[winner]["point"]
    validation = score_report(selected_val_score, val, point)
    baseline_validation = score_report(old_val["mixture"], val, old_point)
    specialist_validation = {
        "drift": family_report(val_specialists[:, 0], val, 3, points["drift"]),
        "noise": family_report(val_specialists[:, 1], val, 4, points["noise"])}
    specialist_calibration = {
        "drift": family_report(cal_specialists[:, 0], cal, 3, points["drift"]),
        "noise": family_report(cal_specialists[:, 1], cal, 4, points["noise"])}
    atomic_npz(output / "predictions_validation.npz", specialists=val_specialists,
        detector=selected_val_score, old_mixture=old_val["mixture"],
        labels=val["labels"], families=val["families"], scenario=val["scenario"],
        timestep=val["timestep"], node=val["node"])
    result = {"status": "completed", "selection": selection,
        "specialists": {"calibration": specialist_calibration,
                        "development_validation": specialist_validation},
        "whole_detector": {"selected": validation, "old_mixture_recalibrated": baseline_validation},
        "train_oof_reference": read_json(SCREEN / "summary.json")["family_oracle_f1"],
        "claim_limits": {"architecture_selected_on_train_oof": True,
            "thresholds_and_detector_candidate_selected_on_calibration": True,
            "validation_is_reused_development_data": True,
            "validation_evaluated_once_after_this_selection_freeze": True,
            "test_evaluated": False},
        "hashes": {"bundle": sha(bundle_path),
            "selection": sha(output / "selection_frozen.json"),
            "calibration_predictions": sha(output / "predictions_calibration.npz"),
            "validation_predictions": sha(output / "predictions_validation.npz")},
        "elapsed_seconds": time.monotonic() - started,
        "calibration_evaluated": True, "validation_evaluated": True, "test_evaluated": False}
    write_json(output / "summary.json", result)
    status("full seasonal calibration/validation complete", candidate=winner,
           drift_f1=specialist_validation["drift"]["f1"],
           noise_f1=specialist_validation["noise"]["f1"],
           overall_f1=validation["validation"]["overall"]["f1"],
           replay_f1=validation["validation"]["per_family"]["replay"]["f1"])
    print(json.dumps({"selection": winner, "specialist_validation": specialist_validation,
        "whole_detector": validation}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/operational/seasonal_family_deployment_v1")
    run(parser.parse_args().output_dir)
