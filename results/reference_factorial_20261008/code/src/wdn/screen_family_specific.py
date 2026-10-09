"""Nested TRAIN-only screen for separate drift and noise expert architectures."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import time

import joblib
import numpy as np

from wdn.family_state_features import family_state_features
from wdn.models.family_specific import FamilySpecificExpert, MonotoneFamilyStacker
from wdn.run_expert_redesign import load_arrays, read_json, sha, write_json
from wdn.screen_latent_ar import STAGE_A
from wdn.screen_normal_nuisance import (BASE, DATA, EXPECTED_FOLDS,
    TrainOnlyCampaign, atomic_joblib, atomic_npz, concatenate, endpoint_key_digest,
    secondary_curve)
from wdn.screen_recovery import expert_diagnostics
from wdn.train_weak_families import add_early


WEAK = Path("runs/operational/weak_family_v1/trial_0000")
LATENT = Path("runs/operational/latent_ar_stage_b_v1")
SOURCE = ("src/wdn/screen_family_specific.py", "src/wdn/family_state_features.py",
          "src/wdn/models/family_specific.py", "thesis_v2/FAMILY_SPECIFIC_EXPERTS_PROTOCOL.md")
PROTOCOL = {"version": 1, "stage": "family_specific_experts",
    "families": ["drift", "noise"], "directed_feature_caches": 6,
    "expert_fit_calls": 18, "stacker_fit_calls": 6, "optuna_trials": 0,
    "weak_head_fits_after_screen": 0,
    "gates": {"macro_ap_gain": .05, "improved_scenarios": 3,
        "maximum_worst_ap_drop": .03, "early_recall_improves": True,
        "clean_and_post_event_fp_not_increased": True,
        "minimum_family_oracle_f1": .70, "minimum_macro_ap": .65},
    "diagnostic_fpr": .001, "secondary_diagnostic_fpr": .005,
    "calibration_evaluated": False, "validation_evaluated": False,
    "test_evaluated": False}
PROTOCOL = json.loads(json.dumps(PROTOCOL))
ARRAY_KEYS = ("X", "labels", "families", "event", "scenario", "timestep", "node", "early")


def _best_f1(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels) > 0
    order = np.argsort(-scores, kind="stable")
    score, y = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(score[:-1] != score[1:]), len(score)-1]
    predicted, tp = ends+1, np.cumsum(y)[ends]
    fp, fn = predicted-tp, y.sum()-tp
    f1 = 2*tp/np.maximum(2*tp+fp+fn, 1)
    best = int(np.argmax(f1))
    return {"f1": float(f1[best]), "precision": float(tp[best]/max(predicted[best], 1)),
        "recall": float(tp[best]/max(y.sum(), 1)),
        "threshold": float(np.nextafter(score[ends[best]], np.array(-np.inf, dtype=score.dtype))),
        "positives": int(y.sum()), "negatives": int((~y).sum())}


def _with_state(arrays, names, events):
    arrays = add_early({key: arrays[key] for key in ARRAY_KEYS if key != "early"}, events)
    state, state_names = family_state_features(arrays, names)
    return {**arrays, "X": np.column_stack((arrays["X"], state))}, names+state_names


def _verify_inputs():
    latent_summary = read_json(LATENT/"summary.json")
    latent_audit = read_json(LATENT/"independent_audit.json")
    latent_signature = read_json(LATENT/"signature.json")
    if (latent_summary["survivors"] or any(latent_summary[key] for key in
            ("calibration_evaluated", "validation_evaluated", "test_evaluated"))):
        raise ValueError("Latent AR completion state changed")
    required = ("saved_oof_metrics_reproduced_exact", "artifact_hashes_match",
                "stage_a_metadata_and_baseline_exact", "accessed_train_scenarios_only")
    if not all(latent_audit.get(key) is True for key in required):
        raise ValueError("Latent AR independent audit is incomplete")
    if sha(LATENT/"oof_predictions.npz") != latent_audit["oof_predictions_sha256"]:
        raise ValueError("Latent AR OOF changed")
    stage_signature = read_json(STAGE_A/"signature.json")
    folds, splits = stage_signature["folds"], stage_signature["splits"]
    if folds != EXPECTED_FOLDS or set().union(*map(set, folds)) != set(splits["train"]):
        raise ValueError("TRAIN folds changed")
    for train in range(3):
        folder = STAGE_A/f"shared_references/train_F{train}"
        record = read_json(folder/"completed.json")
        if (record["fit_scenarios"] != folds[train]
                or sha(folder/"reference.joblib") != record["model_sha256"]):
            raise ValueError("Frozen directed reference changed")
    base_summary = read_json(BASE/"summary.json")
    for name, expected in base_summary["source_and_inputs"]["source"].items():
        if sha(Path("src/wdn")/name) != expected:
            raise ValueError(f"Frozen base feature source changed: {name}")
    for name, expected in base_summary["source_and_inputs"]["data_files"].items():
        if sha(DATA/name) != expected:
            raise ValueError(f"Frozen data changed: {name}")
    weak = load_arrays(WEAK/"oof_predictions.npz")
    latent = load_arrays(LATENT/"oof_predictions.npz")
    for key in ("labels", "families", "scenario"):
        np.testing.assert_array_equal(weak[key], latent[key])
    return splits, folds, stage_signature, latent_audit, base_summary


def _directed_cache(output, data, names, folds, train, held):
    folder = output/f"directed_features/train_F{train}__held_F{held}"
    folder.mkdir(parents=True, exist_ok=True)
    feature_path, completed = folder/"features.npz", folder/"completed.json"
    reference_path = STAGE_A/f"shared_references/train_F{train}/reference.joblib"
    if not completed.exists():
        reference = joblib.load(reference_path)
        arrays, found_names = data.features(folds[held], reference)
        if found_names != names:
            raise ValueError("Directed base feature schema changed")
        arrays, full_names = _with_state(arrays, names, data.events)
        atomic_npz(feature_path, **{key: arrays[key] for key in ARRAY_KEYS})
        write_json(folder/"feature_names.json", full_names)
        write_json(folder/"fit_scope.json", {"reference_fit_scenarios": folds[train],
            "predicted_scenarios": folds[held], "reference_sha256": sha(reference_path),
            "rows": len(arrays["labels"]), "endpoint_key_digest": endpoint_key_digest(arrays)})
        write_json(completed, {"features_sha256": sha(feature_path),
            "feature_names_sha256": sha(folder/"feature_names.json"),
            "fit_scope_sha256": sha(folder/"fit_scope.json"),
            "reference_sha256": sha(reference_path), "fit_scenarios": folds[train],
            "predicted_scenarios": folds[held]})
    record, scope = read_json(completed), read_json(folder/"fit_scope.json")
    if (record["fit_scenarios"] != folds[train] or record["predicted_scenarios"] != folds[held]
            or set(folds[train]) & set(folds[held]) or sha(feature_path) != record["features_sha256"]
            or sha(folder/"feature_names.json") != record["feature_names_sha256"]
            or sha(folder/"fit_scope.json") != record["fit_scope_sha256"]
            or sha(reference_path) != record["reference_sha256"]
            or scope["endpoint_key_digest"] != endpoint_key_digest(load_arrays(feature_path))):
        raise ValueError("Directed expert feature cache mismatch")
    return load_arrays(feature_path), read_json(folder/"feature_names.json"), record


def _gate(candidate, baseline, oracle):
    gains = {sid: candidate["by_scenario_ap"][sid]-value
             for sid, value in baseline["by_scenario_ap"].items()}
    checks = {"macro_ap_gain": candidate["macro_ap"] >= baseline["macro_ap"]+.05,
        "at_least_three_scenarios_improve": sum(value > 0 for value in gains.values()) >= 3,
        "worst_scenario_preserved": candidate["worst_ap"] >= baseline["worst_ap"]-.03,
        "early_recall_improves": candidate["early_macro_recall"] > baseline["early_macro_recall"],
        "clean_fpr_not_increased": candidate["curve_clean_fp"] <= baseline["curve_clean_fp"],
        "post_event_fp_not_increased": candidate["curve_post_event_fp"] <= baseline["curve_post_event_fp"],
        "family_oracle_f1": oracle["f1"] >= .70,
        "minimum_macro_ap": candidate["macro_ap"] >= .65}
    return {"passed": all(checks.values()), "checks": checks,
        "scenario_ap_gains": gains, "macro_ap_gain": candidate["macro_ap"]-baseline["macro_ap"]}


def _verify_completed(output, signature):
    summary = read_json(output/"summary.json")
    if sha(output/"oof_predictions.npz") != summary["artifact_manifest"]["oof_predictions_sha256"]:
        raise ValueError("Completed expert OOF hash mismatch")
    for outer, held in enumerate(signature["folds"]):
        folder = output/f"outer_{outer}"
        record = read_json(folder/"completed.json")
        if (record["held_scenarios"] != held
                or sha(folder/"bundle.joblib") != record["bundle_sha256"]
                or sha(folder/"held_predictions.npz") != record["predictions_sha256"]
                or sha(folder/"fit_scope.json") != record["fit_scope_sha256"]):
            raise ValueError("Completed family expert outer artifact mismatch")
    if any(summary[key] for key in
           ("calibration_evaluated", "validation_evaluated", "test_evaluated")):
        raise ValueError("Completed family expert run touched a forbidden split")


def run(output_dir, preflight_only=False):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/"campaign.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Family-specific screen is already running") from error

    splits, folds, stage_signature, latent_audit, base_summary = _verify_inputs()
    base_names = read_json(BASE/"feature_names.json")
    signature = {"sources": {path: sha(Path(path)) for path in SOURCE}, "protocol": PROTOCOL,
        "splits": splits, "folds": folds, "data_files": base_summary["source_and_inputs"]["data_files"],
        "stage_a_signature_sha": sha(STAGE_A/"signature.json"),
        "latent_oof_sha": latent_audit["oof_predictions_sha256"],
        "legacy_local_oof_sha": sha(WEAK/"oof_predictions.npz"),
        "base_feature_names_sha": sha(BASE/"feature_names.json"),
        "base_held_features": {str(path): sha(path) for path in sorted(BASE.glob("fold_*/features_held_out.npz"))},
        "shared_references": {str(path): sha(path) for path in
            sorted(STAGE_A.glob("shared_references/train_F*/reference.joblib"))}}
    if (output/"signature.json").exists() and read_json(output/"signature.json") != signature:
        raise ValueError("Family-specific source/input signature changed; refusing silent resume")
    write_json(output/"signature.json", signature)
    write_json(output/"protocol.json", PROTOCOL)
    write_json(output/"outer_folds.json", folds)
    if preflight_only:
        print("Family-specific expert preflight complete; no data extracted and no model fitted", flush=True)
        return
    if (output/"summary.json").exists():
        _verify_completed(output, signature)
        print("Family-specific expert screen complete; artifacts verified; no refit", flush=True)
        return

    started = time.monotonic()
    def status(phase, **details):
        write_json(output/"status.json", {"phase": phase,
            "elapsed_seconds": time.monotonic()-started,
            "calibration_evaluated": False, "validation_evaluated": False,
            "test_evaluated": False, **details})
        print(phase, details, flush=True)

    data = TrainOnlyCampaign(DATA, splits)
    status("building six directed TRAIN feature caches")
    directed, full_names, directed_records = {}, None, {}
    for train in range(3):
        for held in range(3):
            if train == held:
                continue
            arrays, found_names, record = _directed_cache(output, data, base_names, folds, train, held)
            if full_names is not None and full_names != found_names:
                raise ValueError("Directed full feature schemas differ")
            full_names = found_names
            directed[(train, held)] = arrays
            directed_records[f"F{train}->F{held}"] = record
    write_json(output/"feature_names.json", full_names)

    latent = load_arrays(LATENT/"oof_predictions.npz")
    legacy = load_arrays(WEAK/"oof_predictions.npz")
    legacy_scores = legacy["experts"][:, 3:5]
    all_parts, all_scores, all_heads, bundle_metadata = [], [], [], []
    cursor = 0
    for outer, held_scenarios in enumerate(folds):
        folder = output/f"outer_{outer}"
        folder.mkdir(exist_ok=True)
        completed = folder/"completed.json"
        training_folds = [index for index in range(3) if index != outer]
        left, right = training_folds
        left_arrays = directed[(right, left)]
        right_arrays = directed[(left, right)]
        training = concatenate([left_arrays, right_arrays])
        expected_fit = sorted(folds[left]+folds[right])
        if sorted(np.unique(training["scenario"]).tolist()) != expected_fit:
            raise ValueError("Outer expert training scope mismatch")

        base_held = load_arrays(BASE/f"fold_{outer}/features_held_out.npz")
        held_arrays, held_names = _with_state(base_held, base_names, data.events)
        if held_names != full_names or sorted(np.unique(held_arrays["scenario"]).tolist()) != held_scenarios:
            raise ValueError("Outer held expert feature mismatch")
        end = cursor+len(held_arrays["labels"])
        for key in ("labels", "families", "event", "scenario", "timestep", "node", "early"):
            np.testing.assert_array_equal(held_arrays[key], latent[key][cursor:end])

        if not completed.exists():
            status("training separate family experts", outer_fold=outer,
                   held_scenarios=held_scenarios, fit_scenarios=expected_fit)
            bundle, held_family_scores, held_head_scores, metadata = {}, [], [], {}
            for family in ("drift", "noise"):
                left_model = FamilySpecificExpert(full_names, family).fit(left_arrays)
                right_model = FamilySpecificExpert(full_names, family).fit(right_arrays)
                inner_scores = np.concatenate((right_model.predict_heads(left_arrays["X"]),
                                               left_model.predict_heads(right_arrays["X"])))
                stacker = MonotoneFamilyStacker(family).fit(inner_scores, training)
                final_model = FamilySpecificExpert(full_names, family).fit(training)
                held_heads = final_model.predict_heads(held_arrays["X"])
                held_score = stacker.predict(held_heads)
                bundle[family] = {"expert": final_model, "stacker": stacker}
                held_family_scores.append(held_score)
                held_head_scores.append(held_heads)
                metadata[family] = {"expert": final_model.metadata(),
                    "stacker": stacker.metadata(), "inner_oof_rows": len(inner_scores)}
            held_family_scores = np.column_stack(held_family_scores)
            held_head_scores = np.concatenate(held_head_scores, axis=1)
            atomic_joblib(bundle, folder/"bundle.joblib")
            atomic_npz(folder/"held_predictions.npz",
                **{key: held_arrays[key] for key in ARRAY_KEYS if key != "X"},
                scores_E=held_family_scores, head_scores=held_head_scores)
            reloaded = joblib.load(folder/"bundle.joblib")
            replay = np.column_stack([reloaded[family]["stacker"].predict(
                reloaded[family]["expert"].predict_heads(held_arrays["X"]))
                for family in ("drift", "noise")])
            np.testing.assert_array_equal(replay, held_family_scores)
            write_json(folder/"fit_scope.json", {"outer_fold": outer,
                "held_scenarios": held_scenarios, "fit_scenarios": expected_fit,
                "inner_directions": [f"F{right}->F{left}", f"F{left}->F{right}"],
                "model_metadata": metadata, "reload_predictions_exact": True})
            write_json(completed, {"held_scenarios": held_scenarios,
                "bundle_sha256": sha(folder/"bundle.joblib"),
                "predictions_sha256": sha(folder/"held_predictions.npz"),
                "fit_scope_sha256": sha(folder/"fit_scope.json")})
        record = read_json(completed)
        if (record["held_scenarios"] != held_scenarios
                or sha(folder/"bundle.joblib") != record["bundle_sha256"]
                or sha(folder/"held_predictions.npz") != record["predictions_sha256"]
                or sha(folder/"fit_scope.json") != record["fit_scope_sha256"]):
            raise ValueError("Completed outer family expert artifact mismatch")
        saved = load_arrays(folder/"held_predictions.npz")
        for key in ("labels", "families", "event", "scenario", "timestep", "node", "early"):
            np.testing.assert_array_equal(saved[key], held_arrays[key])
        all_parts.append({key: saved[key] for key in
            ("labels", "families", "event", "scenario", "timestep", "node", "early")})
        all_scores.append(saved["scores_E"])
        all_heads.append(saved["head_scores"])
        bundle_metadata.append(read_json(folder/"fit_scope.json")["model_metadata"])
        cursor = end
    if cursor != len(latent["labels"]):
        raise ValueError("Family expert held rows do not cover TRAIN OOF")
    arrays = concatenate(all_parts)
    scores_e, head_scores = np.concatenate(all_scores), np.concatenate(all_heads)
    for key in ("labels", "families", "event", "scenario", "timestep", "node", "early"):
        np.testing.assert_array_equal(arrays[key], latent[key])
    diagnostics = {"L": expert_diagnostics(arrays, legacy_scores, [
        {**event, "_event_id": event_id} for event_id, event in enumerate(data.events)
        if event["scenario_id"] in set(splits["train"])]),
        "E": expert_diagnostics(arrays, scores_e, [
        {**event, "_event_id": event_id} for event_id, event in enumerate(data.events)
        if event["scenario_id"] in set(splits["train"])])}
    oracle, gates, survivors = {}, {}, []
    for family, family_id, column in (("drift", 3, 0), ("noise", 4, 1)):
        selected = arrays["families"] == family_id
        oracle[family] = _best_f1(scores_e[selected, column], arrays["labels"][selected])
        gates[family] = _gate(diagnostics["E"][family], diagnostics["L"][family], oracle[family])
        if gates[family]["passed"]:
            survivors.append(family)
    secondary = {arm: secondary_curve(arrays, score, [
        {**event, "_event_id": event_id} for event_id, event in enumerate(data.events)
        if event["scenario_id"] in set(splits["train"])], .005)
        for arm, score in (("L", legacy_scores), ("E", scores_e))}
    atomic_npz(output/"oof_predictions.npz", **arrays, scores_L=legacy_scores,
               scores_E=scores_e, head_scores=head_scores)
    for path, expected in signature["sources"].items():
        if sha(Path(path)) != expected:
            raise ValueError("Family expert source changed during execution")
    manifest = {"oof_predictions_sha256": sha(output/"oof_predictions.npz"),
        "directed_features": directed_records,
        "outer": {str(index): read_json(output/f"outer_{index}/completed.json") for index in range(3)}}
    result = {"scope": "TRAIN-only nested family-specific expert screen",
        "protocol": PROTOCOL, "diagnostics": diagnostics, "family_oracle_f1": oracle,
        "gates": gates, "secondary_fpr_005": secondary, "survivors": survivors,
        "bundle_metadata": bundle_metadata, "artifact_manifest": manifest,
        "endpoints": len(arrays["labels"]), "features": len(full_names),
        "accessed_scenarios": sorted(data.accessed), "directed_feature_caches": 6,
        "expert_fit_calls": 18, "stacker_fit_calls": 6, "optuna_trials": 0,
        "calibration_evaluated": False, "validation_evaluated": False,
        "test_evaluated": False, "elapsed_seconds": time.monotonic()-started,
        "next_action": ("eligible_for_independent_family_expert_audit" if survivors
                        else "stop_no_family_specific_expert_passed")}
    write_json(output/"summary.json", result)
    write_json(output/"verification.json", {"legacy_oof_exact": True,
        "held_metadata_exact": True, "model_reload_predictions_exact": True,
        "source_hashes_match": True, "artifact_hashes_checked": True,
        "accessed_train_scenarios_only": set(data.accessed) <= set(splits["train"]),
        "calibration_evaluated": False, "validation_evaluated": False,
        "test_evaluated": False})
    status("Family-specific expert screen complete", survivors=survivors,
           next_action=result["next_action"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="runs/operational/family_specific_experts_v1")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    run(args.output_dir, args.preflight_only)
