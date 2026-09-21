"""Bounded scenario-nested TRAIN-only screen for causal latent AR error."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import time

import joblib
import numpy as np
import yaml

from wdn.conditional_change import ConditionalChangeConfig, ConditionalChangeFilter
from wdn.models.latent_ar import LatentARConfig, LatentARNormalError
from wdn.run_expert_redesign import load_arrays, read_json, sha, write_json
from wdn.screen_normal_nuisance import (BASE, CONTROL, DATA, RECOVERY, EXPECTED_FOLDS,
    TrainOnlyCampaign, atomic_joblib, atomic_npz, concatenate, grouped_metrics,
    observation_metadata, secondary_curve)
from wdn.screen_recovery import expert_diagnostics, gate, post_event_mask


STAGE_A = Path("runs/operational/normal_nuisance_stage_a_v2")
AR_CONFIG = LatentARConfig()
CHANGE_CONFIG = ConditionalChangeConfig()
SOURCE = ("src/wdn/screen_latent_ar.py", "src/wdn/models/latent_ar.py",
          "src/wdn/conditional_change.py", "thesis_v2/LATENT_AR_STAGE_B_PROTOCOL.md")
PROTOCOL = {"version": 1, "stage": "latent_ar_B",
    "ar_model": asdict(AR_CONFIG), "change_filter": asdict(CHANGE_CONFIG),
    "normal_gate": {"macro_mae_ratio": .85, "improved_scenarios": 10,
        "macro_nll_improves": True, "no_outer_nll_worse": True,
        "every_outer_abs_z_gt3_decreases": True,
        "macro_mean_width95_ratio": 1.25,
        "maximum_absolute_pooled_lag1": .20,
        "every_outer_absolute_lag1_decreases": True},
    "evidence_gate": {"macro_ap_gain": .05, "improved_scenarios": 3,
        "max_worst_ap_drop": .05, "diagnostic_fpr": .001,
        "minimum_control_macro_ap_fraction": .75,
        "clean_and_post_event_fp_not_increased": True},
    "secondary_diagnostic_fpr": .005, "ar_bundle_fits": 3,
    "weak_head_fits": 0, "optuna_trials": 0,
    "calibration_evaluated": False, "validation_evaluated": False,
    "test_evaluated": False}
PROTOCOL = json.loads(json.dumps(PROTOCOL))


def _verify_stage_a():
    summary = read_json(STAGE_A/"summary.json")
    audit = read_json(STAGE_A/"independent_audit.json")
    signature = read_json(STAGE_A/"signature.json")
    if (summary["survivors"] or summary["next_action"] != "stop_no_normal_nuisance_candidate_passed"
            or any(summary[key] for key in
                   ("calibration_evaluated", "validation_evaluated", "test_evaluated"))):
        raise ValueError("Stage A completion state changed")
    required = ("saved_oof_metrics_reproduced_exact", "normal_gate_reproduced_exact",
        "evidence_gates_reproduced_exact", "artifact_hashes_match",
        "metadata_matches_frozen_control", "accessed_train_scenarios_only")
    if not all(audit.get(key) is True for key in required):
        raise ValueError("Stage A independent audit is incomplete")
    if sha(STAGE_A/"oof_predictions.npz") != audit["oof_predictions_sha256"]:
        raise ValueError("Stage A OOF hash changed")
    for path, expected in signature["sources"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"Frozen Stage A source changed: {path}")
    prior = read_json(RECOVERY/"signature.json")
    if sha(RECOVERY/"signature.json") != signature["prior_signature_sha"]:
        raise ValueError("Recovery signature changed")
    for path, expected in prior["prior_signature"]["base_signature"]["data_files"].items():
        if sha(DATA/path) != expected:
            raise ValueError(f"Frozen data changed: {path}")
    if sha(CONTROL/"trial_0000/oof_predictions.npz") != signature["control_oof_sha"]:
        raise ValueError("Frozen local-control OOF changed")
    for path, expected in signature["outer_references"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"Frozen outer reference changed: {path}")
    config = yaml.safe_load((DATA/"generate_config.yaml").read_text())
    if config["missing_rate_pressure"] != .5 or config["missing_rate_flow"] != .5:
        raise ValueError("Pressure and flow missing rates must remain 0.50")
    folds, splits = signature["folds"], signature["splits"]
    if folds != EXPECTED_FOLDS or set().union(*map(set, folds)) != set(splits["train"]):
        raise ValueError("Frozen TRAIN folds changed")
    for outer, held in enumerate(folds):
        folder = STAGE_A/f"outer_{outer}"
        record = read_json(folder/"completed.json")
        if (record["held_scenarios"] != held
                or sha(folder/"held_predictions.npz") != record["predictions_sha256"]
                or sha(folder/"fit_scope.json") != record["fit_scope_sha256"]):
            raise ValueError("Stage A outer artifact changed")
    for train in range(3):
        for held in range(3):
            if train == held:
                continue
            folder = STAGE_A/f"directed_predictions/train_F{train}__held_F{held}"
            record = read_json(folder/"completed.json")
            if (record["fit_scenarios"] != folds[train]
                    or record["predicted_scenarios"] != folds[held]
                    or sha(folder/"normal_rows.npz") != record["rows_sha256"]):
                raise ValueError("Stage A directed cache changed")
    return summary, audit, signature


def _normal_diagnostics(arrays, events):
    post = (post_event_mask(arrays, events, "stealthy")
            | post_event_mask(arrays, events, "noise"))
    groups = {"clean": arrays["families"] == 0,
              "all_label_zero": arrays["labels"] == 0, "post_event": post}
    return {arm: {name: grouped_metrics(arrays["error"], arrays[f"mean_{arm}"],
        arrays[f"scale_{arm}"], selected, arrays["scenario"], arrays["outer_fold"])
        for name, selected in groups.items()} for arm in ("B", "A")}


def _lag_one(arrays, arm):
    selected = arrays["families"] == 0
    residual = arrays["error"]-arrays[f"mean_{arm}"]
    index = np.flatnonzero(selected)
    order = np.lexsort((arrays["timestep"][index], arrays["node"][index],
                        arrays["scenario"][index]))
    index = index[order]
    adjacent = ((arrays["scenario"][index][1:] == arrays["scenario"][index][:-1])
        & (arrays["node"][index][1:] == arrays["node"][index][:-1])
        & (np.diff(arrays["timestep"][index]) == 1))
    left, right = index[:-1][adjacent], index[1:][adjacent]
    if len(left) < 2:
        raise ValueError("Lag-one diagnostics require clean consecutive pairs")

    def correlation(loc):
        if loc.sum() < 2:
            raise ValueError("Each lag-one group requires at least two pairs")
        value = float(np.corrcoef(residual[left[loc]], residual[right[loc]])[0, 1])
        if not np.isfinite(value):
            raise FloatingPointError("Nonfinite lag-one correlation")
        return {"pairs": int(loc.sum()), "pooled_pearson_r": value}

    result = correlation(np.ones(len(left), dtype=bool))
    result["by_outer"] = {str(int(fold)): correlation(arrays["outer_fold"][right] == fold)
        for fold in np.unique(arrays["outer_fold"][right])}
    return result


def _normal_gate(normal, lag):
    baseline, candidate = normal["B"]["clean"], normal["A"]["clean"]
    bmacro, amacro = baseline["scenario_macro"], candidate["scenario_macro"]
    improved = sum(candidate["by_scenario"][sid]["mae_m"] < row["mae_m"]
                   for sid, row in baseline["by_scenario"].items())
    checks = {"macro_mae_ratio": amacro["mae_m"] <= .85*bmacro["mae_m"],
        "at_least_ten_scenarios_improve_mae": improved >= 10,
        "macro_nll_improves": amacro["nll"] < bmacro["nll"],
        "no_outer_nll_worse": all(candidate["by_outer"][key]["nll"] <= row["nll"]
                                  for key, row in baseline["by_outer"].items()),
        "every_outer_abs_z_gt3_decreases": all(
            candidate["by_outer"][key]["fraction_abs_z_gt3"] < row["fraction_abs_z_gt3"]
            for key, row in baseline["by_outer"].items()),
        "interval_width_bounded": amacro["mean_width95_m"] <= 1.25*bmacro["mean_width95_m"],
        "absolute_pooled_lag1_below_point_two": abs(lag["A"]["pooled_pearson_r"]) <= .20,
        "every_outer_absolute_lag1_decreases": all(
            abs(lag["A"]["by_outer"][key]["pooled_pearson_r"])
            < abs(row["pooled_pearson_r"]) for key, row in lag["B"]["by_outer"].items())}
    return {"passed": all(checks.values()), "checks": checks,
        "macro_mae_ratio": amacro["mae_m"]/bmacro["mae_m"],
        "macro_mean_width95_ratio": amacro["mean_width95_m"]/bmacro["mean_width95_m"],
        "improved_scenarios": improved}


def _fit_rows_for_outer(outer, folds):
    training_folds = [index for index in range(3) if index != outer]
    left, right = training_folds
    paths = (STAGE_A/f"directed_predictions/train_F{left}__held_F{right}/normal_rows.npz",
             STAGE_A/f"directed_predictions/train_F{right}__held_F{left}/normal_rows.npz")
    rows = concatenate([load_arrays(path) for path in paths])
    expected = sorted(folds[left]+folds[right])
    if sorted(int(sid) for sid in np.unique(rows["scenario"])) != expected:
        raise ValueError("AR outer fit scope mismatch")
    return rows, [str(path) for path in paths], expected


def _verify_completed_outer(folder, held, expected_fit):
    record = read_json(folder/"completed.json")
    scope = read_json(folder/"fit_scope.json")
    if (record["held_scenarios"] != held or scope["held_scenarios"] != held
            or scope["fit_scenarios"] != expected_fit
            or sha(folder/"fit_scope.json") != record["fit_scope_sha256"]
            or sha(folder/"latent_ar.joblib") != record["model_sha256"]
            or sha(folder/"held_predictions.npz") != record["predictions_sha256"]):
        raise ValueError("Completed latent-AR outer artifact mismatch")
    return record


def _verify_completed_run(output, signature):
    summary = read_json(output/"summary.json")
    manifest = summary["artifact_manifest"]
    if sha(output/"oof_predictions.npz") != manifest["oof_predictions_sha256"]:
        raise ValueError("Completed latent-AR OOF hash mismatch")
    for outer, held in enumerate(signature["folds"]):
        expected_fit = sorted(set(signature["splits"]["train"])-set(held))
        _verify_completed_outer(output/f"outer_{outer}", held, expected_fit)
    if any(summary[key] for key in
           ("calibration_evaluated", "validation_evaluated", "test_evaluated")):
        raise ValueError("Completed latent-AR run evaluated a forbidden split")


def run(output_dir, preflight_only=False):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/"campaign.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Latent AR Stage B is already running") from error

    stage_summary, stage_audit, stage_signature = _verify_stage_a()
    signature = {"sources": {path: sha(Path(path)) for path in SOURCE},
        "protocol": PROTOCOL, "stage_a_signature_sha": sha(STAGE_A/"signature.json"),
        "stage_a_summary_sha": sha(STAGE_A/"summary.json"),
        "stage_a_audit_sha": sha(STAGE_A/"independent_audit.json"),
        "stage_a_oof_sha": stage_audit["oof_predictions_sha256"],
        "control_oof_sha": stage_signature["control_oof_sha"],
        "outer_references": stage_signature["outer_references"],
        "generate_config_sha": stage_signature["generate_config_sha"],
        "splits": stage_signature["splits"], "folds": stage_signature["folds"]}
    if (output/"signature.json").exists() and read_json(output/"signature.json") != signature:
        raise ValueError("Latent AR source/input signature changed; refusing silent resume")
    write_json(output/"signature.json", signature)
    write_json(output/"protocol.json", PROTOCOL)
    write_json(output/"outer_folds.json", signature["folds"])
    if preflight_only:
        print("Latent AR Stage B preflight complete; no data extracted and no model fitted", flush=True)
        return
    if (output/"summary.json").exists():
        _verify_completed_run(output, signature)
        print("Latent AR Stage B already complete; artifacts verified; no refit", flush=True)
        return

    started = time.monotonic()

    def status(phase, **details):
        row = {"phase": phase, "elapsed_seconds": time.monotonic()-started,
            "calibration_evaluated": False, "validation_evaluated": False,
            "test_evaluated": False, **details}
        write_json(output/"status.json", row)
        print(phase, details, flush=True)

    status("preflight verified; TRAIN-only AR fits starting")
    splits, folds = signature["splits"], signature["folds"]
    data = TrainOnlyCampaign(DATA, splits)
    all_events = read_json(DATA/"events.json")
    train = set(splits["train"])
    events = [{**event, "_event_id": event_id} for event_id, event in enumerate(all_events)
              if event["scenario_id"] in train]
    change_filter = ConditionalChangeFilter(CHANGE_CONFIG)
    all_parts, all_scores = [], {"B": [], "A": []}
    fitted_models = []

    for outer, held in enumerate(folds):
        folder = output/f"outer_{outer}"
        folder.mkdir(exist_ok=True)
        completed = folder/"completed.json"
        expected_fit = sorted(set(splits["train"])-set(held))
        if not completed.exists():
            status("fitting AR bundle", outer_fold=outer, held_scenarios=held)
            rows, fit_paths, fit_scenarios = _fit_rows_for_outer(outer, folds)
            if fit_scenarios != expected_fit:
                raise ValueError("Unexpected latent-AR fit scenarios")
            model = LatentARNormalError(AR_CONFIG).fit(rows["residual"],
                scenario=rows["scenario"], timestep=rows["timestep"], node=rows["node"])
            atomic_joblib(model, folder/"latent_ar.joblib")
            reloaded = joblib.load(folder/"latent_ar.joblib")
            if reloaded.metadata() != model.metadata():
                raise ValueError("Reloaded latent-AR model metadata differs")

            base_held = load_arrays(STAGE_A/f"outer_{outer}/held_predictions.npz")
            base_scores_b = base_held.pop("scores_B")
            for key in ("scores_C", "scores_S", "scores_M"):
                base_held.pop(key)
            outer_reference = joblib.load(BASE/f"fold_{outer}/reference.joblib")
            parts, score_b_parts, score_a_parts = [], [], []
            for sid in held:
                a = data.scenario(sid)
                prediction, _, _ = outer_reference.predict_details(a["values"], a["mask"])
                raw_residual = np.where(a["mask"], a["values"]-prediction, 0.0)
                mean_a, scale_a = model.predict_sequence(raw_residual, a["mask"],
                    timestep=a["timestep"], node=np.arange(a["values"].shape[1]))
                state, names = change_filter.transform(a["values"], a["mask"], prediction,
                    mean_a, scale_a, outer_reference.noise_scale_)
                if names[:2] != ["conditional_drift_probability", "conditional_noise_probability"]:
                    raise ValueError("Conditional change score schema changed")
                endpoint = a["mask"][15:]
                meta = observation_metadata({**a, "scenario_id": sid}, events)
                base_loc = base_held["scenario"] == sid
                for key in ("labels", "families", "event", "timestep", "node", "early"):
                    np.testing.assert_array_equal(meta[key], base_held[key][base_loc])
                error = (a["values"][15:]-prediction[15:])[endpoint]
                np.testing.assert_array_equal(error, base_held["error"][base_loc])
                part = {**meta, "scenario": np.full(int(endpoint.sum()), sid, dtype=np.int16),
                    "outer_fold": np.full(int(endpoint.sum()), outer, dtype=np.int8),
                    "error": error, "mean_B": base_held["mean_B"][base_loc],
                    "scale_B": base_held["scale_B"][base_loc],
                    "mean_A": mean_a[15:][endpoint], "scale_A": scale_a[15:][endpoint]}
                parts.append(part)
                score_b_parts.append(base_scores_b[base_loc])
                score_a_parts.append(state[endpoint][:, :2])
            held_arrays = concatenate(parts)
            held_scores_b, held_scores_a = np.concatenate(score_b_parts), np.concatenate(score_a_parts)
            atomic_npz(folder/"held_predictions.npz", **held_arrays,
                       scores_B=held_scores_b, scores_A=held_scores_a)
            write_json(folder/"fit_scope.json", {"outer_fold": outer,
                "held_scenarios": held, "fit_scenarios": fit_scenarios,
                "fit_directed_caches": fit_paths, "family_zero_only": True,
                "consecutive_pairs": model.fit_pair_count_, "phi": model.phi_})
            write_json(completed, {"held_scenarios": held,
                "model_sha256": sha(folder/"latent_ar.joblib"),
                "predictions_sha256": sha(folder/"held_predictions.npz"),
                "fit_scope_sha256": sha(folder/"fit_scope.json"),
                "reload_metadata_exact": True})
        record = _verify_completed_outer(folder, held, expected_fit)
        held_arrays = load_arrays(folder/"held_predictions.npz")
        all_scores["B"].append(held_arrays.pop("scores_B"))
        all_scores["A"].append(held_arrays.pop("scores_A"))
        all_parts.append(held_arrays)
        fitted_models.append(joblib.load(folder/"latent_ar.joblib").metadata())

    arrays = concatenate(all_parts)
    scores = {arm: np.concatenate(parts) for arm, parts in all_scores.items()}
    stage_oof = load_arrays(STAGE_A/"oof_predictions.npz")
    for key in ("labels", "families", "event", "timestep", "node", "early",
                "scenario", "outer_fold", "error", "mean_B", "scale_B"):
        np.testing.assert_array_equal(arrays[key], stage_oof[key])
    np.testing.assert_array_equal(scores["B"], stage_oof["scores_B"])
    control = load_arrays(CONTROL/"trial_0000/oof_predictions.npz")
    for key in ("labels", "families", "scenario"):
        np.testing.assert_array_equal(arrays[key], control[key])

    normal = _normal_diagnostics(arrays, events)
    lag = {arm: _lag_one(arrays, arm) for arm in ("B", "A")}
    normal_decision = _normal_gate(normal, lag)
    diagnostics = {arm: expert_diagnostics(arrays, value, events)
                   for arm, value in scores.items()}
    diagnostics["local_control"] = expert_diagnostics(arrays, control["experts"][:, 3:5], events)
    gates, survivors = {}, []
    for family in ("drift", "noise"):
        decision = gate(diagnostics["A"][family], diagnostics["B"][family], True)
        control_fraction = (diagnostics["A"][family]["macro_ap"]
                            / diagnostics["local_control"][family]["macro_ap"])
        decision["checks"]["minimum_control_macro_ap_fraction"] = control_fraction >= .75
        decision["checks"]["global_normal_gate"] = normal_decision["passed"]
        decision["passed"] = all(decision["checks"].values())
        decision["control_macro_ap_fraction"] = control_fraction
        decision["comparison"] = "latent AR A versus coherent iid B; global normal gate required"
        gates[family] = decision
        if decision["passed"]:
            survivors.append(family)
    secondary = {arm: secondary_curve(arrays, value, events, .005)
                 for arm, value in scores.items()}
    atomic_npz(output/"oof_predictions.npz", **arrays,
               **{f"scores_{arm}": value for arm, value in scores.items()})
    for path, expected in signature["sources"].items():
        if sha(Path(path)) != expected:
            raise ValueError("Latent AR source changed during execution")
    manifest = {"oof_predictions_sha256": sha(output/"oof_predictions.npz"),
        "outer": {str(index): read_json(output/f"outer_{index}/completed.json")
                  for index in range(3)}}
    result = {"scope": "TRAIN-only nested causal latent AR screen", "protocol": PROTOCOL,
        "normal_diagnostics": normal, "lag_one_diagnostics": lag,
        "normal_gate": normal_decision, "evidence_diagnostics": diagnostics,
        "evidence_gates": gates, "secondary_fpr_005": secondary,
        "survivors": survivors, "fitted_models": fitted_models,
        "artifact_manifest": manifest, "endpoints": len(arrays["labels"]),
        "accessed_scenarios": sorted(data.accessed), "ar_bundle_fit_calls": 3,
        "weak_head_fit_calls": 0, "optuna_trials": 0,
        "calibration_evaluated": False, "validation_evaluated": False,
        "test_evaluated": False, "elapsed_seconds": time.monotonic()-started,
        "next_action": ("eligible_for_separate_specialist_screen" if survivors
                        else "stop_no_latent_ar_candidate_passed")}
    write_json(output/"summary.json", result)
    write_json(output/"verification.json", {"stage_a_oof_exact": True,
        "model_reload_metadata_exact": True, "artifact_hashes_checked": True,
        "source_hashes_match": True,
        "accessed_train_scenarios_only": set(data.accessed) <= set(splits["train"]),
        "calibration_evaluated": False, "validation_evaluated": False,
        "test_evaluated": False})
    status("Latent AR Stage B complete", survivors=survivors,
           next_action=result["next_action"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="runs/operational/latent_ar_stage_b_v1")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    run(args.output_dir, args.preflight_only)
