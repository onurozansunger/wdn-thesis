"""Frozen expanded-TRAIN source-held screen for daily seasonal experts."""
from __future__ import annotations

import argparse
import fcntl
import gc
import json
from pathlib import Path
import time

import joblib
import numpy as np

from wdn.expanded_train_data import ExpandedTrainData
from wdn.models.seasonal_family import SeasonalFamilyExperts
from wdn.run_expert_redesign import load_arrays, read_json, sha, write_json
from wdn.screen_family_specific import _best_f1
from wdn.screen_recovery import expert_diagnostics
from wdn.seasonal_pressure_features import endpoint_seasonal_features


BASE = Path("runs/operational/expanded_train_weak_experts_v1")
TUNING = Path("runs/operational/expanded_train_optuna_v1")
PROTOCOL = Path("thesis_v2/SEASONAL_FAMILY_EXPERT_PROTOCOL.md")
ORIGINAL = Path("data/thesis_v2/operational_modena_seed811")
EXPANSIONS = tuple(Path(f"data/thesis_v2/operational_train_expansion_seed{seed}")
                   for seed in (1811, 2811, 3811))
SPLITS = Path("runs/operational/blind_reference_probe_rank16/splits.json")
META_KEYS = ("labels", "families", "event", "scenario", "source", "timestep", "node", "early")


def atomic_npz(path, **arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def logit_blend(first, second, first_weight):
    def logit(value):
        value = np.clip(value, 1e-6, 1 - 1e-6)
        return np.log(value / (1 - value))
    linear = first_weight * logit(first) + (1 - first_weight) * logit(second)
    return 1 / (1 + np.exp(-np.clip(linear, -30, 30)))


def oracle(scores, arrays, family_id):
    selected = arrays["families"] == family_id
    return _best_f1(scores[selected], arrays["labels"][selected])


def run(output_dir):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    lock = (output / "campaign.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Seasonal family screen is already running") from error
    signature = {"protocol_sha256": sha(PROTOCOL),
        "source_sha256": {path: sha(Path(path)) for path in (
            "src/wdn/seasonal_pressure_features.py", "src/wdn/models/seasonal_family.py",
            "thesis_v2/experiments/screen_seasonal_family_experts.py")},
        "base_summary_sha256": sha(BASE / "summary.json"),
        "base_signature_sha256": sha(BASE / "signature.json"),
        "tuned_trial_2_summary_sha256": sha(TUNING / "trial_0002/summary.json"),
        "splits_sha256": sha(SPLITS), "drift_seasonal_weight": .90,
        "noise_fast_weight": .95, "calibration_evaluated": False,
        "validation_evaluated": False, "test_evaluated": False}
    if (output / "signature.json").exists() and read_json(output / "signature.json") != signature:
        raise ValueError("Seasonal screen source/input signature changed")
    write_json(output / "signature.json", signature)
    if (output / "summary.json").exists():
        print("Seasonal family screen already complete; no refit", flush=True); return
    started = time.monotonic()

    def status(phase, **details):
        write_json(output / "status.json", {"phase": phase,
            "elapsed_seconds": time.monotonic() - started,
            "calibration_evaluated": False, "validation_evaluated": False,
            "test_evaluated": False, **details})
        print(phase, details, flush=True)

    splits = read_json(SPLITS)
    data = ExpandedTrainData(ORIGINAL, EXPANSIONS, splits)
    base_names = read_json(BASE / "feature_names.json")
    seasonal_names = None
    fold_scores, held_parts = [], []
    for outer, held_ids in enumerate(data.source_folds):
        folder = output / f"fold_{outer}"; folder.mkdir(exist_ok=True)
        train = load_arrays(BASE / f"fold_{outer}/features_train.npz")
        held = load_arrays(BASE / f"fold_{outer}/features_held_out.npz")
        held_parts.append({key: held[key] for key in META_KEYS})
        for split, arrays in (("train", train), ("held_out", held)):
            path = folder / f"seasonal_{split}.npz"
            if not path.exists():
                status("building causal daily seasonal features", outer=outer, split=split)
                features, found_names = endpoint_seasonal_features(arrays, data.scenario)
                atomic_npz(path, X=features)
            else:
                found_names = list(read_json(output / "seasonal_feature_names.json")) \
                    if (output / "seasonal_feature_names.json").exists() else None
            if found_names is not None:
                if seasonal_names is not None and seasonal_names != found_names:
                    raise ValueError("Seasonal feature schemas differ")
                seasonal_names = found_names
        if seasonal_names is None:
            from wdn.seasonal_pressure_features import NAMES
            seasonal_names = list(NAMES)
        write_json(output / "seasonal_feature_names.json", seasonal_names)
        train_extended = {**train, "X": np.column_stack((train["X"],
            load_arrays(folder / "seasonal_train.npz")["X"]))}
        held_X = np.column_stack((held["X"], load_arrays(
            folder / "seasonal_held_out.npz")["X"]))
        prediction_path = folder / "predictions.npz"
        if not prediction_path.exists():
            status("fitting frozen seasonal family experts", outer=outer)
            model = SeasonalFamilyExperts(base_names + seasonal_names, seed=4100 + outer).fit(train_extended)
            predictions = model.predict(held_X)
            joblib.dump(model, folder / "bundle.joblib")
            atomic_npz(prediction_path, **predictions)
        fold_scores.append(load_arrays(prediction_path))
        del train, held, train_extended, held_X
        gc.collect()
    oof = {key: np.concatenate([part[key] for part in held_parts]) for key in META_KEYS}
    seasonal_drift = np.concatenate([part["drift"] for part in fold_scores])
    noise_full = np.concatenate([part["noise_full"] for part in fold_scores])
    noise_fast = np.concatenate([part["noise_fast"] for part in fold_scores])
    tuned_drift = np.concatenate([load_arrays(
        TUNING / f"trial_0002/fold_{outer}_predictions.npz")["scores"][:, 0]
        for outer in range(4)])
    scores = np.column_stack((logit_blend(seasonal_drift, tuned_drift, .90),
                              logit_blend(noise_fast, noise_full, .95)))
    diagnostics = expert_diagnostics(oof, scores, data.events)
    family_f1 = {"drift": oracle(scores[:, 0], oof, 3),
                 "noise": oracle(scores[:, 1], oof, 4)}
    baseline = read_json(BASE / "summary.json")["candidates"]["mechanism"]
    gates = {}
    for family in ("drift", "noise"):
        row, control = diagnostics[family], baseline["diagnostics"][family]
        f1, control_f1 = family_f1[family], baseline["family_oracle_f1"][family]
        gains = {sid: row["by_scenario_ap"][sid] - value
                 for sid, value in control["by_scenario_ap"].items()}
        checks = {"macro_ap_improves": row["macro_ap"] > control["macro_ap"],
            "oracle_f1_improves": f1["f1"] > control_f1["f1"],
            "oracle_f1_at_least_0_70": f1["f1"] >= .70,
            "clean_fpr_within_allowance": row["curve_clean_fpr"] <= control["curve_clean_fpr"] + .00025,
            "post_event_fp_within_allowance": row["curve_post_event_fp"] <= control["curve_post_event_fp"] + 5,
            "worst_ap_preserved": row["worst_ap"] >= control["worst_ap"] - .03}
        gates[family] = {"passed": bool(all(checks.values())), "checks": checks,
            "macro_ap_gain": row["macro_ap"] - control["macro_ap"],
            "oracle_f1_gain": f1["f1"] - control_f1["f1"],
            "improved_scenarios": sum(value > 0 for value in gains.values()),
            "scenario_count": len(gains)}
    atomic_npz(output / "oof_predictions.npz", scores=scores,
               seasonal_drift=seasonal_drift, noise_full=noise_full,
               noise_fast=noise_fast, tuned_drift=tuned_drift, **oof)
    summary = {"status": "completed",
        "scope": "expanded TRAIN-only four-generator-held OOF; development-selected architecture",
        "diagnostics": diagnostics, "family_oracle_f1": family_f1,
        "promotion_gates": gates,
        "both_families_promoted": bool(all(gate["passed"] for gate in gates.values())),
        "next_action": ("independent_audit_then_full_train_and_calibration"
            if all(gate["passed"] for gate in gates.values()) else "stop_before_calibration"),
        "selection_limit": "seasonal architecture and blends were selected after expanded TRAIN OOF diagnostics",
        "calibration_evaluated": False, "validation_evaluated": False,
        "test_evaluated": False, "oof_sha256": sha(output / "oof_predictions.npz"),
        "elapsed_seconds": time.monotonic() - started}
    write_json(output / "summary.json", summary)
    status("seasonal family OOF screen complete", gates=gates,
           next_action=summary["next_action"])
    print(json.dumps({"family_oracle_f1": family_f1, "gates": gates}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/operational/seasonal_family_experts_v1")
    run(parser.parse_args().output_dir)
