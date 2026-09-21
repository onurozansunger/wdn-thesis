"""Exactly four Optuna trials on frozen expanded-TRAIN source-held features."""
from __future__ import annotations

import argparse
import fcntl
import gc
import json
from pathlib import Path
import time

import joblib
import numpy as np
import optuna

from wdn.models.tuned_family_tree import TunedFamilyTreeConfig, TunedFamilyTrees
from wdn.run_expert_redesign import load_arrays, read_json, sha, write_json
from wdn.screen_family_specific import _best_f1
from wdn.screen_recovery import expert_diagnostics


BASE = Path("runs/operational/expanded_train_weak_experts_v1")
PROTOCOL = Path("thesis_v2/EXPANDED_TRAIN_OPTUNA_PROTOCOL.md")
TRIALS = (
    {"max_leaf_nodes": 15, "min_samples_leaf": 30, "learning_rate": .08,
     "l2_regularization": 10., "max_iter": 120, "noise_gate": 0.},
    {"max_leaf_nodes": 31, "min_samples_leaf": 20, "learning_rate": .05,
     "l2_regularization": 20., "max_iter": 180, "noise_gate": .5},
    {"max_leaf_nodes": 23, "min_samples_leaf": 50, "learning_rate": .10,
     "l2_regularization": 5., "max_iter": 150, "noise_gate": .5},
    {"max_leaf_nodes": 9, "min_samples_leaf": 15, "learning_rate": .04,
     "l2_regularization": 30., "max_iter": 220, "noise_gate": .5},
)
ARRAY_KEYS = ("labels", "families", "event", "scenario", "source", "timestep", "node", "early")


def atomic_npz(path, **arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def oracle(scores, arrays, family_id):
    selected = arrays["families"] == family_id
    return _best_f1(scores[selected], arrays["labels"][selected])


def trial_report(scores, arrays, events, baseline_noise_post):
    diagnostics = expert_diagnostics(arrays, scores, events)
    family_f1 = {"drift": oracle(scores[:, 0], arrays, 3),
                 "noise": oracle(scores[:, 1], arrays, 4)}
    worst_f1 = min(row["f1"] for row in family_f1.values())
    mean_ap = np.mean([diagnostics[name]["macro_ap"] for name in ("drift", "noise")])
    excess_post = max(0, diagnostics["noise"]["curve_post_event_fp"] - baseline_noise_post - 5)
    value = float(worst_f1 + .10 * mean_ap - .001 * excess_post)
    return {"diagnostics": diagnostics, "family_oracle_f1": family_f1,
            "objective": {"value": value, "worst_family_f1": worst_f1,
                "mean_macro_ap": float(mean_ap), "noise_excess_post_event_fp": int(excess_post)}}


def run(output_dir):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    lock = (output / "campaign.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Expanded TRAIN Optuna tuning is already running") from error
    signature = json.loads(json.dumps({
        "protocol_sha256": sha(PROTOCOL), "base_signature_sha256": sha(BASE / "signature.json"),
        "base_summary_sha256": sha(BASE / "summary.json"),
        "base_oof_sha256": sha(BASE / "oof_predictions.npz"),
        "feature_names_sha256": sha(BASE / "feature_names.json"),
        "source_sha256": {
            "src/wdn/models/tuned_family_tree.py": sha(Path("src/wdn/models/tuned_family_tree.py")),
            "thesis_v2/experiments/tune_expanded_train_weak_experts.py": sha(Path(__file__)),
        },
        "trial_budget": 4, "fixed_trials": TRIALS,
        "calibration_evaluated": False, "validation_evaluated": False, "test_evaluated": False,
    }))
    if (output / "signature.json").exists() and read_json(output / "signature.json") != signature:
        raise ValueError("Expanded TRAIN tuning signature changed")
    write_json(output / "signature.json", signature)
    if (output / "summary.json").exists():
        print("Expanded TRAIN tuning already complete; no refit", flush=True); return
    started = time.monotonic()

    def status(phase, **details):
        write_json(output / "status.json", {"phase": phase,
            "elapsed_seconds": time.monotonic() - started,
            "calibration_evaluated": False, "validation_evaluated": False,
            "test_evaluated": False, **details})
        print(phase, details, flush=True)

    events = read_json(BASE / "events_train_global.json")
    names = read_json(BASE / "feature_names.json")
    base_summary = read_json(BASE / "summary.json")
    baseline = base_summary["candidates"]["mechanism"]
    baseline_noise_post = baseline["diagnostics"]["noise"]["curve_post_event_fp"]
    held_parts = [load_arrays(BASE / f"fold_{outer}/features_held_out.npz") for outer in range(4)]
    oof = {key: np.concatenate([part[key] for part in held_parts]) for key in ARRAY_KEYS}

    storage = f"sqlite:///{output / 'study.sqlite3'}"
    study = optuna.create_study(study_name="expanded_train_family_trees_v1", storage=storage,
        load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=2421, n_startup_trials=4))
    if not study.trials:
        for parameters in TRIALS:
            study.enqueue_trial(parameters)

    def objective(trial):
        parameters = {
            "max_leaf_nodes": trial.suggest_categorical("max_leaf_nodes", [9, 15, 23, 31]),
            "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [15, 20, 30, 50]),
            "learning_rate": trial.suggest_categorical("learning_rate", [.04, .05, .08, .10]),
            "l2_regularization": trial.suggest_categorical("l2_regularization", [5., 10., 20., 30.]),
            "max_iter": trial.suggest_categorical("max_iter", [120, 150, 180, 220]),
            "noise_gate": trial.suggest_categorical("noise_gate", [0., .5]),
        }
        folder = output / f"trial_{trial.number:04d}"; folder.mkdir(exist_ok=True)
        fold_scores = []
        for outer in range(4):
            prediction_path = folder / f"fold_{outer}_predictions.npz"
            if prediction_path.exists():
                fold_scores.append(load_arrays(prediction_path)["scores"]); continue
            status("fitting bounded Optuna family trees", trial=trial.number, outer=outer,
                   params=parameters)
            train = load_arrays(BASE / f"fold_{outer}/features_train.npz")
            held = load_arrays(BASE / f"fold_{outer}/features_held_out.npz")
            config = TunedFamilyTreeConfig(**{key: parameters[key] for key in
                ("max_leaf_nodes", "min_samples_leaf", "learning_rate",
                 "l2_regularization", "max_iter")}, seed=2421 + 100 * trial.number + outer)
            model = TunedFamilyTrees(names, config).fit(train)
            scores = model.predict(held["X"])
            if parameters["noise_gate"]:
                current = np.maximum(held["X"][:, names.index("dynamic_abs_innovation")], 0)
                scores[:, 1] *= np.clip(current / parameters["noise_gate"], 0, 1)
            joblib.dump(model, folder / f"fold_{outer}_model.joblib")
            atomic_npz(prediction_path, scores=scores)
            fold_scores.append(scores)
            del train, held, model, scores
            gc.collect()
        scores = np.concatenate(fold_scores)
        report = trial_report(scores, oof, events, baseline_noise_post)
        report["trial"] = trial.number; report["params"] = parameters
        trial.set_user_attr("report", report)
        write_json(folder / "summary.json", report)
        atomic_npz(folder / "oof_predictions.npz", scores=scores, **oof)
        return report["objective"]["value"]

    consumed = sum(trial.state != optuna.trial.TrialState.WAITING for trial in study.trials)
    study.optimize(objective, n_trials=max(0, 4 - consumed))
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) != 4:
        raise RuntimeError("Exactly four completed Optuna trials are required")
    winner = max(completed, key=lambda trial: trial.value)
    reports = {str(trial.number): trial.user_attrs["report"] for trial in completed}
    candidate = reports[str(winner.number)]
    gates = {}
    for family in ("drift", "noise"):
        row, control = candidate["diagnostics"][family], baseline["diagnostics"][family]
        f1, control_f1 = candidate["family_oracle_f1"][family], baseline["family_oracle_f1"][family]
        checks = {
            "macro_ap_improves": row["macro_ap"] > control["macro_ap"],
            "oracle_f1_improves": f1["f1"] > control_f1["f1"],
            "clean_fpr_not_materially_higher": row["curve_clean_fpr"] <= control["curve_clean_fpr"] + .00025,
            "post_event_fp_within_allowance": row["curve_post_event_fp"] <= control["curve_post_event_fp"] + 5,
        }
        gates[family] = {"passed": bool(all(checks.values())), "checks": checks,
            "target_0_70_reached_in_train_oof_oracle": f1["f1"] >= .70,
            "target_0_80_reached_in_train_oof_oracle": f1["f1"] >= .80,
            "f1_gain": f1["f1"] - control_f1["f1"],
            "macro_ap_gain": row["macro_ap"] - control["macro_ap"]}
    summary = {"status": "completed", "scope": "expanded TRAIN-only generator-held OOF",
        "completed_optuna_trials": len(completed), "winner": winner.number,
        "winner_params": winner.params, "winner_report": candidate, "trials": reports,
        "promotion_gates": gates,
        "both_families_promoted": bool(all(gate["passed"] for gate in gates.values())),
        "next_action": ("freeze_full_train_fit_then_use_existing_calibration_once"
            if all(gate["passed"] for gate in gates.values()) else "stop_before_calibration"),
        "selection_limit": "winner selected and reported on the same TRAIN OOF study",
        "calibration_evaluated": False, "validation_evaluated": False,
        "test_evaluated": False, "elapsed_seconds": time.monotonic() - started}
    write_json(output / "summary.json", summary)
    status("four-trial expanded TRAIN Optuna study complete", winner=winner.number,
           gates=gates, next_action=summary["next_action"])
    print(json.dumps({"winner": winner.number, "params": winner.params,
                      "gates": gates}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/operational/expanded_train_optuna_v1")
    run(parser.parse_args().output_dir)
