"""Independent artifact/metric audit for expanded-TRAIN weak-family runs."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import optuna

from wdn.expanded_train_data import ALLOWED_CONFIG_CHANGES, scenario_uid
from wdn.run_expert_redesign import load_arrays, read_json, sha, write_json
from wdn.screen_family_specific import _best_f1
from wdn.screen_recovery import expert_diagnostics


BASE = Path("runs/operational/expanded_train_weak_experts_v1")
TUNING = Path("runs/operational/expanded_train_optuna_v1")
OUTPUT = Path("thesis_v2/outputs/expanded_train_weak_experts_results.json")
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
    base, tuning = read_json(BASE / "summary.json"), read_json(TUNING / "summary.json")
    base_signature, tuning_signature = read_json(BASE / "signature.json"), read_json(TUNING / "signature.json")
    events, names = read_json(BASE / "events_train_global.json"), read_json(BASE / "feature_names.json")
    held = [load_arrays(BASE / f"fold_{outer}/features_held_out.npz") for outer in range(4)]
    oof = {key: np.concatenate([part[key] for part in held]) for key in META_KEYS}
    checks = {}
    checks["no_later_split_evaluated"] = not any(base[key] or tuning[key] for key in
        ("calibration_evaluated", "validation_evaluated", "test_evaluated"))
    checks["base_source_hashes_match"] = all(sha(Path(path)) == digest
        for path, digest in base_signature["source_sha256"].items())
    checks["tuning_source_hashes_match"] = all(sha(Path(path)) == digest
        for path, digest in tuning_signature["source_sha256"].items())
    checks["data_hashes_match"] = all(sha(Path(directory) / name) == digest
        for directory, files in base_signature["data"].items() for name, digest in files.items())
    checks["tuning_inputs_match"] = (
        tuning_signature["base_signature_sha256"] == sha(BASE / "signature.json")
        and tuning_signature["base_summary_sha256"] == sha(BASE / "summary.json")
        and tuning_signature["base_oof_sha256"] == sha(BASE / "oof_predictions.npz"))
    audit = read_json(BASE / "data_audit.json")
    checks["missing_probabilities_fixed_at_half"] = all(
        row["pressure_missing_probability"] == row["flow_missing_probability"] == .5
        for row in audit["config_audit"].values())
    checks["only_identity_and_sample_count_changed"] = all(
        not row["changed_fields"] or set(row["changed_fields"]) == ALLOWED_CONFIG_CHANGES
        for row in audit["config_audit"].values())
    checks["all_expansion_scenarios_retained"] = (
        audit["all_expansion_scenarios_retained"]
        and all(audit["config_audit"][str(seed)]["scenario_count_used"] == 16
                for seed in (1811, 2811, 3811)))

    folds = base["folds"]
    checks["source_folds_are_disjoint_and_complete"] = (
        len(folds) == 4 and len(set().union(*map(set, folds))) == sum(map(len, folds)) == 62)
    scope_ok = True
    for outer, fold in enumerate(folds):
        train = load_arrays(BASE / f"fold_{outer}/features_train.npz")
        held_arrays = held[outer]
        scope = read_json(BASE / f"fold_{outer}/scope.json")
        scope_ok &= (set(train["scenario"]).isdisjoint(set(held_arrays["scenario"]))
                     and set(held_arrays["scenario"]) == set(fold)
                     and set(scope["held_sources"]) == set(held_arrays["source"])
                     and set(scope["training_sources"]).isdisjoint(scope["held_sources"]))
    checks["feature_fold_scope_matches"] = bool(scope_ok)
    protected = read_json(Path("runs/operational/blind_reference_probe_rank16/splits.json"))
    forbidden = {scenario_uid(811, sid) for key in ("calibration", "validation", "test")
                 for sid in protected[key]}
    checks["no_protected_original_scenario_in_oof"] = set(oof["scenario"]).isdisjoint(forbidden)

    classical = [load_arrays(BASE / f"fold_{outer}/classical_predictions.npz") for outer in range(4)]
    family = np.concatenate([part["family_mean"] for part in classical])
    short = np.concatenate([load_arrays(BASE / f"fold_{outer}/causal_short_predictions.npz")["scores"]
                            for outer in range(4)])
    long = np.concatenate([load_arrays(BASE / f"fold_{outer}/causal_long_predictions.npz")["scores"]
                           for outer in range(4)])
    selected_scores = np.column_stack((logit_blend(family, short, .75)[:, 0],
                                       logit_blend(family, long, .75)[:, 1]))
    saved = load_arrays(BASE / "oof_predictions.npz")
    np.testing.assert_array_equal(selected_scores, saved["scores"])
    for key in META_KEYS:
        np.testing.assert_array_equal(oof[key], saved[key])
    checks["base_selected_oof_reproduced_exact"] = True
    base_metrics_ok = True
    for column, family_name, family_id in ((0, "drift", 3), (1, "noise", 4)):
        name = base["selection"][family_name]
        score_pair = logit_blend(family, short if "short" in name else long, .75)
        diagnostics = expert_diagnostics(oof, score_pair, events)[family_name]
        base_metrics_ok &= diagnostics == base["candidates"][name]["diagnostics"][family_name]
        base_metrics_ok &= oracle(score_pair[:, column], oof, family_id) == (
            base["candidates"][name]["family_oracle_f1"][family_name])
    checks["base_saved_metrics_reproduced_exact"] = bool(base_metrics_ok)

    study = optuna.load_study(study_name="expanded_train_family_trees_v1",
        storage=f"sqlite:///{TUNING / 'study.sqlite3'}")
    complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    checks["exactly_four_completed_optuna_trials"] = len(complete) == 4 == tuning["completed_optuna_trials"]
    winner = int(tuning["winner"])
    winner_scores = np.concatenate([load_arrays(
        TUNING / f"trial_{winner:04d}/fold_{outer}_predictions.npz")["scores"] for outer in range(4)])
    report = tuning["winner_report"]
    diagnostics = expert_diagnostics(oof, winner_scores, events)
    tuning_metrics_ok = diagnostics == report["diagnostics"]
    for column, family_name, family_id in ((0, "drift", 3), (1, "noise", 4)):
        tuning_metrics_ok &= oracle(winner_scores[:, column], oof, family_id) == report["family_oracle_f1"][family_name]
    checks["tuning_saved_metrics_reproduced_exact"] = bool(tuning_metrics_ok)
    reload_ok = True
    gate = float(tuning["winner_params"]["noise_gate"])
    for outer in range(4):
        held_arrays = held[outer]
        model = joblib.load(TUNING / f"trial_{winner:04d}/fold_{outer}_model.joblib")
        fresh = model.predict(held_arrays["X"])
        if gate:
            current = np.maximum(held_arrays["X"][:, names.index("dynamic_abs_innovation")], 0)
            fresh[:, 1] *= np.clip(current / gate, 0, 1)
        expected = load_arrays(TUNING / f"trial_{winner:04d}/fold_{outer}_predictions.npz")["scores"]
        reload_ok &= np.array_equal(fresh, expected)
    checks["winner_models_reload_predictions_exact"] = bool(reload_ok)
    checks["base_oof_hash_matches"] = base["oof_predictions_sha256"] == sha(BASE / "oof_predictions.npz")
    result = {"all_checks_pass": bool(all(checks.values())), **checks,
        "base_summary": base, "optuna_summary": tuning,
        "claim_limits": {"expanded_data_is_train_only": True,
            "selection_and_scores_share_train_oof": True,
            "calibration_evaluated": False, "validation_evaluated": False,
            "test_evaluated": False}}
    write_json(BASE / "independent_audit.json", result)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT, result)
    print(json.dumps({key: value for key, value in result.items()
                      if isinstance(value, bool)}, indent=2))


if __name__ == "__main__":
    run()
