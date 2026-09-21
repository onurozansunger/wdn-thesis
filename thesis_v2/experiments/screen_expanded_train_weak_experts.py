"""Multi-seed TRAIN-only OOF screen for strict-online drift/noise experts."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import gc
import json
from pathlib import Path
import time

import joblib
import numpy as np

from wdn.expanded_train_data import ExpandedTrainData
from wdn.family_state_features import family_state_features
from wdn.models.causal_sequence import (
    CausalSequenceConfig, CausalSequenceExpert, causal_row_ids)
from wdn.models.family_specific import FamilySpecificExpert
from wdn.models.residual_hybrid import ResidualHybrid
from wdn.run_expert_redesign import load_arrays, read_json, sha, write_json
from wdn.screen_family_specific import _best_f1
from wdn.screen_recovery import expert_diagnostics
from wdn.train_weak_families import add_early


ORIGINAL = Path("data/thesis_v2/operational_modena_seed811")
EXPANSIONS = tuple(Path(f"data/thesis_v2/operational_train_expansion_seed{seed}")
                   for seed in (1811, 2811, 3811))
SPLITS = Path("runs/operational/blind_reference_probe_rank16/splits.json")
PROTOCOL = Path("thesis_v2/EXPANDED_TRAIN_WEAK_EXPERT_PROTOCOL.md")
BASE_NAMES = Path("runs/operational/mechanism_redesign_v1/feature_names.json")
ARRAY_KEYS = ("X", "labels", "families", "event", "scenario", "source",
              "timestep", "node", "early")
SEQUENCE_CONFIGS = {
    "causal_short": CausalSequenceConfig(window=8, hidden_size=16,
        embedding_size=20, epochs=8),
    "causal_long": CausalSequenceConfig(window=16, hidden_size=24,
        embedding_size=28, epochs=10),
}


def atomic_npz(path, **arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def concatenate(parts):
    return {key: np.concatenate([part[key] for part in parts]) for key in ARRAY_KEYS}


def family_oracle(scores, arrays, family_id):
    selected = arrays["families"] == family_id
    return _best_f1(scores[selected], arrays["labels"][selected])


def logit_blend(first, second, first_weight):
    def logit(value):
        value = np.clip(value, 1e-6, 1 - 1e-6)
        return np.log(value / (1 - value))
    linear = first_weight * logit(first) + (1 - first_weight) * logit(second)
    return 1 / (1 + np.exp(-np.clip(linear, -30, 30)))


def event_counts(events):
    by_source = {}
    for source in sorted({event["source_seed"] for event in events}):
        by_source[str(source)] = dict(Counter(
            event["family"] for event in events if event["source_seed"] == source))
    return {"total": dict(Counter(event["family"] for event in events)),
            "by_source": by_source}


def prepare_features(output, data, names, status):
    all_scenarios = set(data.scenario_ids)
    for outer, held in enumerate(data.source_folds):
        folder = output / f"fold_{outer}"
        folder.mkdir(exist_ok=True)
        training = sorted(all_scenarios - set(held))
        if set(training) & set(held) or set(training) | set(held) != all_scenarios:
            raise ValueError("Source-held fold partition is invalid")
        reference_path = folder / "reference.joblib"
        if not reference_path.exists():
            status("fitting source-held blind normal reference", outer=outer,
                   training_scenarios=len(training), held_scenarios=len(held))
            data.save_reference(reference_path, training)
        reference = joblib.load(reference_path)
        for split, scenarios in (("train", training), ("held_out", held)):
            path = folder / f"features_{split}.npz"
            if path.exists():
                continue
            status("building expanded TRAIN features", outer=outer, split=split,
                   scenarios=len(scenarios))
            arrays, found_names = data.features(scenarios, reference)
            if found_names != names:
                raise ValueError("Expanded feature schema differs from frozen mechanism schema")
            arrays = add_early(arrays, data.events)
            atomic_npz(path, **arrays)
        write_json(folder / "scope.json", {
            "reference_fit_scenarios": training, "held_scenarios": held,
            "training_sources": sorted(set(data.scenario(uid)["source"] for uid in training)),
            "held_sources": sorted(set(data.scenario(uid)["source"] for uid in held)),
            "test_evaluated": False,
        })


def fit_classical_fold(output, outer, names, status):
    folder = output / f"fold_{outer}"
    prediction_path = folder / "classical_predictions.npz"
    if prediction_path.exists():
        return
    train, held = load_arrays(folder / "features_train.npz"), load_arrays(
        folder / "features_held_out.npz")
    predictions, bundle = {}, {}
    for kind in ("mechanism", "trees"):
        status("fitting expanded classical experts", outer=outer, candidate=kind)
        model = ResidualHybrid(names, kind=kind, C=.1, seed=821 + outer).fit(train)
        predictions[kind] = model.predict(held["X"])[:, 3:5]
        bundle[kind] = model
    status("building causal family-state features", outer=outer)
    train_state, state_names = family_state_features(train, names)
    held_state, found_names = family_state_features(held, names)
    if state_names != found_names:
        raise ValueError("Family-state feature schemas differ")
    train_family = {**train, "X": np.column_stack((train["X"], train_state))}
    held_X = np.column_stack((held["X"], held_state))
    full_names = names + state_names
    heads, family_bundle = [], {}
    for family in ("drift", "noise"):
        status("fitting expanded family-specific expert", outer=outer, family=family)
        model = FamilySpecificExpert(full_names, family, seed=911 + outer).fit(train_family)
        heads.append(model.predict_heads(held_X))
        family_bundle[family] = model
    # (rows, family, fast/persistent)
    heads = np.stack(heads, axis=1)
    predictions["family_fast"] = heads[:, :, 0]
    predictions["family_persistent"] = heads[:, :, 1]
    predictions["family_mean"] = heads.mean(axis=2)
    bundle["family_specific"] = family_bundle
    joblib.dump(bundle, folder / "classical_bundle.joblib")
    atomic_npz(prediction_path, **predictions)
    del train, held, train_state, held_state, train_family, held_X, bundle
    gc.collect()


def fit_sequence_fold(output, outer, names, status):
    folder = output / f"fold_{outer}"
    train, held = None, None
    for candidate, config in SEQUENCE_CONFIGS.items():
        prediction_path = folder / f"{candidate}_predictions.npz"
        if prediction_path.exists():
            continue
        if train is None:
            train = load_arrays(folder / "features_train.npz")
            held = load_arrays(folder / "features_held_out.npz")
        held_rows = causal_row_ids(held, config.window)
        scores, models = [], {}
        for family in ("drift", "noise"):
            status("fitting expanded causal sequence expert", outer=outer,
                   candidate=candidate, family=family)
            model = CausalSequenceExpert(names, family, config,
                seed_offset=100 * outer).fit(train)
            scores.append(model.predict(held, row_ids=held_rows))
            models[family] = model
        scores = np.column_stack(scores)
        joblib.dump(models, folder / f"{candidate}_bundle.joblib")
        atomic_npz(prediction_path, scores=scores)
        del scores, models, held_rows
        gc.collect()
    del train, held
    gc.collect()


def run(output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "campaign.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Expanded TRAIN campaign is already running") from error
    if (output / "summary.json").exists():
        print("Expanded TRAIN campaign already complete; no refit", flush=True)
        return
    splits = read_json(SPLITS)
    signature = {
        "scope": "TRAIN-only four-source held-out weak-family architecture screen",
        "protocol_sha256": sha(PROTOCOL), "splits_sha256": sha(SPLITS),
        "feature_names_sha256": sha(BASE_NAMES),
        "source_sha256": {
            "src/wdn/expanded_train_data.py": sha(Path("src/wdn/expanded_train_data.py")),
            "src/wdn/models/causal_sequence.py": sha(Path("src/wdn/models/causal_sequence.py")),
            "thesis_v2/experiments/screen_expanded_train_weak_experts.py": sha(Path(__file__)),
        },
        "sequence_configs": {key: value.__dict__ for key, value in SEQUENCE_CONFIGS.items()},
        "data": {str(directory): {name: sha(directory / name) for name in
            ("generate_config.yaml", "manifest.json", "events.json", "snapshots.pkl", "corrupted.pkl")}
            for directory in (ORIGINAL,) + EXPANSIONS},
        "calibration_evaluated": False, "validation_evaluated": False,
        "test_evaluated": False,
    }
    if (output / "signature.json").exists() and read_json(output / "signature.json") != signature:
        raise ValueError("Expanded TRAIN source/input signature changed")
    write_json(output / "signature.json", signature)
    started = time.monotonic()

    def status(phase, **details):
        write_json(output / "status.json", {"phase": phase,
            "elapsed_seconds": time.monotonic() - started,
            "calibration_evaluated": False, "validation_evaluated": False,
            "test_evaluated": False, **details})
        print(phase, details, flush=True)

    status("loading and auditing frozen multi-seed TRAIN data")
    data = ExpandedTrainData(ORIGINAL, EXPANSIONS, splits)
    names = read_json(BASE_NAMES)
    write_json(output / "data_audit.json", {
        "config_audit": data.config_audit,
        "event_counts": event_counts(data.events),
        "source_folds": data.source_folds,
        "original_protected_scenarios": {
            key: splits[key] for key in ("calibration", "validation", "test")},
        "all_expansion_scenarios_retained": True,
        "calibration_evaluated": False, "validation_evaluated": False,
        "test_evaluated": False,
    })
    write_json(output / "events_train_global.json", data.events)
    prepare_features(output, data, names, status)
    write_json(output / "feature_names.json", names)
    for outer in range(len(data.source_folds)):
        fit_classical_fold(output, outer, names, status)
    for outer in range(len(data.source_folds)):
        fit_sequence_fold(output, outer, names, status)

    held_parts = [load_arrays(output / f"fold_{outer}/features_held_out.npz")
                  for outer in range(len(data.source_folds))]
    oof = concatenate(held_parts)
    candidates = {}
    classical = [load_arrays(output / f"fold_{outer}/classical_predictions.npz")
                 for outer in range(len(data.source_folds))]
    for name in classical[0]:
        candidates[name] = np.concatenate([part[name] for part in classical])
    for candidate in SEQUENCE_CONFIGS:
        candidates[candidate] = np.concatenate([
            load_arrays(output / f"fold_{outer}/{candidate}_predictions.npz")["scores"]
            for outer in range(len(data.source_folds))])
    # Prespecified score-level combinations add no fitted parameters and remain OOF.
    candidates["mechanism_family_blend"] = logit_blend(
        candidates["mechanism"], candidates["family_mean"], .5)
    for sequence in SEQUENCE_CONFIGS:
        candidates[f"family_{sequence}_blend25"] = logit_blend(
            candidates["family_mean"], candidates[sequence], .75)
        candidates[f"family_{sequence}_blend50"] = logit_blend(
            candidates["family_mean"], candidates[sequence], .5)

    reports = {}
    for candidate, scores in candidates.items():
        status("auditing expanded OOF candidate", candidate=candidate)
        reports[candidate] = {
            "diagnostics": expert_diagnostics(oof, scores, data.events),
            "family_oracle_f1": {
                "drift": family_oracle(scores[:, 0], oof, 3),
                "noise": family_oracle(scores[:, 1], oof, 4),
            },
        }
    selected, gates = {}, {}
    baseline = reports["mechanism"]
    for column, family in enumerate(("drift", "noise")):
        winner = max(reports, key=lambda key: (
            reports[key]["diagnostics"][family]["macro_ap"],
            reports[key]["family_oracle_f1"][family]["f1"],
            reports[key]["diagnostics"][family]["worst_ap"]))
        selected[family] = winner
        candidate = reports[winner]["diagnostics"][family]
        control = baseline["diagnostics"][family]
        candidate_oracle = reports[winner]["family_oracle_f1"][family]
        control_oracle = baseline["family_oracle_f1"][family]
        gains = {sid: candidate["by_scenario_ap"][sid] - value
                 for sid, value in control["by_scenario_ap"].items()}
        required_improvements = max(3, len(gains) // 2)
        checks = {
            "macro_ap_gain_at_least_0_015": candidate["macro_ap"] >= control["macro_ap"] + .015,
            "oracle_f1_gain_at_least_0_03": candidate_oracle["f1"] >= control_oracle["f1"] + .03,
            "at_least_half_scenarios_improve": sum(value > 0 for value in gains.values()) >= required_improvements,
            "worst_ap_preserved": candidate["worst_ap"] >= control["worst_ap"] - .03,
            "clean_fpr_not_materially_higher": candidate["curve_clean_fpr"] <= control["curve_clean_fpr"] + .00025,
            "post_event_fp_not_materially_higher": candidate["curve_post_event_fp"] <= control["curve_post_event_fp"] + 5,
        }
        gates[family] = {"passed": bool(all(checks.values())), "checks": checks,
            "required_improved_scenarios": required_improvements,
            "improved_scenarios": sum(value > 0 for value in gains.values()),
            "macro_ap_gain": candidate["macro_ap"] - control["macro_ap"],
            "oracle_f1_gain": candidate_oracle["f1"] - control_oracle["f1"],
            "target_0_70_reached_in_train_oof_oracle": candidate_oracle["f1"] >= .70,
            "target_0_80_reached_in_train_oof_oracle": candidate_oracle["f1"] >= .80,
        }
    selected_scores = np.column_stack((candidates[selected["drift"]][:, 0],
                                       candidates[selected["noise"]][:, 1]))
    atomic_npz(output / "oof_predictions.npz", scores=selected_scores,
               **{key: oof[key] for key in ARRAY_KEYS if key != "X"})
    summary = {
        "status": "completed", "scope": signature["scope"],
        "event_counts": event_counts(data.events), "folds": data.source_folds,
        "baseline": "mechanism", "candidates": reports,
        "selection": selected, "promotion_gates": gates,
        "both_families_promoted": bool(all(gate["passed"] for gate in gates.values())),
        "next_action": ("freeze_full_train_fit_then_use_existing_calibration_once"
                        if all(gate["passed"] for gate in gates.values())
                        else "stop_before_calibration_and_report_train_oof_limit"),
        "calibration_evaluated": False, "validation_evaluated": False,
        "test_evaluated": False,
        "oof_predictions_sha256": sha(output / "oof_predictions.npz"),
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(output / "summary.json", summary)
    status("expanded TRAIN OOF screen complete", selection=selected,
           gates=gates, next_action=summary["next_action"])
    print(json.dumps({"selection": selected, "gates": gates,
                      "event_counts": summary["event_counts"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/operational/expanded_train_weak_experts_v1")
    run(parser.parse_args().output_dir)
