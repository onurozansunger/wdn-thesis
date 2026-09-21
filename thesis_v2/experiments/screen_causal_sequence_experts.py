"""TRAIN-only scenario-OOF screen for causal drift/noise GRU experts."""
from __future__ import annotations

import argparse, fcntl, json, time
from pathlib import Path

import joblib
import numpy as np
import yaml

from wdn.models.causal_sequence import (CausalSequenceConfig, CausalSequenceExpert,
    causal_score_memory)
from wdn.run_expert_redesign import load_arrays, read_json, sha, write_json
from wdn.screen_family_specific import _best_f1
from wdn.screen_recovery import expert_diagnostics
from wdn.train_weak_families import add_early, concatenate

BASE = Path("runs/operational/mechanism_redesign_v1")
FAMILY = Path("runs/operational/family_specific_experts_v1")
DATA = Path("data/thesis_v2/operational_modena_seed811")
PROTOCOL = Path("thesis_v2/CAUSAL_SEQUENCE_EXPERT_PROTOCOL.md")
CONFIGS = {
    "short": CausalSequenceConfig(window=8, hidden_size=16, embedding_size=20, epochs=8),
    "long": CausalSequenceConfig(window=16, hidden_size=24, embedding_size=28, epochs=10),
}
KEYS = ("X", "labels", "families", "event", "scenario", "timestep", "node")


def arrays(path, events):
    raw = load_arrays(path)
    return add_early({key: raw[key] for key in KEYS}, events)


def transformed(scores, oof, half_life):
    if half_life == "raw":
        return scores
    return np.column_stack([causal_score_memory(scores[:, column], oof["scenario"],
        oof["timestep"], oof["node"], half_life) for column in range(2)])


def oracle(scores, oof, family_id):
    selected = oof["families"] == family_id
    return _best_f1(scores[selected], oof["labels"][selected])


def run(output_dir):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    lock = (output/"campaign.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Causal sequence screen already running") from error
    if (output/"summary.json").exists():
        print("Causal sequence screen already complete; no refit", flush=True); return
    generation = yaml.safe_load((DATA/"generate_config.yaml").read_text())
    if generation["missing_rate_pressure"] != .5 or generation["missing_rate_flow"] != .5:
        raise ValueError("Fixed missingness changed")
    signature = read_json(BASE/"signature.json")
    folds = read_json(FAMILY/"outer_folds.json")
    events = json.loads((DATA/"events.json").read_text())
    names = read_json(BASE/"feature_names.json")
    protocol = {"version": 1, "scope": "TRAIN-only whole-scenario OOF component screen",
        "configs": {name: config.__dict__ for name, config in CONFIGS.items()},
        "memory_half_lives": ["raw", 3, 6, 12], "optuna_trials": 0,
        "calibration_evaluated": False, "validation_evaluated": False,
        "test_evaluated": False, "protocol_sha256": sha(PROTOCOL),
        "input_hashes": {"family_oof": sha(FAMILY/"oof_predictions.npz"),
            "feature_names": sha(BASE/"feature_names.json"),
            "generate_config": sha(DATA/"generate_config.yaml"),
            **{str(path): sha(path) for path in sorted(BASE.glob("fold_*/features_*.npz"))
               if "validation" not in path.name}}}
    write_json(output/"protocol.json", protocol); write_json(output/"folds.json", folds)
    started = time.monotonic()
    def status(phase, **details):
        write_json(output/"status.json", {"phase": phase,
            "elapsed_seconds": time.monotonic()-started, "calibration_evaluated": False,
            "validation_evaluated": False, "test_evaluated": False, **details})
        print(phase, details, flush=True)

    baseline = load_arrays(FAMILY/"oof_predictions.npz")
    parts = [arrays(path, events) for path in sorted(BASE.glob("fold_*/features_held_out.npz"))]
    oof = concatenate(parts)
    for key in ("labels", "families", "event", "scenario", "timestep", "node", "early"):
        np.testing.assert_array_equal(oof[key], baseline[key])
    baseline_scores = baseline["scores_E"]
    baseline_diag = expert_diagnostics(oof, baseline_scores, events)
    baseline_oracle = {"drift": oracle(baseline_scores[:, 0], oof, 3),
                       "noise": oracle(baseline_scores[:, 1], oof, 4)}
    candidate_scores, candidates = {}, {}
    for config_name, config in CONFIGS.items():
        held_scores = []
        for outer, held in enumerate(folds):
            training = arrays(BASE/f"fold_{outer}/features_train.npz", events)
            held_arrays = arrays(BASE/f"fold_{outer}/features_held_out.npz", events)
            if set(training["scenario"]) & set(held) or set(held_arrays["scenario"]) != set(held):
                raise ValueError("Outer scenario isolation changed")
            fold_scores, models = [], {}
            for family_name in ("drift", "noise"):
                status("fitting causal sequence expert", candidate=config_name,
                    outer_fold=outer, family=family_name)
                model = CausalSequenceExpert(names, family_name, config,
                    seed_offset=100*outer).fit(training)
                fold_scores.append(model.predict(held_arrays)); models[family_name] = model
            folder = output/config_name/f"outer_{outer}"; folder.mkdir(parents=True, exist_ok=True)
            joblib.dump(models, folder/"bundle.joblib")
            fold_score = np.column_stack(fold_scores); held_scores.append(fold_score)
            np.savez_compressed(folder/"held_predictions.npz", scores=fold_score,
                labels=held_arrays["labels"], families=held_arrays["families"],
                scenario=held_arrays["scenario"], timestep=held_arrays["timestep"],
                node=held_arrays["node"])
            status("causal sequence fold complete", candidate=config_name,
                outer_fold=outer, rows=len(fold_score))
        raw = np.concatenate(held_scores)
        for memory in ("raw", 3, 6, 12):
            key = f"{config_name}_memory_{memory}"
            score = transformed(raw, oof, memory); candidate_scores[key] = score
            diag = expert_diagnostics(oof, score, events)
            candidates[key] = {"config": config_name, "memory_half_life": memory,
                "diagnostics": diag,
                "family_oracle_f1": {"drift": oracle(score[:, 0], oof, 3),
                                     "noise": oracle(score[:, 1], oof, 4)}}
        np.savez_compressed(output/config_name/"oof_raw.npz", scores=raw)

    selected = {}; gates = {}
    for column, family_name in enumerate(("drift", "noise")):
        winner = max(candidates, key=lambda key: (
            candidates[key]["diagnostics"][family_name]["macro_ap"],
            candidates[key]["diagnostics"][family_name]["worst_ap"],
            candidates[key]["family_oracle_f1"][family_name]["f1"],
            candidates[key]["diagnostics"][family_name]["early_macro_recall"]))
        selected[family_name] = winner
        candidate = candidates[winner]["diagnostics"][family_name]
        control = baseline_diag[family_name]
        candidate_oracle = candidates[winner]["family_oracle_f1"][family_name]
        control_oracle = baseline_oracle[family_name]
        gains = {sid: candidate["by_scenario_ap"][sid]-value
                 for sid, value in control["by_scenario_ap"].items()}
        checks = {"macro_ap_gain_at_least_0_015": candidate["macro_ap"] >= control["macro_ap"]+.015,
            "oracle_f1_gain_at_least_0_03": candidate_oracle["f1"] >= control_oracle["f1"]+.03,
            "three_scenarios_improve": sum(value > 0 for value in gains.values()) >= 3,
            "worst_ap_preserved": candidate["worst_ap"] >= control["worst_ap"]-.03}
        gates[family_name] = {"passed": bool(all(checks.values())), "checks": checks,
            "scenario_ap_gains": gains, "macro_ap_gain": candidate["macro_ap"]-control["macro_ap"],
            "oracle_f1_gain": candidate_oracle["f1"]-control_oracle["f1"]}
    selected_scores = np.column_stack([candidate_scores[selected["drift"]][:, 0],
                                       candidate_scores[selected["noise"]][:, 1]])
    np.savez_compressed(output/"oof_predictions.npz", scores=selected_scores,
        labels=oof["labels"], families=oof["families"], event=oof["event"],
        scenario=oof["scenario"], timestep=oof["timestep"], node=oof["node"], early=oof["early"])
    summary = {"scope": protocol["scope"], "protocol": protocol,
        "baseline": {"diagnostics": baseline_diag, "family_oracle_f1": baseline_oracle},
        "candidates": candidates, "selection": selected, "gates": gates,
        "both_families_passed": bool(all(gate["passed"] for gate in gates.values())),
        "next_action": ("fit_full_train_then_calibrate" if all(gate["passed"] for gate in gates.values())
                        else "stop_before_calibration"),
        "calibration_evaluated": False, "validation_evaluated": False,
        "test_evaluated": False, "oof_sha256": sha(output/"oof_predictions.npz"),
        "elapsed_seconds": time.monotonic()-started}
    write_json(output/"summary.json", summary)
    status("causal sequence screen complete", selected=selected, gates=gates,
        next_action=summary["next_action"])
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/operational/causal_sequence_experts_v1")
    run(parser.parse_args().output_dir)
