"""Nested TRAIN-only screen for a retrospective injected-noise trajectory expert."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import average_precision_score

from wdn.models.retrospective_trajectory import (TrajectoryConfig, fit_trajectory_model,
    model_metadata, predict_trajectory_model)
from wdn.run_expert_redesign import load_arrays, sha, write_json


FAMILY_RUN = Path("runs/operational/family_specific_experts_v1")
WEAK_RUN = Path("runs/operational/weak_family_v1/trial_0000")
BASE_RUN = Path("runs/operational/mechanism_redesign_v1")
PROTOCOL_PATH = Path("thesis_v2/RETROSPECTIVE_TRAJECTORY_PROTOCOL.md")
NOISE_ID = 4
NOISE_SCENARIOS = (15, 17, 18, 23)
BASE_COLUMNS = ("residual", "abs_residual", "normal_error_scale", "last_gap",
    "dynamic_innovation", "dynamic_abs_innovation", "dynamic_sigma",
    "seq_innovation", "seq_abs_innovation", "seq_normal_sigma", "noise_state_4.0",
    "noise_energy_8")


def _logit(values):
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1-1e-6)
    return np.log(values/(1-values))


def _scenario_ranks(values, scenarios):
    ranks = np.empty_like(values, dtype=np.float32)
    for scenario in np.unique(scenarios):
        index = np.flatnonzero(scenarios == scenario)
        for column in range(values.shape[1]):
            order = np.argsort(values[index, column], kind="stable")
            local = np.empty(len(index), dtype=np.float32)
            local[order] = (np.arange(len(index), dtype=np.float32)+0.5)/len(index)
            ranks[index, column] = local
    return ranks


def _load_inputs():
    family = load_arrays(FAMILY_RUN/"oof_predictions.npz")
    weak = load_arrays(WEAK_RUN/"oof_predictions.npz")
    pieces = [load_arrays(path) for path in sorted(BASE_RUN.glob("fold_*/features_held_out.npz"))]
    base = {key: np.concatenate([piece[key] for piece in pieces])
            for key in ("X", "labels", "families", "scenario", "timestep", "node")}
    for key in ("labels", "families", "scenario"):
        np.testing.assert_array_equal(family[key], weak[key])
    for key in ("labels", "families", "scenario", "timestep", "node"):
        np.testing.assert_array_equal(family[key], base[key])
    names = json.loads((BASE_RUN/"feature_names.json").read_text())
    columns = np.asarray([names.index(name) for name in BASE_COLUMNS], dtype=int)
    score_probabilities = np.column_stack((weak["experts"], family["scores_E"],
                                           family["head_scores"]))
    score_logits = _logit(score_probabilities)
    rank_values = _scenario_ranks(score_logits, family["scenario"])
    continuous = np.column_stack((score_logits, base["X"][:, columns])).astype(np.float32)
    return family, continuous, rank_values


def _window_rows(arrays, radius):
    selected = np.flatnonzero(arrays["families"] == NOISE_ID)
    width = 2*radius+1
    row_ids = np.full((len(selected), width), -1, dtype=np.int64)
    lookup = {(int(s), int(n), int(t)): i for i, (s, n, t) in enumerate(zip(
        arrays["scenario"], arrays["node"], arrays["timestep"]))}
    for sample, row in enumerate(selected):
        scenario, node, center = (int(arrays[key][row])
                                  for key in ("scenario", "node", "timestep"))
        for offset in range(-radius, radius+1):
            row_ids[sample, offset+radius] = lookup.get((scenario, node, center+offset), -1)
    if np.any(row_ids[:, radius] != selected):
        raise ValueError("Every trajectory must contain its centre observation")
    return selected, row_ids


def _materialise(row_ids, continuous, ranks, arrays, fit_scenarios):
    fit_rows = np.isin(arrays["scenario"], fit_scenarios)
    mean = continuous[fit_rows].mean(axis=0)
    scale = continuous[fit_rows].std(axis=0)
    scale = np.maximum(scale, 1e-3)
    observed = row_ids >= 0
    safe = np.maximum(row_ids, 0)
    raw = np.clip((continuous[safe]-mean)/scale, -8, 8)
    rank = np.clip((ranks[safe]-.5)/np.sqrt(1/12), -2, 2)
    raw[~observed] = 0
    rank[~observed] = 0
    return np.concatenate((raw, rank, observed[..., None].astype(np.float32)), axis=2)


def _best_threshold(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels, dtype=bool)
    order = np.argsort(-scores, kind="stable")
    score, target = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(score[:-1] != score[1:]), len(score)-1]
    predicted, true_positive = ends+1, np.cumsum(target)[ends]
    f1 = 2*true_positive/np.maximum(predicted+target.sum(), 1)
    best = int(np.argmax(f1))
    return float(np.nextafter(score[ends[best]], -np.inf)), float(f1[best])


def _metrics(labels, decisions):
    labels, decisions = np.asarray(labels, bool), np.asarray(decisions, bool)
    tp = int(np.sum(labels & decisions)); fp = int(np.sum(~labels & decisions))
    fn = int(np.sum(labels & ~decisions))
    return {"f1": float(2*tp/max(2*tp+fp+fn, 1)),
            "precision": float(tp/max(tp+fp, 1)), "recall": float(tp/max(tp+fn, 1)),
            "tp": tp, "fp": fp, "fn": fn}


def run(output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"summary.json").exists():
        summary = json.loads((output/"summary.json").read_text())
        if sha(output/"oof_predictions.npz") != summary["oof_sha256"]:
            raise ValueError("Completed retrospective OOF artifact changed")
        print("Retrospective trajectory screen already complete; no refit", flush=True)
        return
    config = TrajectoryConfig()
    protocol = {"version": 1, "scope": "TRAIN-only nested retrospective noise localisation",
        "noise_scenarios": list(NOISE_SCENARIOS), "model": model_metadata(config),
        "base_columns": list(BASE_COLUMNS), "optuna_trials": 0,
        "calibration_evaluated": False, "validation_evaluated": False,
        "test_evaluated": False, "protocol_sha256": sha(PROTOCOL_PATH),
        "input_sha256": {"family_oof": sha(FAMILY_RUN/"oof_predictions.npz"),
            "weak_oof": sha(WEAK_RUN/"oof_predictions.npz"),
            "base_feature_names": sha(BASE_RUN/"feature_names.json"),
            **{str(path): sha(path) for path in sorted(BASE_RUN.glob("fold_*/features_held_out.npz"))}}}
    write_json(output/"protocol.json", protocol)
    started = time.monotonic()
    arrays, continuous, ranks = _load_inputs()
    selected, row_ids = _window_rows(arrays, config.radius)
    scenarios = arrays["scenario"][selected]
    labels = arrays["labels"][selected] > 0
    if tuple(np.unique(scenarios).tolist()) != NOISE_SCENARIOS:
        raise ValueError("Noise TRAIN scenario set changed")
    oof_path, folds_path = output/"oof_predictions.npz", output/"folds.json"
    if oof_path.exists():
        if not folds_path.exists():
            raise ValueError("OOF predictions exist without fold metadata; refusing a silent refit")
        saved = load_arrays(oof_path)
        for key, expected in (("selected_row", selected), ("label", labels),
                              ("scenario", scenarios)):
            np.testing.assert_array_equal(saved[key], expected)
        scores, decisions = saved["score"], saved["decision"].astype(bool)
        folds = json.loads(folds_path.read_text())
        print("Resuming report generation from completed OOF predictions; no refit", flush=True)
    else:
        scores = np.zeros(len(selected), dtype=np.float64)
        decisions = np.zeros(len(selected), dtype=bool)
        folds = {}
        for outer_number, held in enumerate(NOISE_SCENARIOS):
            fit_scenarios = tuple(s for s in NOISE_SCENARIOS if s != held)
            x = _materialise(row_ids, continuous, ranks, arrays, fit_scenarios)
            inner_scores, inner_labels = [], []
            for inner_number, inner_held in enumerate(fit_scenarios):
                inner_fit = tuple(s for s in fit_scenarios if s != inner_held)
                train = np.isin(scenarios, inner_fit)
                predict = scenarios == inner_held
                inner_model = fit_trajectory_model(x[train], labels[train], scenarios[train], config,
                    seed_offset=100*outer_number+10*inner_number)
                inner_scores.append(predict_trajectory_model(inner_model, x[predict]))
                inner_labels.append(labels[predict])
            threshold, inner_f1 = _best_threshold(np.concatenate(inner_scores),
                                                  np.concatenate(inner_labels))
            train = np.isin(scenarios, fit_scenarios)
            predict = scenarios == held
            outer_model = fit_trajectory_model(x[train], labels[train], scenarios[train], config,
                                               seed_offset=1000+outer_number)
            held_score = predict_trajectory_model(outer_model, x[predict])
            scores[predict] = held_score
            decisions[predict] = held_score > threshold
            folds[str(held)] = {"fit_scenarios": list(fit_scenarios), "threshold": threshold,
                "inner_oof_f1_at_selected_threshold": inner_f1,
                "held_metrics": _metrics(labels[predict], decisions[predict]),
                "held_ap": float(average_precision_score(labels[predict], held_score))}
            write_json(folds_path, folds)
            print(f"held noise scenario {held}: {folds[str(held)]}", flush=True)
        np.savez_compressed(oof_path, selected_row=selected, score=scores,
                            decision=decisions, label=labels, scenario=scenarios)
    scenario_f1 = {scenario: folds[str(scenario)]["held_metrics"]["f1"]
                   for scenario in NOISE_SCENARIOS}
    scenario_ap = {scenario: folds[str(scenario)]["held_ap"] for scenario in NOISE_SCENARIOS}
    oracle_threshold, oracle_f1 = _best_threshold(scores, labels)
    nested = _metrics(labels, decisions)
    gates = {"nested_f1_at_least_080": bool(nested["f1"] >= .8),
        "every_scenario_f1_at_least_080": bool(min(scenario_f1.values()) >= .8),
        "macro_ap_at_least_080": bool(np.mean(list(scenario_ap.values())) >= .8),
        "split_integrity": True}
    summary = {"scope": protocol["scope"], "protocol": protocol, "folds": folds,
        "nested_decision_metrics": nested, "scenario_f1": scenario_f1,
        "scenario_ap": scenario_ap, "macro_ap": float(np.mean(list(scenario_ap.values()))),
        "hindsight_oof_ranking_diagnostic": {"f1": oracle_f1,
            "threshold_not_deployable": oracle_threshold},
        "gates": gates, "passed": all(gates.values()),
        "calibration_evaluated": False, "validation_evaluated": False,
        "test_evaluated": False, "elapsed_seconds": time.monotonic()-started,
        "oof_sha256": sha(output/"oof_predictions.npz")}
    write_json(output/"summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/operational/retrospective_trajectory_v1")
    args = parser.parse_args()
    run(args.output_dir)
