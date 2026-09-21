"""TRAIN-only nested-threshold screen for conditional incident-window localisation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from wdn.models.incident_window import aggregate_incident_scores
from wdn.run_expert_redesign import load_arrays, sha, write_json


INPUT = Path("runs/operational/family_specific_experts_v1/oof_predictions.npz")
PROTOCOL = Path("thesis_v2/INCIDENT_WINDOW_LOCALIZER_PROTOCOL.md")
CANDIDATES = {"drift": {"family_id": 3, "source": "scores_E[:,0]", "mode": "top2"},
              "noise": {"family_id": 4, "source": "head_scores[:,3]", "mode": "logit"}}


def best_threshold(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels, bool)
    order = np.argsort(-scores, kind="stable")
    score, target = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(score[:-1] != score[1:]), len(score)-1]
    predicted, tp = ends+1, np.cumsum(target)[ends]
    f1 = 2*tp/np.maximum(predicted+target.sum(), 1)
    best = int(np.argmax(f1))
    threshold = np.nextafter(score[ends[best]], -np.inf)
    return float(threshold), float(f1[best])


def metrics(labels, decisions):
    labels, decisions = np.asarray(labels, bool), np.asarray(decisions, bool)
    tp = int(np.sum(labels & decisions)); fp = int(np.sum(~labels & decisions))
    fn = int(np.sum(labels & ~decisions))
    return {"f1": float(2*tp/max(2*tp+fp+fn, 1)),
        "precision": float(tp/max(tp+fp, 1)), "recall": float(tp/max(tp+fn, 1)),
        "tp": tp, "fp": fp, "fn": fn}


def run(output_dir):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    if (output/"summary.json").exists():
        summary = json.loads((output/"summary.json").read_text())
        if sha(output/"oof_predictions.npz") != summary["oof_sha256"]:
            raise ValueError("Completed incident-window OOF artifact changed")
        print("Incident-window screen complete; no recomputation")
        return
    arrays = load_arrays(INPUT)
    result, saved = {}, {}
    for family, recipe in CANDIDATES.items():
        selected = arrays["families"] == recipe["family_id"]
        source = arrays["scores_E"][:, 0] if family == "drift" else arrays["head_scores"][:, 3]
        scores = aggregate_incident_scores(source, arrays["event"], arrays["node"],
            arrays["timestep"], selected, recipe["mode"])[selected]
        labels = arrays["labels"][selected] > 0
        scenarios = arrays["scenario"][selected]
        decisions = np.zeros(len(scores), dtype=bool)
        folds = {}
        for held in np.unique(scenarios):
            train, test = scenarios != held, scenarios == held
            threshold, inner_f1 = best_threshold(scores[train], labels[train])
            decisions[test] = scores[test] > threshold
            folds[str(int(held))] = {"threshold_from_other_three_scenarios": threshold,
                "training_hindsight_f1": inner_f1,
                "held_metrics": metrics(labels[test], decisions[test]),
                "held_ap": float(average_precision_score(labels[test], scores[test]))}
        nested = metrics(labels, decisions)
        oracle_threshold, oracle_f1 = best_threshold(scores, labels)
        result[family] = {"recipe": recipe, "nested_decision_metrics": nested,
            "folds": folds, "macro_ap": float(np.mean([row["held_ap"] for row in folds.values()])),
            "hindsight_oof_diagnostic": {"f1": oracle_f1,
                "threshold_not_deployable": oracle_threshold}, "passed": bool(nested["f1"] >= .8)}
        saved.update({f"{family}_score": scores, f"{family}_decision": decisions,
            f"{family}_label": labels, f"{family}_scenario": scenarios,
            f"{family}_row": np.flatnonzero(selected)})
    np.savez_compressed(output/"oof_predictions.npz", **saved)
    protocol = {"version": 1, "conditional_offline_task": True,
        "external_window_and_family_hypothesis_required": True, "candidates": CANDIDATES,
        "protocol_sha256": sha(PROTOCOL), "input_sha256": sha(INPUT),
        "calibration_evaluated": False, "validation_evaluated": False,
        "test_evaluated": False}
    summary = {"scope": "TRAIN-only conditional retrospective incident localisation",
        "protocol": protocol, "families": result,
        "passed": all(row["passed"] for row in result.values()),
        "calibration_evaluated": False, "validation_evaluated": False,
        "test_evaluated": False, "oof_sha256": sha(output/"oof_predictions.npz")}
    write_json(output/"protocol.json", protocol); write_json(output/"summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/operational/incident_window_localizer_v1")
    run(parser.parse_args().output_dir)
