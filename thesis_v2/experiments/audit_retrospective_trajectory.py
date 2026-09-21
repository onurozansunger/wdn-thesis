"""Independent artifact/metric audit for the retrospective trajectory screen."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from wdn.run_expert_redesign import sha, write_json


RUN = Path("runs/operational/retrospective_trajectory_v1")
OUTPUT = Path("thesis_v2/outputs/retrospective_trajectory_audit.json")


def metrics(labels, decisions):
    labels, decisions = np.asarray(labels, bool), np.asarray(decisions, bool)
    tp = int(np.sum(labels & decisions)); fp = int(np.sum(~labels & decisions))
    fn = int(np.sum(labels & ~decisions))
    return {"f1": float(2*tp/max(2*tp+fp+fn, 1)),
            "precision": float(tp/max(tp+fp, 1)), "recall": float(tp/max(tp+fn, 1)),
            "tp": tp, "fp": fp, "fn": fn}


def run():
    summary = json.loads((RUN/"summary.json").read_text())
    protocol = json.loads((RUN/"protocol.json").read_text())
    arrays = dict(np.load(RUN/"oof_predictions.npz"))
    checks = {"oof_hash_exact": sha(RUN/"oof_predictions.npz") == summary["oof_sha256"],
        "protocol_exact": summary["protocol"] == protocol,
        "protocol_document_hash_exact": protocol["protocol_sha256"] == sha(
            Path("thesis_v2/RETROSPECTIVE_TRAJECTORY_PROTOCOL.md")),
        "train_scenarios_exact": set(np.unique(arrays["scenario"]).tolist()) == {15, 17, 18, 23},
        "calibration_not_evaluated": summary["calibration_evaluated"] is False,
        "validation_not_evaluated": summary["validation_evaluated"] is False,
        "test_not_evaluated": summary["test_evaluated"] is False}
    reproduced = metrics(arrays["label"], arrays["decision"])
    checks["nested_metrics_exact"] = reproduced == summary["nested_decision_metrics"]
    scenario = {}
    for sid in (15, 17, 18, 23):
        local = arrays["scenario"] == sid
        row = {"metrics": metrics(arrays["label"][local], arrays["decision"][local]),
            "ap": float(average_precision_score(arrays["label"][local], arrays["score"][local]))}
        scenario[str(sid)] = row
        checks[f"scenario_{sid}_exact"] = (row["metrics"] == summary["folds"][str(sid)]["held_metrics"]
            and row["ap"] == summary["folds"][str(sid)]["held_ap"])
    result = {"independent_recalculation": True, "checks": checks,
        "all_checks_pass": all(checks.values()), "reproduced_nested_metrics": reproduced,
        "reproduced_by_scenario": scenario, "screen_passed": summary["passed"],
        "conclusion": "fixed bidirectional trajectory candidate failed TRAIN promotion gates"}
    write_json(OUTPUT, result)
    write_json(RUN/"independent_audit.json", result)
    if not result["all_checks_pass"]:
        raise ValueError("Retrospective trajectory audit failed")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
