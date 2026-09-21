"""Recalculate and consolidate online and conditional incident-window results."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wdn.run_expert_redesign import sha, write_json


TRAIN = Path("runs/operational/incident_window_localizer_v1")
VALIDATION = Path("runs/operational/incident_window_validation_v1")
FALLBACK = Path("runs/operational/incident_window_noise_fallback_v1")
ONLINE = Path("runs/operational/drift_noise_recovery_v1/summary.json")
OUTPUT = Path("thesis_v2/outputs/incident_window_localizer_results.json")
REPORT = Path("thesis_v2/INCIDENT_WINDOW_LOCALIZER_RESULTS.md")


def metrics(labels, decisions):
    labels, decisions = np.asarray(labels, bool), np.asarray(decisions, bool)
    tp = int(np.sum(labels & decisions)); fp = int(np.sum(~labels & decisions))
    fn = int(np.sum(labels & ~decisions))
    return {"f1": float(2*tp/max(2*tp+fp+fn, 1)),
        "precision": float(tp/max(tp+fp, 1)), "recall": float(tp/max(tp+fn, 1)),
        "tp": tp, "fp": fp, "fn": fn}


def run():
    train_summary = json.loads((TRAIN/"summary.json").read_text())
    train = dict(np.load(TRAIN/"oof_predictions.npz"))
    validation_summary = json.loads((VALIDATION/"summary.json").read_text())
    validation = dict(np.load(VALIDATION/"predictions_validation.npz"))
    fallback_summary = json.loads((FALLBACK/"summary.json").read_text())
    fallback = dict(np.load(FALLBACK/"predictions_validation.npz"))
    online_summary = json.loads(ONLINE.read_text())["previous_development_validation_unchanged"]
    checks = {"train_oof_hash_exact": sha(TRAIN/"oof_predictions.npz") == train_summary["oof_sha256"],
        "train_passed_both_families": train_summary["passed"] is True,
        "train_test_untouched": train_summary["test_evaluated"] is False,
        "validation_test_untouched": validation_summary["test_evaluated"] is False,
        "fallback_hash_exact": sha(FALLBACK/"predictions_validation.npz") ==
            fallback_summary["validation_predictions_sha256"],
        "fallback_test_untouched": fallback_summary["test_evaluated"] is False,
        "primary_selection_precedes_validation_artifact":
            (VALIDATION/"selection_frozen.json").stat().st_mtime_ns <=
            (VALIDATION/"predictions_validation.npz").stat().st_mtime_ns,
        "fallback_selection_precedes_validation_artifact":
            (FALLBACK/"selection_frozen.json").stat().st_mtime_ns <=
            (FALLBACK/"predictions_validation.npz").stat().st_mtime_ns}
    reproduced_train = {}
    for family in ("drift", "noise"):
        row = metrics(train[f"{family}_label"], train[f"{family}_decision"])
        reproduced_train[family] = row
        checks[f"train_{family}_metrics_exact"] = (
            row == train_summary["families"][family]["nested_decision_metrics"])
    thresholds = validation_summary["selection"]["thresholds"]
    reproduced_validation = {}
    for family in ("drift", "noise"):
        row = metrics(validation[f"{family}_label"],
                      validation[f"{family}_score"] > thresholds[family])
        reproduced_validation[family] = row
        checks[f"validation_{family}_metrics_exact"] = (
            row == validation_summary["validation"][family]["metrics"])
    reproduced_fallback = metrics(fallback["label"], fallback["decision"])
    checks["fallback_metrics_exact"] = reproduced_fallback == fallback_summary["validation"]["metrics"]
    result = {"audit_checks": checks, "all_checks_pass": all(checks.values()),
        "online_development_validation_unchanged": {
            "overall_f1": online_summary["overall"]["f1"],
            "replay_f1": online_summary["per_family"]["replay"]["f1"],
            "drift_f1": online_summary["per_family"]["stealthy"]["f1"],
            "noise_f1": online_summary["per_family"]["noise"]["f1"]},
        "conditional_incident_window_train_nested": reproduced_train,
        "conditional_incident_window_development_validation": reproduced_validation,
        "conditional_noise_fallback_development_validation": reproduced_fallback,
        "interpretation": {"train_conditional_target_passed": True,
            "development_validation_both_target_failed": True,
            "drift_validation_reached_080": reproduced_validation["drift"]["f1"] >= .8,
            "noise_validation_reached_080": reproduced_validation["noise"]["f1"] >= .8,
            "test_evaluated": False,
            "external_window_and_family_required": True}}
    if not result["all_checks_pass"]:
        raise ValueError("Incident-window independent audit failed")
    write_json(OUTPUT, result); write_json(TRAIN/"independent_audit.json", result)
    lines = ["# Incident-window localizer results", "",
        "All numbers below were independently recalculated from saved predictions. Test was not read.", "",
        "| Evaluation | Drift F1 | Noise F1 | Meaning |", "|---|---:|---:|---|",
        f"| Existing online development validation | {result['online_development_validation_unchanged']['drift_f1']:.4f} | {result['online_development_validation_unchanged']['noise_f1']:.4f} | causal point detector |",
        f"| Incident-window nested TRAIN | {reproduced_train['drift']['f1']:.4f} | {reproduced_train['noise']['f1']:.4f} | conditional retrospective localizer |",
        f"| Incident-window development validation | {reproduced_validation['drift']['f1']:.4f} | {reproduced_validation['noise']['f1']:.4f} | conditional retrospective localizer |",
        "", "The conditional TRAIN target passed for both families, but development validation did not:",
        "drift reached exactly 0.8000 and noise fell to 0.4043. The bounded separate-stack noise",
        f"fallback reached {reproduced_fallback['f1']:.4f}. Neither noise result supports a 0.80 claim.", "",
        "The localizer requires a correct externally supplied incident window and family hypothesis.",
        "It does not infer onset/family, has no clean-period FPR of its own, and must not replace the",
        "online detector's latency and false-alarm report. The evidence now points to event-count",
        "generalisation: four TRAIN noise incidents and one calibration incident do not cover the",
        "weaker development-validation noise event reliably. More tuning on this validation split is stopped.", ""]
    REPORT.write_text("\n".join(lines))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
