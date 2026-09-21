"""Freeze the one-sided feedback veto after the symmetric candidate failed."""
from __future__ import annotations

import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np

from wdn.evidence_feedback import specialist_veto_masks
from wdn.latency_deployment import FAMILY_NAMES, family_scores

from evaluate_cross_mechanism_deployment import branch_scores, decide


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "runs/operational/cross_mechanism_deployment_v1"
META = ROOT / "runs/operational/feedback_router_calibration_v2"
ADDENDUM = ROOT / "thesis_v2/FEEDBACK_ROUTER_VETO_ADDENDUM.md"
OUTPUT = ROOT / "runs/operational/feedback_router_veto_v1"
MARGINS = (.05, .10, .20)
CUTOFFS = (.10, .30, .70)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(report):
    return {name: float(report[name]["f1"]) for name in FAMILY_NAMES.values()} | {
        "overall": float(report["_overall"]["f1"]),
        "clean_fpr": float(report["_overall"]["clean_fpr"]),
        "worst": float(report["_overall"]["worst_family_f1"])}


def gate(report, baseline):
    return (report["_overall"]["clean_fpr"] <= .005
        and report["random"]["f1"] >= .90 and report["targeted"]["f1"] >= .90
        and report["replay"]["f1"] >= .82
        and report["drift"]["f1"] >= .81 and report["noise"]["f1"] >= .81
        and report["_overall"]["worst_family_f1"] >= baseline["_overall"]["worst_family_f1"]
        and report["_overall"]["f1"] >= baseline["_overall"]["f1"] - .005)


def apply_veto(scores, thresholds, router, feedback, margin, cutoff=None):
    mixture = scores["mixture"] > thresholds["mixture"]
    specialists = np.column_stack((scores["drift"] > thresholds["drift"],
                                   scores["noise"] > thresholds["noise"]))
    if cutoff is None:
        replay = router[:, 2]
        veto = np.column_stack((replay > router[:, 3] + margin,
                                replay > router[:, 4] + margin))
    else:
        veto = specialist_veto_masks(router, feedback, margin, cutoff)
    return mixture | np.any(specialists & ~veto, axis=1), veto


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selection_path = OUTPUT / "selection_frozen.json"
    if selection_path.exists():
        print("Feedback veto already frozen; no reselection")
        return
    base = dict(np.load(BASE / "scores_calibration.npz"))
    meta = dict(np.load(META / "feedback_calibration.npz"))
    old = json.loads((BASE / "selection_frozen.json").read_text())
    rule = old["selected_rule"]
    scores = branch_scores(base, rule["mixture_delta"], rule["expert_source"],
                           rule["specialist_variant"], rule["specialist_delta"])
    baseline_decision = decide(rule, scores)
    baseline = family_scores(baseline_decision, base)
    labels = np.asarray(base["labels"]) > 0
    families = np.asarray(base["families"])
    passing = []
    diagnostics = []
    for margin, cutoff in product(MARGINS, CUTOFFS):
        decision, veto = apply_veto(scores, rule["thresholds"], meta["router"],
                                    meta["feedback"], margin, cutoff)
        report = family_scores(decision, base)
        removed = baseline_decision & ~decision
        item = {"margin": margin, "feedback_cutoff": cutoff,
            "report": compact(report), "removed_alarms": int(removed.sum()),
            "removed_positive_alarms": int(np.sum(removed & labels)),
            "removed_replay_false_alarms": int(np.sum(removed & ~labels & (families == 2))),
            "passes": bool(gate(report, baseline))}
        diagnostics.append(item)
        if item["passes"]:
            key = (-item["removed_positive_alarms"], report["replay"]["f1"],
                   report["_overall"]["worst_family_f1"], report["_overall"]["f1"])
            passing.append((key, item, report, decision, veto))
    if not passing:
        raise RuntimeError("No one-sided feedback veto passes the declared gate")
    _, chosen, report, decision, veto = max(passing, key=lambda value: value[0])
    router_only_decision, router_only_veto = apply_veto(
        scores, rule["thresholds"], meta["router"], meta["feedback"],
        chosen["margin"], None)
    router_only = family_scores(router_only_decision, base)
    feedback_active = (int(np.sum((router_only_decision != decision) & labels)) > 0
                       and report["_overall"]["worst_family_f1"]
                           >= router_only["_overall"]["worst_family_f1"])
    result = {
        "status": "promoted" if feedback_active else "rejected",
        "scope": "99-scenario calibration only",
        "rule": {"type": "replay_router_feedback_veto",
                 "margin": chosen["margin"],
                 "feedback_cutoff": chosen["feedback_cutoff"],
                 "base_thresholds": rule["thresholds"],
                 "mixture_delta": 0, "specialist_delta": 3},
        "baseline": compact(baseline),
        "selected": chosen,
        "router_only_ablation": compact(router_only),
        "router_only_vetoes": int(router_only_veto.sum()),
        "feedback_vetoes": int(veto.sum()),
        "feedback_changes_positive_decisions": int(np.sum((router_only_decision != decision) & labels)),
        "feedback_active": bool(feedback_active),
        "passing_candidates": len(passing),
        "grid": {"margins": MARGINS, "cutoffs": CUTOFFS},
        "diagnostics": diagnostics,
        "addendum_sha256": sha(ADDENDUM),
        "base_selection_sha256": sha(BASE / "selection_frozen.json"),
        "feedback_predictions_sha256": sha(META / "feedback_calibration.npz"),
        "eval1_evaluated": False, "eval2_evaluated": False,
        "eval3_evaluated": False, "test_evaluated": False,
    }
    selection_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
