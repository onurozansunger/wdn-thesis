"""One-time EVAL-3 comparison of frozen baseline and feedback-router veto."""
from __future__ import annotations

import fcntl
import gc
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np

from wdn.delayed_decision_features import delayed_decision_features
from wdn.evidence_feedback import (SCORE_EVIDENCE_NAMES, contiguous_groups,
                                   group_values, local_evidence,
                                   specialist_veto_masks)
from wdn.latency_deployment import (FAMILY_NAMES, family_scores, forward_max,
                                    logit_blend, specialist_bank,
                                    specialist_scores)
from wdn.run_expert_redesign import CampaignData

from evaluate_cross_mechanism_deployment import bootstrap, expert_report


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/thesis_v2"
SEEDS = (30811, 31811, 32811, 33811, 34811, 35811)
DETECTOR = ROOT / "runs/operational/expanded_detector_v1/bundle.joblib"
HEAD = ROOT / "runs/operational/delayed_head_full_v1/bundle.joblib"
REFERENCE = ROOT / "runs/operational/seasonal_family_deployment_v2/full/reference.joblib"
FEEDBACK = ROOT / "runs/operational/feedback_router_v2/bundle.joblib"
BASE_SELECTION = ROOT / "runs/operational/cross_mechanism_deployment_v1/selection_frozen.json"
VETO_SELECTION = ROOT / "runs/operational/feedback_router_veto_v1/selection_frozen.json"
OUTPUT = ROOT / "runs/operational/feedback_router_eval3_v1"
DELTA = 3


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def atomic_npz(path, **arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def score_seed(seed, models):
    detector, head, reference, feedback = models
    data = CampaignData(DATA / f"operational_eval3_seed{seed}")
    arrays, names = specialist_bank(data, list(range(24)), reference)
    if names != detector["names"] or tuple(names) != feedback.names:
        raise ValueError("EVAL-3 feature schema differs from the frozen models")
    X = arrays["X"]
    indexed = {**arrays, "source": np.full(len(X), seed),
               "scenario": np.asarray(arrays["scenario"]) + seed * 1000}
    mixture = detector["mixture"].predict(X)
    frozen = specialist_scores(detector, X)
    pooled = np.column_stack((forward_max(frozen[:, 0], indexed, DELTA),
                              forward_max(frozen[:, 1], indexed, DELTA)))
    forward, forward_names = delayed_decision_features(indexed, names, DELTA)
    if forward_names != head["forward_names"]:
        raise ValueError("EVAL-3 forward feature schema differs from the frozen head")
    predicted = head["delayed"].predict(np.column_stack((X, forward)).astype(np.float32))
    delayed = np.column_stack((predicted["drift"], predicted["noise"]))
    final = np.column_stack((logit_blend(delayed[:, 0], pooled[:, 0], .5),
                             logit_blend(delayed[:, 1], pooled[:, 1], .5)))
    score_parts = {
        "frozen_drift": frozen[:, 0], "frozen_noise": frozen[:, 1],
        "maxpool_drift": pooled[:, 0], "maxpool_noise": pooled[:, 1],
        "delayed_drift": delayed[:, 0], "delayed_noise": delayed[:, 1],
        "final_drift": final[:, 0], "final_noise": final[:, 1],
    }
    if tuple(score_parts) != SCORE_EVIDENCE_NAMES:
        raise AssertionError("Score evidence order changed")
    local = local_evidence(X, names, score_parts)
    meta = feedback.predict(local, indexed)
    result = {key: np.asarray(indexed[key]) for key in
              ("labels", "families", "source", "scenario", "timestep", "node")}
    result.update({"mixture": mixture["mixture"], "general": mixture["general"],
                   "old_router": mixture["routing"], "specialists": final,
                   "evidence_router": meta["router"], "feedback": meta["feedback"]})
    del data, arrays, X, indexed, forward, local, mixture, meta
    gc.collect()
    return result


def decision(scores, thresholds, veto=None):
    mixture = scores["mixture"] > thresholds["mixture"]
    specialists = np.column_stack((scores["specialists"][:, 0] > thresholds["drift"],
                                   scores["specialists"][:, 1] > thresholds["noise"]))
    if veto is not None:
        specialists &= ~veto
    return mixture | specialists.any(axis=1)


def enrich(report, predicted, corpus):
    for code, name in FAMILY_NAMES.items():
        report[name].update(bootstrap(predicted, corpus, code))
    return report


def expert_metrics(corpus, thresholds, veto=None):
    labels = corpus["labels"] > 0
    clean = (corpus["families"] == 0) & ~labels
    result = {}
    for column, (name, code) in enumerate((("drift", 3), ("noise", 4))):
        predicted = corpus["specialists"][:, column] > thresholds[name]
        if veto is not None:
            predicted &= ~veto[:, column]
        scope = corpus["families"] == code
        tp = int(np.sum(predicted & labels & scope)); fp = int(np.sum(predicted & ~labels & scope))
        fn = int(np.sum(~predicted & labels & scope))
        result[name] = {"f1": 2 * tp / max(1, 2 * tp + fp + fn),
            "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
            "tp": tp, "fp": fp, "fn": fn, "clean_fpr": float(predicted[clean].mean()),
            **bootstrap(predicted, corpus, code)}
    return result


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / "campaign.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    report_path = OUTPUT / "eval3_report.json"
    if report_path.exists():
        print("EVAL-3 already evaluated once; refusing to repeat", flush=True)
        return
    started = time.monotonic()
    base_selection = json.loads(BASE_SELECTION.read_text())
    veto_selection = json.loads(VETO_SELECTION.read_text())
    if veto_selection["status"] != "promoted" or not veto_selection["feedback_active"]:
        raise RuntimeError("Feedback veto was not promoted on calibration")
    models = (joblib.load(DETECTOR), joblib.load(HEAD),
              joblib.load(REFERENCE), joblib.load(FEEDBACK))
    parts = []
    for seed in SEEDS:
        cache = OUTPUT / f"scores_seed{seed}.npz"
        if not cache.exists():
            print("scoring locked EVAL-3 seed", seed, flush=True)
            atomic_npz(cache, **score_seed(seed, models))
        parts.append(dict(np.load(cache)))
    corpus = {key: np.concatenate([part[key] for part in parts], axis=0)
              for key in parts[0]}
    atomic_npz(OUTPUT / "scores_eval3.npz", **corpus)
    thresholds = base_selection["selected_rule"]["thresholds"]
    baseline_decision = decision(corpus, thresholds)
    active = veto_selection["rule"]
    veto = specialist_veto_masks(corpus["evidence_router"], corpus["feedback"],
                                  active["margin"], active["feedback_cutoff"])
    candidate_decision = decision(corpus, thresholds, veto)
    baseline_report = enrich(family_scores(baseline_decision, corpus), baseline_decision, corpus)
    candidate_report = enrich(family_scores(candidate_decision, corpus), candidate_decision, corpus)
    labels = corpus["labels"] > 0
    families = corpus["families"]
    removed = baseline_decision & ~candidate_decision

    group, starts = contiguous_groups(corpus)
    group_family = group_values(families, starts, "family").astype(int)
    group_family[group_family == 5] = 1
    router_group = corpus["evidence_router"][starts]
    router_pred = router_group.argmax(1)
    router_report = {}
    for code, name in enumerate(("clean", "abrupt", "replay", "drift", "noise")):
        selected = group_family == code
        router_report[name] = {"groups": int(selected.sum()),
            "accuracy": float(np.mean(router_pred[selected] == code)),
            "mean_probability": float(router_group[selected, code].mean())}

    result = {
        "status": "completed", "scope": "locked EVAL-3, one scoring pass",
        "seeds": list(SEEDS), "rows": int(len(labels)),
        "declared_decision_latency_hours": DELTA,
        "baseline": baseline_report, "candidate": candidate_report,
        "baseline_experts": expert_metrics(corpus, thresholds),
        "candidate_experts": expert_metrics(corpus, thresholds, veto),
        "router": router_report,
        "feedback_effect": {
            "veto_rows": int(veto.sum()), "removed_alarms": int(removed.sum()),
            "removed_positive_alarms": int(np.sum(removed & labels)),
            "removed_replay_false_alarms": int(np.sum(removed & ~labels & (families == 2))),
            "removed_clean_false_alarms": int(np.sum(removed & ~labels & (families == 0))),
        },
        "all_families_at_or_above_080": bool(all(
            candidate_report[name]["f1"] >= .80 for name in FAMILY_NAMES.values())),
        "selection_sha256": sha(VETO_SELECTION),
        "base_selection_sha256": sha(BASE_SELECTION),
        "feedback_bundle_sha256": sha(FEEDBACK),
        "manifest_sha256": sha(OUTPUT / "eval3_manifest_frozen.json"),
        "evaluated_once": True, "test_evaluated": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(report_path, result)
    print(json.dumps({
        "baseline": {name: round(baseline_report[name]["f1"], 4)
                     for name in FAMILY_NAMES.values()},
        "candidate": {name: round(candidate_report[name]["f1"], 4)
                      for name in FAMILY_NAMES.values()},
        "overall": {"baseline": round(baseline_report["_overall"]["f1"], 4),
                    "candidate": round(candidate_report["_overall"]["f1"], 4)},
        "clean_fpr": {"baseline": baseline_report["_overall"]["clean_fpr"],
                      "candidate": candidate_report["_overall"]["clean_fpr"]},
        "experts": {name: round(value["f1"], 4)
                    for name, value in result["candidate_experts"].items()},
        "feedback_effect": result["feedback_effect"],
        "all_families": result["all_families_at_or_above_080"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
