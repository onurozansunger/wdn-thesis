"""Fit the feedback-guided evidence router on generator-held TRAIN OOF scores.

The specialist scores are genuinely OOF by generator source. The observable
feature bank is aligned row-for-row and supplies evidence, never family labels,
at inference. Calibration, validation, EVAL and test are not read here.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from wdn.evidence_feedback import (FeedbackRouterBundle, ROUTER_CLASSES,
    SCORE_EVIDENCE_NAMES, aggregate_router_evidence, balanced_weights,
    contiguous_groups, group_values, local_evidence, router_targets)


ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "runs/operational/seasonal_family_deployment_v2/full/features_train.npz"
SEASONAL = ROOT / "runs/operational/seasonal_family_deployment_v2/full/seasonal_train.npz"
NAMES = ROOT / "runs/operational/seasonal_family_deployment_v2/feature_names.json"
SEASONAL_NAMES = ROOT / "runs/operational/seasonal_family_deployment_v2/seasonal_feature_names.json"
SEASONAL_OOF = ROOT / "runs/operational/seasonal_family_experts_v3/oof_predictions.npz"
DELAYED_OOF = ROOT / "runs/operational/delayed_decision_head_v1/oof_predictions.npz"
PROTOCOL = ROOT / "thesis_v2/FEEDBACK_ROUTER_EVAL3_PROTOCOL.md"
OUTPUT = ROOT / "runs/operational/feedback_router_v2"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def model(seed, *, multiclass=False):
    return HistGradientBoostingClassifier(max_iter=180, max_leaf_nodes=23,
        min_samples_leaf=30, learning_rate=.06, l2_regularization=12.,
        early_stopping=False, random_state=seed,
        class_weight=None if multiclass else None)


def feedback_sample(labels, families, base_score, family_id, rng):
    labels = np.asarray(labels) > 0
    families = np.asarray(families)
    positive = np.flatnonzero(labels & (families == family_id))
    other_positive = np.flatnonzero(labels & (families != family_id))
    negative = np.flatnonzero(~labels)
    order = negative[np.argsort(-np.asarray(base_score)[negative], kind="stable")]
    hard = order[:min(60000, len(order))]
    replay_negative = np.flatnonzero((~labels) & (families == 2))
    if len(replay_negative) > 30000:
        replay_negative = rng.choice(replay_negative, 30000, replace=False)
    remaining = np.setdiff1d(negative, np.union1d(hard, replay_negative), assume_unique=False)
    random_negative = rng.choice(remaining, min(40000, len(remaining)), replace=False)
    return np.unique(np.r_[positive, other_positive, hard, replay_negative, random_negative])


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / "campaign.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUTPUT / "bundle.joblib").exists():
        print("Feedback router already fitted; no refit", flush=True)
        return
    started = time.monotonic()

    base = dict(np.load(FEATURES))
    seasonal_X = np.load(SEASONAL)["X"]
    names = json.loads(NAMES.read_text()) + json.loads(SEASONAL_NAMES.read_text())
    X = np.column_stack((base["X"], seasonal_X)).astype(np.float32)
    seasonal = dict(np.load(SEASONAL_OOF))
    delayed = dict(np.load(DELAYED_OOF))
    for key in ("labels", "families", "scenario", "source", "timestep", "node", "event"):
        if not np.array_equal(base[key], seasonal[key]) or not np.array_equal(base[key], delayed[key]):
            raise ValueError(f"TRAIN OOF row alignment failed for {key}")
    if not np.array_equal(seasonal["scores"], delayed["frozen"]):
        raise ValueError("Seasonal and delayed OOF specialist scores differ")

    score_parts = {
        "frozen_drift": delayed["frozen"][:, 0],
        "frozen_noise": delayed["frozen"][:, 1],
        "maxpool_drift": delayed["maxpool"][:, 0],
        "maxpool_noise": delayed["maxpool"][:, 1],
        "delayed_drift": delayed["delayed"][:, 0],
        "delayed_noise": delayed["delayed"][:, 1],
        "final_drift": delayed["delayed_blend"][:, 0],
        "final_noise": delayed["delayed_blend"][:, 1],
    }
    if tuple(score_parts) != SCORE_EVIDENCE_NAMES:
        raise AssertionError("Score evidence order changed")
    local = local_evidence(X, names, score_parts)
    arrays = {key: base[key] for key in ("source", "scenario", "timestep")}
    group, starts = contiguous_groups(arrays)
    aggregate = aggregate_router_evidence(local, group, starts)
    group_family = group_values(base["families"], starts, "family")
    group_source = group_values(base["source"], starts, "source")
    target = router_targets(group_family)

    # Source-held router predictions become honest inputs to the feedback heads.
    router_oof_group = np.zeros((len(starts), len(ROUTER_CLASSES)), dtype=np.float32)
    for fold, source in enumerate(np.unique(group_source)):
        train = group_source != source
        held = ~train
        fold_model = model(6100 + fold, multiclass=True)
        fold_model.fit(aggregate[train], target[train],
                       sample_weight=balanced_weights(target[train]))
        if not np.array_equal(fold_model.classes_, np.arange(len(ROUTER_CLASSES))):
            raise ValueError("A router fold is missing a declared class")
        router_oof_group[held] = fold_model.predict_proba(aggregate[held])
        print("router fold", int(source), "groups", int(held.sum()), flush=True)

    router = model(6200, multiclass=True)
    router.fit(aggregate, target, sample_weight=balanced_weights(target))
    router_oof = router_oof_group[group]
    feedback_X = np.column_stack((local, router_oof)).astype(np.float32)
    rng = np.random.default_rng(6300)
    feedback_models = []
    sample_sizes = {}
    for column, (name, family_id) in enumerate((("drift", 3), ("noise", 4))):
        selected = feedback_sample(base["labels"], base["families"],
                                   delayed["delayed_blend"][:, column], family_id, rng)
        y = ((base["labels"][selected] > 0)
             & (base["families"][selected] == family_id)).astype(int)
        fitted = model(6400 + column)
        fitted.fit(feedback_X[selected], y, sample_weight=balanced_weights(y))
        feedback_models.append(fitted)
        sample_sizes[name] = {"rows": int(len(selected)), "positive": int(y.sum())}
        print("feedback", name, sample_sizes[name], flush=True)

    bundle = FeedbackRouterBundle(tuple(names), router, feedback_models[0], feedback_models[1])
    joblib.dump(bundle, OUTPUT / "bundle.joblib")
    router_prediction = router_oof_group.argmax(1)
    router_by_family = {}
    for code, name in enumerate(ROUTER_CLASSES):
        selected = target == code
        router_by_family[name] = {
            "groups": int(selected.sum()),
            "accuracy": float(np.mean(router_prediction[selected] == code)),
            "mean_probability": float(router_oof_group[selected, code].mean()),
        }
    summary = {
        "status": "completed",
        "scope": "62-scenario generator-held TRAIN OOF; no calibration/EVAL/test",
        "architecture": "graph-time evidence router plus row-level drift/noise feedback",
        "router_classes": ROUTER_CLASSES,
        "local_feature_count": int(local.shape[1]),
        "router_feature_count": int(aggregate.shape[1]),
        "train_rows": int(len(local)),
        "train_groups": int(len(starts)),
        "sources": [int(x) for x in np.unique(base["source"])],
        "router_oof_by_family": router_by_family,
        "feedback_training": sample_sizes,
        "uses_labels_at_inference": False,
        "uses_event_boundaries_at_inference": False,
        "declared_decision_latency_hours": 3,
        "calibration_evaluated": False,
        "eval1_evaluated": False,
        "eval2_evaluated": False,
        "eval3_evaluated": False,
        "test_evaluated": False,
        "source_sha256": {
            "features": sha(FEATURES), "seasonal": sha(SEASONAL),
            "seasonal_oof": sha(SEASONAL_OOF), "delayed_oof": sha(DELAYED_OOF),
            "protocol": sha(PROTOCOL),
        },
        "bundle_sha256": sha(OUTPUT / "bundle.joblib"),
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(OUTPUT / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
