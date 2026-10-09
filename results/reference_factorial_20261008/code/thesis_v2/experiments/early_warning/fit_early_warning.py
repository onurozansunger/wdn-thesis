"""Stage B: fit the strictly causal early-warning head on TRAIN out-of-fold.

The head is a single small gradient-boosted tree over graph-time aggregates of
causal evidence. It is not a neural search: a standalone GRU already failed the
prior OOF screen (``thesis_v2/CAUSAL_SEQUENCE_EXPERT_RESULTS.md``), and nothing
in the audit suggests sequence capacity is what is missing.

Training data is the 62-scenario TRAIN out-of-fold corpus, where both the bank
features and the causal specialist scores come from source-held folds that each
fitted their own normal reference. The head's *own* honesty comes from a second
source-held pass: predictions for source ``s`` come from a head that never saw
``s``.

The abstention threshold is chosen here, on TRAIN out-of-fold only, as the
smallest value in a fixed grid whose clean-network-hour false-warning rate stays
within the declared budget. Calibration is not read by this script.

    /opt/miniconda3/bin/python thesis_v2/experiments/early_warning/fit_early_warning.py
"""
from __future__ import annotations

import fcntl
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from wdn.early_warning import (EarlyWarningHead, early_group_features,
                               early_local_evidence, group_family_state)
from wdn.early_warning_metrics import (clean_hour_false_alarms, confusion,
                                       event_warnings, summarise_events,
                                       warning_episodes)
from wdn.evidence_feedback import ROUTER_CLASSES, balanced_weights

from build_feature_cache import ROOT, sha, write_json
from campaign_data import load_train_oof

OUTPUT = ROOT / "runs/operational/early_warning_multiseed_v1/early_head_v1"

#: Fixed before fitting.
MODEL = dict(max_iter=250, max_leaf_nodes=15, min_samples_leaf=20,
             learning_rate=.06, l2_regularization=10., early_stopping=False)

#: Graph-time hours inside the first ``EARLY_PHASE_HOURS`` of an event are the
#: ones the supervisor actually cares about, so they are weighted up. This
#: changes what the head optimises; it does not change any denominator.
EARLY_PHASE_HOURS = 3
EARLY_PHASE_WEIGHT = 3.0

#: Abstention grid and budget, both frozen before calibration is read.
ABSTAIN_GRID = (.30, .35, .40, .45, .50, .55, .60, .65, .70)
CLEAN_WARNING_BUDGET = .05


def build_surface(corpus):
    local = early_local_evidence(corpus["X"], corpus["names"], corpus["causal"])
    features, names, group, starts = early_group_features(local, corpus)
    target = group_family_state(corpus["families"], starts)
    keys = np.column_stack([np.asarray(corpus[key])[starts]
                            for key in ("source", "scenario", "timestep")])
    return {"features": features, "names": names, "target": target,
            "group": group, "starts": starts, "keys": keys,
            "source": keys[:, 0], "scenario": keys[:, 1], "timestep": keys[:, 2]}


def early_phase_weights(surface, events):
    """Extra weight on the opening hours of each event, on the onset clock."""
    weight = balanced_weights(surface["target"])
    early = np.zeros(len(weight), dtype=bool)
    for scenario, entries in events.items():
        selected = surface["scenario"] == scenario
        if not selected.any():
            continue
        for event in entries:
            early |= selected & (surface["timestep"] >= event["start"]) & (
                surface["timestep"] < event["start"] + EARLY_PHASE_HOURS)
    weight = weight * np.where(early & (surface["target"] != 0), EARLY_PHASE_WEIGHT, 1.)
    return weight, int(early.sum())


def fit_model(features, target, weight, seed):
    model = HistGradientBoostingClassifier(random_state=seed, **MODEL)
    model.fit(features, target, sample_weight=weight)
    if not np.array_equal(model.classes_, np.arange(len(ROUTER_CLASSES))):
        raise ValueError("A fold is missing one of the five declared evidence classes")
    return model


def choose_abstention(probabilities, target, keys, events):
    """Smallest grid threshold whose clean-hour false-warning rate meets budget."""
    prediction = probabilities.argmax(1)
    confidence = probabilities.max(1)
    open_event = np.zeros(len(target), dtype=bool)
    for scenario, entries in events.items():
        selected = keys[:, 1] == scenario
        for event in entries:
            open_event |= selected & (keys[:, 2] >= event["start"]) & (
                keys[:, 2] < event["start"] + event["steps"])
    clean = ~open_event
    rows = []
    chosen = None
    for threshold in ABSTAIN_GRID:
        warned = (prediction != 0) & (confidence >= threshold)
        rate = float(np.mean(warned[clean])) if clean.any() else 0.
        recall = float(np.mean(warned[~clean])) if (~clean).any() else 0.
        rows.append({"threshold": threshold, "clean_false_warning_rate": rate,
                     "event_hour_warning_rate": recall})
        if chosen is None and rate <= CLEAN_WARNING_BUDGET:
            chosen = threshold
    if chosen is None:
        chosen = ABSTAIN_GRID[-1]
    return chosen, rows, {"clean_graph_hours": int(clean.sum()),
                          "event_graph_hours": int((~clean).sum())}


def report(head, corpus, surface, events, label):
    local = early_local_evidence(corpus["X"], corpus["names"], corpus["causal"])
    result = head.predict_groups(local, corpus)
    rows = event_warnings(result, events)
    return {
        "surface": label,
        "graph_hours": int(len(surface["starts"])),
        "events_scored": len(rows),
        "by_family": summarise_events(rows),
        "confusion": confusion(result, corpus["families"], surface["starts"]),
        "episodes": warning_episodes(result),
        "clean_hour_false_alarms": clean_hour_false_alarms(
            result, corpus["families"], surface["starts"], events),
        "clean_hour_false_alarms_excluding_6h_tail": clean_hour_false_alarms(
            result, corpus["families"], surface["starts"], events, span_hours=6),
        "per_event": rows,
    }, result


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / "campaign.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUTPUT / "bundle.joblib").exists():
        print("Early head already fitted; no refit", flush=True)
        return
    started = time.monotonic()

    print("loading TRAIN out-of-fold corpus", flush=True)
    corpus = load_train_oof()
    surface = build_surface(corpus)
    events = {key: value for key, value in corpus["events"].items()
              if key in set(surface["scenario"].tolist())}
    weight, early_hours = early_phase_weights(surface, events)
    print(f"graph-time rows {len(surface['starts'])}, features {len(surface['names'])}, "
          f"events {sum(len(v) for v in events.values())}", flush=True)

    sources = np.unique(surface["source"])
    oof = np.zeros((len(surface["target"]), len(ROUTER_CLASSES)), dtype=float)
    for fold, source in enumerate(sources):
        held = surface["source"] == source
        model = fit_model(surface["features"][~held], surface["target"][~held],
                          weight[~held], 7100 + fold)
        oof[held] = model.predict_proba(surface["features"][held])
        print(f"  source-held fold {int(source)}: {int(held.sum())} graph hours", flush=True)

    threshold, grid, counts = choose_abstention(oof, surface["target"], surface["keys"], events)
    print(f"abstention threshold {threshold} (budget {CLEAN_WARNING_BUDGET})", flush=True)

    final = fit_model(surface["features"], surface["target"], weight, 7200)
    head = EarlyWarningHead(tuple(surface["names"]), final, threshold)

    oof_head = EarlyWarningHead(tuple(surface["names"]), None, threshold)
    oof_result = {"probabilities": oof, "prediction": oof.argmax(1),
                  "confidence": oof.max(1), "abstain": oof.max(1) < threshold,
                  "keys": surface["keys"], "starts": surface["starts"],
                  "group": surface["group"]}
    oof_rows = event_warnings(oof_result, events)
    oof_report = {
        "surface": "train_out_of_fold",
        "scope": "62 TRAIN scenarios, four generator-source-held folds; the head "
                 "predicting a source never saw that source",
        "graph_hours": int(len(surface["starts"])),
        "events_scored": len(oof_rows),
        "by_family": summarise_events(oof_rows),
        "confusion": confusion(oof_result, corpus["families"], surface["starts"]),
        "episodes": warning_episodes(oof_result),
        "clean_hour_false_alarms": clean_hour_false_alarms(
            oof_result, corpus["families"], surface["starts"], events),
        "clean_hour_false_alarms_excluding_6h_tail": clean_hour_false_alarms(
            oof_result, corpus["families"], surface["starts"], events, span_hours=6),
        "per_event": oof_rows,
    }
    del oof_head

    joblib.dump({"head": head, "names": list(surface["names"]),
                 "local_names": list(__import__("wdn.early_warning", fromlist=["x"]).EARLY_LOCAL_NAMES),
                 "abstain_threshold": threshold, "model_config": MODEL},
                OUTPUT / "bundle.joblib")
    np.savez_compressed(OUTPUT / "train_oof_predictions.npz",
                        probabilities=oof, keys=surface["keys"],
                        target=surface["target"])

    summary = {
        "status": "completed",
        "scope": "TRAIN out-of-fold only; calibration, validation, locked EVAL and "
                 "the locked test were not read",
        "architecture": "graph-time HistGradientBoosting over causal aggregates",
        "model_config": MODEL,
        "feature_count": len(surface["names"]),
        "local_channels": len(__import__("wdn.early_warning", fromlist=["x"]).EARLY_LOCAL_NAMES),
        "graph_time_rows": int(len(surface["starts"])),
        "sources": [int(s) for s in sources],
        "early_phase_hours": EARLY_PHASE_HOURS,
        "early_phase_weight": EARLY_PHASE_WEIGHT,
        "early_phase_graph_hours": early_hours,
        "abstention": {"grid": list(ABSTAIN_GRID), "budget": CLEAN_WARNING_BUDGET,
                       "selected": threshold, "sweep": grid, **counts},
        "train_oof_report": oof_report,
        "uses_labels_at_inference": False,
        "uses_future_observations_at_inference": False,
        "uses_delayed_or_pooled_scores": False,
        "advisory_only": True,
        "calibration_evaluated": False, "eval_evaluated": False, "test_evaluated": False,
        "bundle_sha256": sha(OUTPUT / "bundle.joblib"),
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(OUTPUT / "summary.json", summary)
    print(json.dumps({
        "abstain_threshold": threshold,
        "train_oof": {
            family: {k: round(v, 4) if isinstance(v, float) else v
                     for k, v in entry.items()
                     if k.startswith(("warned_by", "correct_family_by", "ever_"))
                     or k == "events"}
            for family, entry in oof_report["by_family"].items()},
        "clean_hour_false_warning_rate": oof_report["clean_hour_false_alarms"]["rate"],
        "warning_episodes": oof_report["episodes"]["episodes"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
