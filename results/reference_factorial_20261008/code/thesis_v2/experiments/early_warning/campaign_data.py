"""Shared loaders for the early-warning campaign's development surfaces.

Two surfaces, and the difference matters for every claim made from them:

* :func:`load_train_oof` — 62 TRAIN scenarios over four generator sources. Each
  source-held fold fitted **its own** normal reference and **its own** experts,
  so both the bank features and the causal specialist scores are full-pipeline
  out-of-fold. This is the only surface on which a new head may be fitted.

* :func:`load_calibration` — 99 calibration scenarios scored by the deployed
  models, which never saw these rows. Selection happens here. The deployed
  *thresholds* were chosen on these same rows, so any false-positive count
  quoted at those thresholds is optimistically biased.

Neither loader touches the locked test or the locked EVAL corpora.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from build_feature_cache import ROOT, load_corpus

OOF_FOLDS = ROOT / "runs/operational/expanded_train_weak_experts_v1"
SEASONAL_FOLDS = ROOT / "runs/operational/seasonal_family_experts_v3"
DELAYED_OOF = ROOT / "runs/operational/delayed_decision_head_v1/oof_predictions.npz"
BASE_SCORES = ROOT / "runs/operational/cross_mechanism_deployment_v1/scores_calibration.npz"
FEEDBACK_SCORES = ROOT / "runs/operational/feedback_router_calibration_v2/feedback_calibration.npz"
SEASONAL_NAMES = ROOT / "runs/operational/seasonal_family_deployment_v2/seasonal_feature_names.json"
DATA = ROOT / "data/thesis_v2"

META_KEYS = ("labels", "families", "event", "scenario", "source", "timestep", "node")

TRAIN_OOF_SOURCES = (811, 1811, 2811, 3811)
CALIBRATION_PIECES = (("operational_modena_seed811", 811),
                      ("operational_calibration_expansion_seed12811", 12811),
                      ("operational_calibration_expansion_seed13811", 13811),
                      ("operational_calibration_expansion_seed14811", 14811),
                      ("operational_calibration_expansion_seed15811", 15811))


def load_train_oof():
    """Full-pipeline TRAIN out-of-fold corpus: 116-column bank plus causal scores."""
    base_names = json.loads((OOF_FOLDS / "feature_names.json").read_text())
    seasonal_names = json.loads(SEASONAL_NAMES.read_text())
    names = list(base_names) + list(seasonal_names)
    parts = []
    for fold in range(4):
        with np.load(OOF_FOLDS / f"fold_{fold}/features_held_out.npz") as loaded:
            entry = {key: loaded[key] for key in META_KEYS}
            base = loaded["X"]
        with np.load(SEASONAL_FOLDS / f"fold_{fold}/seasonal_held_out.npz") as loaded:
            seasonal = loaded["X"]
        entry["X"] = np.column_stack((base, seasonal)).astype(np.float32)
        parts.append(entry)
    corpus = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}

    delayed = dict(np.load(DELAYED_OOF))
    for key in ("labels", "families", "scenario", "source", "timestep", "node"):
        if not np.array_equal(np.asarray(corpus[key]), np.asarray(delayed[key])):
            raise ValueError(f"TRAIN OOF fold caches disagree with the OOF scores on {key}")
    corpus["causal"] = np.asarray(delayed["frozen"], dtype=float)
    corpus["delayed"] = np.asarray(delayed["delayed"], dtype=float)
    corpus["delayed_blend"] = np.asarray(delayed["delayed_blend"], dtype=float)
    corpus["maxpool"] = np.asarray(delayed["maxpool"], dtype=float)
    corpus["names"] = names
    corpus["events"] = events_by_scenario([("operational_modena_seed811", 811)]
                                          + [(f"operational_train_expansion_seed{s}", s)
                                             for s in (1811, 2811, 3811)])
    if set(np.unique(corpus["source"]).tolist()) != set(TRAIN_OOF_SOURCES):
        raise ValueError("TRAIN OOF sources are not the four declared generator seeds")
    return corpus


def load_calibration():
    """Deployed-model calibration corpus with every branch score already frozen."""
    corpus, manifest = load_corpus("modena_calibration")
    base = dict(np.load(BASE_SCORES))
    meta = dict(np.load(FEEDBACK_SCORES))
    for key in ("labels", "families", "timestep", "node", "scenario", "source"):
        if not np.array_equal(np.asarray(base[key]), np.asarray(corpus[key])):
            raise ValueError(f"Calibration cache and frozen scores disagree on {key}")
        if not np.array_equal(np.asarray(meta[key]), np.asarray(corpus[key])):
            raise ValueError(f"Calibration cache and feedback scores disagree on {key}")
    corpus["names"] = manifest["feature_names"]
    corpus["causal"] = np.asarray(base["promoted"], dtype=float)
    corpus["mixture"] = np.asarray(base["mixture"], dtype=float)
    corpus["promoted_head"] = np.asarray(base["promoted_head"], dtype=float)
    corpus["router"] = np.asarray(meta["router"], dtype=float)
    corpus["feedback"] = np.asarray(meta["feedback"], dtype=float)
    corpus["events"] = events_by_scenario(CALIBRATION_PIECES)
    corpus["scenarios"] = manifest["total_scenarios"]
    return corpus


def events_by_scenario(pieces, data_root=DATA):
    """Event ledgers keyed by global scenario id: evaluator metadata, never a feature."""
    result = {}
    for name, seed in pieces:
        for event in json.loads((data_root / name / "events.json").read_text()):
            key = int(event["scenario_id"]) + seed * 1000
            entry = {"scenario": key, "source": seed,
                     "family": {"stealthy": "drift"}.get(event["family"], event["family"]),
                     "start": int(event["start_timestep"]),
                     "steps": int(event["actual_steps"])}
            result.setdefault(key, []).append(entry)
    for events in result.values():
        events.sort(key=lambda e: e["start"])
    return result


def restrict_events(events, corpus):
    """Drop ledger entries for scenarios that are not in this corpus."""
    present = set(np.unique(np.asarray(corpus["scenario"])).tolist())
    return {key: value for key, value in events.items() if key in present}
