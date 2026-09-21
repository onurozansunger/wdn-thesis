"""Whole-detector scoring under a declared decision latency.

Two frozen models are applied to fresh generator seeds:

* the old residual expert mixture, on the 29-column rank-16 bank,
* the promoted seasonal drift/noise specialists, on the 109-column expanded
  bank plus the seven daily seasonal features.

Both references were fitted on TRAIN and are loaded frozen; nothing here fits,
calibrates or rescales anything on the data it scores. The latency operator is
a forward maximum over ``[t, t + delta]`` inside each sensor series, matched by
timestep so that a gap does not silently refuse a neighbour.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from wdn.delayed_decision_features import offset_index
from wdn.probe_blind_reference import residual_features
from wdn.run_expert_redesign import CampaignData
from wdn.seasonal_pressure_features import endpoint_seasonal_features

FAMILY_NAMES = {1: "random", 2: "replay", 3: "drift", 4: "noise", 5: "targeted"}


def logit(value):
    value = np.clip(np.asarray(value, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(value / (1 - value))


def logit_blend(first, second, first_weight):
    return expit(np.clip(first_weight * logit(first) +
                         (1 - first_weight) * logit(second), -30, 30))


def mixture_bank(data: CampaignData, scenarios, reference):
    """The 29-column rank-16 bank, exactly as the old mixture was trained on."""
    collected = {key: [] for key in ("X", "labels", "families", "scenario", "timestep", "node")}
    for sid in sorted(scenarios):
        arrays = data.scenario(sid)
        values, mask = arrays["values"], arrays["mask"]
        prediction, support = reference.predict(values, mask)
        features, _ = residual_features(values, mask, prediction, support,
                                        reference.noise_scale_, window=16)
        endpoint = mask[15:]
        labels = arrays["labels"][15:]
        times = arrays["timestep"][15:]
        family = np.broadcast_to(arrays["families"][15:, None], endpoint.shape)
        node = np.broadcast_to(np.arange(labels.shape[1]), labels.shape)
        collected["X"].append(features[endpoint])
        collected["labels"].append(labels[endpoint])
        collected["families"].append(family[endpoint])
        collected["scenario"].append(np.full(int(endpoint.sum()), sid))
        collected["timestep"].append(np.broadcast_to(times[:, None], endpoint.shape)[endpoint])
        collected["node"].append(node[endpoint])
    return {key: np.concatenate(value) for key, value in collected.items()}


def specialist_bank(data: CampaignData, scenarios, reference):
    """The 109-column expanded bank plus the seven daily seasonal features."""
    arrays, names = data.features(scenarios, reference)
    seasonal, seasonal_names = endpoint_seasonal_features(arrays, data.scenario)
    arrays["X"] = np.column_stack((arrays["X"], seasonal))
    return arrays, list(names) + list(seasonal_names)


def specialist_scores(bundle, X):
    """Promoted drift and noise scores, with the frozen 0.90 / 0.95 blends."""
    base = X[:, :109]
    seasonal = bundle["seasonal"].predict(X)
    tuned = bundle["tuned_drift"].predict(base)[:, 0]
    return np.column_stack((logit_blend(seasonal["drift"], tuned, .90),
                            logit_blend(seasonal["noise_fast"], seasonal["noise_full"], .95)))


def forward_max(score, arrays, delta):
    """max(score_t .. score_{t+delta}) inside each sensor series, by timestep."""
    if delta == 0:
        return np.asarray(score, dtype=float)
    index = offset_index(arrays, range(1, delta + 1))
    pooled = np.asarray(score, dtype=float).copy()
    for column in range(delta):
        present = index[:, column] >= 0
        pooled[present] = np.maximum(pooled[present], np.asarray(score)[index[present, column]])
    return pooled


def family_scores(decision, arrays):
    """Per-family F1 with false positives counted inside that family's rows."""
    labels = np.asarray(arrays["labels"]) > 0
    families = np.asarray(arrays["families"])
    result = {}
    for code, name in FAMILY_NAMES.items():
        scope = families == code
        if not scope.any():
            continue
        y, p = labels[scope], decision[scope]
        tp = int(np.sum(y & p)); fp = int(np.sum(~y & p)); fn = int(np.sum(y & ~p))
        result[name] = {"f1": 2 * tp / max(1, 2 * tp + fp + fn),
                        "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
                        "tp": tp, "fp": fp, "fn": fn, "positives": int(y.sum())}
    clean = families == 0
    negatives = ~labels
    result["_overall"] = {
        "f1": 2 * int(np.sum(labels & decision)) / max(1, 2 * int(np.sum(labels & decision))
              + int(np.sum(~labels & decision)) + int(np.sum(labels & ~decision))),
        "clean_fpr": float(decision[clean].mean()) if clean.any() else 0.,
        "all_negative_fpr": float(decision[negatives].mean()),
        "worst_family_f1": min(entry["f1"] for key, entry in result.items()
                               if not key.startswith("_"))}
    return result


def quantile_threshold(score, keep, budget):
    """Highest threshold whose false-positive rate on `keep` stays within budget."""
    values = np.sort(np.asarray(score, dtype=float)[keep])
    if not len(values):
        raise ValueError("A false-alarm budget needs negative rows")
    if budget <= 0:
        return float(np.nextafter(values[-1], np.inf))
    position = int(np.floor((1 - budget) * len(values)))
    position = min(max(position, 0), len(values) - 1)
    return float(values[position])
