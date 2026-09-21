"""Shared, strictly causal history for the existing five tree experts.

No labels, clean targets, event boundaries, or future observations are read.
History is matched by source/scenario/sensor/hour, not by observed-row position.
This preserves trajectories of existing evidence; it does not reconstruct the
individual lag scores already discarded by the original feature extractor.
"""
from __future__ import annotations

import numpy as np

from wdn.probe_residual_experts import ResidualExpertMixture

HISTORY_OFFSETS = (1, 2, 3, 4, 5, 6, 8, 12)
HISTORY_COLUMNS = ("residual", "seq_innovation", "dynamic_innovation",
                   "lag_advantage", "reference_support")


def causal_history_features(arrays, names, rows=None):
    """Return past evidence, residual changes and explicit availability masks.

    ``rows`` optionally selects query endpoints without dropping their history.
    Missing history is NaN (not a zero residual). Cached banks start at hour 15;
    unavailable earlier endpoints stay missing in this pilot, in both splits.
    Structured keys avoid overflow from packing large generator seed IDs.
    """
    lookup = {name: i for i, name in enumerate(names)}
    missing = set(HISTORY_COLUMNS) - lookup.keys()
    if missing:
        raise ValueError(f"Missing history columns: {sorted(missing)}")
    n = len(arrays["X"])
    rows = np.arange(n) if rows is None else np.asarray(rows, dtype=np.int64)
    fields = ("source", "scenario", "node", "timestep")
    keys = np.empty(n, dtype=[(field, np.int64) for field in fields])
    for field in fields:
        keys[field] = arrays[field]
    order = np.argsort(keys, order=fields, kind="stable")
    ordered = keys[order]
    if np.any(ordered[1:] == ordered[:-1]):
        raise ValueError("Duplicate source/scenario/node/timestep endpoints")
    result = np.full((len(rows), len(HISTORY_OFFSETS) * 7), np.nan, np.float32)
    built_names = []
    current = arrays["X"][rows, lookup["residual"]]
    for k, lag in enumerate(HISTORY_OFFSETS):
        query = keys[rows].copy()
        query["timestep"] -= lag
        pos = np.searchsorted(ordered, query)
        present = pos < n
        present[present] &= ordered[pos[present]] == query[present]
        previous = order[pos[present]]
        for j, name in enumerate(HISTORY_COLUMNS):
            result[present, k * 7 + j] = arrays["X"][previous, lookup[name]]
            built_names.append(f"history_{name}_lag_{lag}")
        result[present, k * 7 + 5] = current[present] - result[present, k * 7]
        result[:, k * 7 + 6] = present.astype(np.float32)
        built_names.extend((f"history_residual_change_{lag}", f"history_available_{lag}"))
    return result, built_names


class SharedHistoryExpertMixture(ResidualExpertMixture):
    """Same experts and router; every expert gets the common history bank."""

    def __init__(self, names, seed=601):
        super().__init__(names, seed)
        shared = np.array([i for i, name in enumerate(names)
                           if name.startswith("history_")], dtype=int)
        self.profiles = [np.union1d(cols, shared) for cols in self.profiles]


def fit_presampled(model, X, labels, families, weights):
    """Original mixture's objective on its exact preselected training sample.

    Sampling outside fit lets the pilot compute history only at training query
    rows, while retaining the entire observed trajectory as context. The clean
    population weights must be supplied; sample prevalence is not substituted.
    """
    positive = labels > .5
    target = np.zeros(len(labels), dtype=int)
    for mechanism, ids in enumerate(((1, 5), (2,), (3,), (4,)), start=1):
        target[positive & np.isin(families, ids)] = mechanism
    model.experts = []
    for mechanism, cols in enumerate(model.profiles):
        domain = np.ones(len(labels), bool) if mechanism == 0 else ((target == mechanism) | ~positive)
        y = labels[domain]
        weight = weights[domain].copy()
        if len(np.unique(y)) != 2:
            raise ValueError(f"Missing class for mechanism {mechanism}")
        if mechanism:
            for cls in (0, 1):
                weight[y == cls] *= len(weight) / (2 * weight[y == cls].sum())
        model.experts.append(model._model().fit(X[domain][:, cols], y, sample_weight=weight))
        print(f"trained mechanism={mechanism} positives={int(y.sum())}", flush=True)
    route_weight = weights.copy()
    for mechanism in range(5):
        selected = target == mechanism
        if not selected.any():
            raise ValueError(f"Missing router class {mechanism}")
        route_weight[selected] *= len(route_weight) / (5 * route_weight[selected].sum())
    model.router = model._model().fit(X, target, sample_weight=route_weight)
    return model
