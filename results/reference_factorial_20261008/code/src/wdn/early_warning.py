"""Strictly causal, network-level early family warning.

The deployed evidence router cannot be renamed an early router: it consumes
``maxpool_*``, ``delayed_*`` and ``final_*`` scores, all of which read
observations inside ``(t, t + 3]``. This module builds a separate input path
that reads nothing after ``t``.

Two things are deliberately kept apart:

* **the warning clock** — this head is scored against the *true attack onset*,
  which only the evaluator knows;
* **the decision clock** — the specialists still finalise the decision for hour
  ``t`` at ``t + 3``.

The head is *advisory*. It never gates a specialist: an absent warning must not
be able to suppress a later detection, and the general mixture's immediate alarm
path stays untouched. Nothing here reads a label, an event boundary, an attack
parameter, a target identity, or any observation after ``t`` at inference.

Resolution is graph-time: one warning per ``(source, scenario, timestep)``. That
is a network-level family warning, not sensor localisation, and it is reported
as such.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wdn.evidence_feedback import (RAW_EVIDENCE_NAMES, ROUTER_CLASSES,
                                   aggregate_router_evidence, contiguous_groups,
                                   feature_columns)

#: The two strictly causal specialist scores. These are the un-pooled,
#: un-delayed drift/noise outputs of ``latency_deployment.specialist_scores``.
CAUSAL_SCORE_NAMES = ("causal_drift", "causal_noise")

#: Every local channel the early head may see at time ``t``.
EARLY_LOCAL_NAMES = tuple(RAW_EVIDENCE_NAMES) + CAUSAL_SCORE_NAMES

#: Past-only graph-time lags, in hours.
HISTORY_LAGS = (1, 2, 3)

#: Channels whose graph-time mean gets a past-only difference at each lag. Kept
#: short on purpose: the graph-time corpus has thousands of rows, not millions,
#: so a delta for all 39 channels would be mostly noise.
HISTORY_CHANNELS = (
    "residual", "abs_residual", "dynamic_abs_innovation",
    "drift_cusum_positive_0.5", "drift_cusum_negative_0.5",
    "noise_state_4.0", "noise_state_9.0", "drift_consistency_3",
    "drift_ramp_strength_3", "noise_energy_3", "seasonal_abs_delta_24",
    "causal_drift", "causal_noise",
)

#: How the five evidence classes are reported to an operator.
REPORT_CLASSES = ("normal", "other_mechanism", "likely_drift", "likely_noise")
CLASS_TO_REPORT = {"clean": "normal", "abrupt": "other_mechanism",
                   "replay": "other_mechanism", "drift": "likely_drift",
                   "noise": "likely_noise"}


def early_local_evidence(X, names, causal_scores):
    """Local causal evidence per observed endpoint: 37 bank columns + 2 scores."""
    raw = np.asarray(X)[:, feature_columns(names)]
    scores = np.asarray(causal_scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != len(CAUSAL_SCORE_NAMES):
        raise ValueError(f"Causal scores must have shape (n, {len(CAUSAL_SCORE_NAMES)})")
    if len(raw) != len(scores):
        raise ValueError("Bank and causal scores have different row counts")
    return np.column_stack((raw, scores)).astype(np.float32)


def group_keys(arrays, starts):
    """The `(source, scenario, timestep)` triple identifying each graph-time group."""
    return np.column_stack([np.asarray(arrays[key])[starts]
                            for key in ("source", "scenario", "timestep")]).astype(np.int64)


def past_lag_index(keys, lags=HISTORY_LAGS):
    """Group index of ``t - k`` for each group and each ``k``, -1 when absent.

    Matched by timestep inside the same ``(source, scenario)``, never by
    position: at a 0.50 missing rate a graph-time hour can be entirely absent,
    and a positional shift would quietly turn a gap into a one-hour lag.
    """
    keys = np.asarray(keys, dtype=np.int64)
    base = 1 << 20
    if keys[:, 2].min() < 0 or keys[:, 2].max() >= base:
        raise ValueError("Timestep does not fit the composite group index")
    composite = (keys[:, 0] * 10**9 + keys[:, 1] * 10**5) * base + keys[:, 2]
    order = np.argsort(composite, kind="stable")
    ordered = composite[order]
    if np.any(np.diff(ordered) == 0):
        raise ValueError("Duplicate (source, scenario, timestep) graph-time groups")
    index = np.full((len(keys), len(lags)), -1, dtype=np.int64)
    for column, lag in enumerate(lags):
        wanted = composite - int(lag)
        position = np.searchsorted(ordered, wanted)
        inside = position < len(ordered)
        found = np.zeros(len(keys), dtype=bool)
        found[inside] = ordered[position[inside]] == wanted[inside]
        index[found, column] = order[position[found]]
    return index


def early_group_features(local, arrays, lags=HISTORY_LAGS):
    """Graph-time features for the early head, plus their names and row mapping.

    Returns ``(features, names, group, starts)``. ``group`` maps every endpoint
    row to its graph-time group; ``starts`` are the group start rows.
    """
    group, starts = contiguous_groups(arrays)
    aggregate = aggregate_router_evidence(local, group, starts)
    channels = len(EARLY_LOCAL_NAMES)
    if aggregate.shape[1] != 3 * channels:
        raise ValueError("Aggregate width does not match the declared channel list")
    names = [f"{stat}_{name}" for stat in ("mean", "std", "max")
             for name in EARLY_LOCAL_NAMES]

    counts = np.diff(np.r_[starts, len(local)]).astype(np.float32)
    blocks = [aggregate, counts[:, None]]
    names.append("observed_sensors")

    keys = group_keys(arrays, starts)
    index = past_lag_index(keys, lags)
    lookup = {name: i for i, name in enumerate(EARLY_LOCAL_NAMES)}
    history = [lookup[name] for name in HISTORY_CHANNELS]
    mean_block = aggregate[:, :channels]
    for column, lag in enumerate(lags):
        present = index[:, column] >= 0
        delta = np.full((len(starts), len(history)), np.nan, dtype=np.float32)
        delta[present] = (mean_block[present][:, history]
                          - mean_block[index[present, column]][:, history])
        blocks.append(delta)
        names.extend(f"delta{lag}_mean_{name}" for name in HISTORY_CHANNELS)
        blocks.append(present.astype(np.float32)[:, None])
        names.append(f"has_lag{lag}")

    features = np.column_stack(blocks).astype(np.float32)
    if features.shape[1] != len(names):
        raise AssertionError("Early feature matrix and name list disagree")
    return features, names, group, starts


def group_family_state(families, starts):
    """The evidence class active at each graph-time group; targeted joins abrupt."""
    families = np.asarray(families, dtype=int)
    counts = np.diff(np.r_[starts, len(families)])
    state = np.zeros(len(starts), dtype=int)
    for position, (start, count) in enumerate(zip(starts, counts)):
        block = families[start:start + count]
        active = block[block != 0]
        state[position] = 0 if not len(active) else int(np.bincount(active).argmax())
    state[state == 5] = 1
    if np.any((state < 0) | (state >= len(ROUTER_CLASSES))):
        raise ValueError("Unknown family id in the early-warning target")
    return state


@dataclass
class EarlyWarningHead:
    """A fitted early head plus the abstention rule frozen with it."""

    names: tuple[str, ...]
    model: object
    abstain_threshold: float
    lags: tuple[int, ...] = HISTORY_LAGS

    def predict_groups(self, local, arrays):
        features, names, group, starts = early_group_features(local, arrays, self.lags)
        if tuple(names) != tuple(self.names):
            raise ValueError("Early feature schema differs from the fitted head")
        probabilities = self.model.predict_proba(features)
        if not np.array_equal(self.model.classes_, np.arange(len(ROUTER_CLASSES))):
            raise RuntimeError("Early head was not fitted on all declared classes")
        confidence = probabilities.max(axis=1)
        prediction = probabilities.argmax(axis=1)
        abstain = confidence < float(self.abstain_threshold)
        report = np.array([CLASS_TO_REPORT[ROUTER_CLASSES[c]] for c in prediction])
        report[abstain] = "abstain"
        return {"probabilities": probabilities, "prediction": prediction,
                "confidence": confidence, "abstain": abstain, "report": report,
                "group": group, "starts": starts,
                "keys": group_keys(arrays, starts)}

    def warned(self, result, family):
        """Groups warned for a named family, abstentions excluded."""
        target = {"drift": 3, "noise": 4}[family]
        return (result["prediction"] == target) & ~result["abstain"]

    def any_warning(self, result):
        """Groups carrying any non-normal, non-abstaining warning."""
        return (result["prediction"] != 0) & ~result["abstain"]
