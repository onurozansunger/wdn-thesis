"""Bounded-future evidence for a declared decision latency.

Every feature here reads observations strictly inside ``[t, t + delta]`` of the
same sensor series. That is not causal in the strict online sense, and it is not
meant to be: the campaign declares that the decision for hour ``t`` is finalised
at ``t + delta``, so ``delta`` hours of future are exactly what the operator has
by the time the alarm is due. Nothing here reads a label, an event boundary, an
attack parameter, or another sensor's future.

The base bank already carries the causal history, so only the forward side is
built.
"""
from __future__ import annotations

import numpy as np


#: Columns gathered at each forward offset. Fixed before fitting.
FORWARD_COLUMNS = (
    "residual", "abs_residual", "residual_rate", "reference_support",
    "normal_error_scale", "seq_innovation", "seq_abs_innovation",
    "drift_ramp_3", "drift_mean_3", "drift_consistency_3", "drift_ramp_strength_3",
    "noise_energy_3", "noise_state_4.0",
    "seasonal_delta_24", "seasonal_abs_delta_24", "seasonal_signed_agreement",
)

SUMMARY_NAMES = (
    "future_residual_max", "future_residual_min", "future_residual_mean",
    "future_residual_std", "future_residual_rise", "future_residual_slope",
    "future_abs_max", "future_abs_mean", "future_abs_std",
    "future_support", "future_sign_agreement",
)

_TIME_BASE = 1 << 20


def _series_key(arrays):
    """One integer per (generator source, scenario, sensor) series."""
    source = arrays["source"].astype(np.int64)
    scenario = arrays["scenario"].astype(np.int64)
    node = arrays["node"].astype(np.int64)
    return (source * 10**9 + scenario * 10**5 + node)


def offset_index(arrays, offsets):
    """Row index of ``t + k`` for each row and each signed ``k`` in ``offsets``.

    Hours are matched by timestep, never by position in the series: with a 50%
    missing rate most sensor series are gappy, and a positional shift would
    silently refuse every neighbour whose intervening hours are absent.
    Returns an ``(n, len(offsets))`` integer array holding -1 where that hour of
    the same series is absent.
    """
    offsets = tuple(int(offset) for offset in offsets)
    key = _series_key(arrays)
    timestep = arrays["timestep"].astype(np.int64)
    if timestep.min() < 0 or timestep.max() >= _TIME_BASE:
        raise ValueError("Timestep does not fit the composite series index")
    composite = key * _TIME_BASE + timestep
    order = np.argsort(composite, kind="stable")
    ordered = composite[order]
    if np.any(np.diff(ordered) == 0):
        raise ValueError("Duplicate (source, scenario, node, timestep) rows")
    index = np.full((len(composite), len(offsets)), -1, dtype=np.int64)
    for column, offset in enumerate(offsets):
        wanted = composite + offset
        position = np.searchsorted(ordered, wanted)
        inside = position < len(ordered)
        found = np.zeros(len(composite), dtype=bool)
        found[inside] = ordered[position[inside]] == wanted[inside]
        index[found, column] = order[position[found]]
    return index


def forward_index(arrays, delta):
    """Row index of ``t + k`` for each row and each ``k`` in ``1..delta``."""
    return offset_index(arrays, range(1, int(delta) + 1))


def delayed_decision_features(arrays, names, delta):
    """Forward-window features and their names, for a declared ``delta``."""
    if delta < 1:
        raise ValueError("A delayed decision needs at least one hour of latency")
    lookup = {name: position for position, name in enumerate(names)}
    missing = [name for name in FORWARD_COLUMNS if name not in lookup]
    if missing:
        raise ValueError(f"Feature bank is missing forward columns: {missing}")
    X = arrays["X"]
    index = forward_index(arrays, delta)
    rows = len(X)

    gathered, built_names = [], []
    for name in FORWARD_COLUMNS:
        column = X[:, lookup[name]]
        for offset in range(delta):
            values = np.full(rows, np.nan, dtype=np.float32)
            present = index[:, offset] >= 0
            values[present] = column[index[present, offset]]
            gathered.append(values)
            built_names.append(f"{name}_t+{offset + 1}")

    residual = X[:, lookup["residual"]].astype(np.float32)
    absolute = X[:, lookup["abs_residual"]].astype(np.float32)
    window_residual = np.full((rows, delta + 1), np.nan, dtype=np.float32)
    window_absolute = np.full((rows, delta + 1), np.nan, dtype=np.float32)
    window_residual[:, 0] = residual
    window_absolute[:, 0] = absolute
    for offset in range(delta):
        present = index[:, offset] >= 0
        window_residual[present, offset + 1] = residual[index[present, offset]]
        window_absolute[present, offset + 1] = absolute[index[present, offset]]

    present = ~np.isnan(window_residual)
    support = present.sum(axis=1).astype(np.float32)
    with np.errstate(invalid="ignore"):
        residual_max = np.nanmax(window_residual, axis=1)
        residual_min = np.nanmin(window_residual, axis=1)
        residual_mean = np.nanmean(window_residual, axis=1)
        residual_std = np.nanstd(window_residual, axis=1)
        absolute_max = np.nanmax(window_absolute, axis=1)
        absolute_mean = np.nanmean(window_absolute, axis=1)
        absolute_std = np.nanstd(window_absolute, axis=1)

    last = np.full(rows, np.nan, dtype=np.float32)
    for offset in range(delta + 1):
        seen = present[:, offset]
        last[seen] = window_residual[seen, offset]
    rise = last - residual

    hours = np.arange(delta + 1, dtype=np.float32)
    centred = np.where(present, hours[None, :], np.nan)
    with np.errstate(invalid="ignore"):
        mean_hour = np.nanmean(centred, axis=1)
        deviation = centred - mean_hour[:, None]
        variance = np.nansum(deviation ** 2, axis=1)
        covariance = np.nansum(deviation * (window_residual - residual_mean[:, None]), axis=1)
    slope = np.where(variance > 0, covariance / np.where(variance > 0, variance, 1.0), np.nan)

    reference_sign = np.sign(residual)
    with np.errstate(invalid="ignore"):
        agreement = np.nanmean(np.sign(window_residual) * reference_sign[:, None], axis=1)

    summaries = [residual_max, residual_min, residual_mean, residual_std, rise, slope,
                 absolute_max, absolute_mean, absolute_std, support, agreement]
    gathered.extend(np.asarray(value, dtype=np.float32) for value in summaries)
    built_names.extend(SUMMARY_NAMES)
    return np.column_stack(gathered).astype(np.float32), built_names
