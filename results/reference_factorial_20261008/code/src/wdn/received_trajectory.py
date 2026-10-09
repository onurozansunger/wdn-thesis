"""Causal received-trajectory evidence for the existing General mixture.

All comparisons are same-sensor, exact-hour pairs. The caller isolates each
source/scenario. No attack metadata, clean values, or future targets are inputs.
"""
from __future__ import annotations

import numpy as np
from wdn.observed_history import (OBSERVED_LAGS, DELTA_BIN_SIGMA,
                                  observed_history_features)

TRAJECTORY_WINDOWS = (3, 6)
TRAJECTORY_STATS = ('median_signed', 'median_abs', 'mad', 'pair_count', 'newest_age')


def _median(x, count):
    ordered = np.sort(np.where(np.isfinite(x), x, np.inf), axis=1)
    row = np.arange(len(x))
    lo, hi = np.maximum((count - 1) // 2, 0), np.maximum(count // 2, 0)
    result = (ordered[row, lo] + ordered[row, hi]) / 2
    result[count == 0] = np.nan
    return result


def received_trajectory_features(values, mask, timestep, query_time, query_node,
                                 query_scale, rounding_m=0.):
    """Return unchanged 42 point features plus 140 fixed trajectory features.

For each lag and window, summarize binned differences between t-age and
t-age-lag. Distribution statistics need >=2 pairs; count and age remain
available with one pair. No alarm state is carried forward.
"""
    base, names = observed_history_features(values, mask, timestep, query_time,
                                           query_node, query_scale, rounding_m)
    values, mask = np.asarray(values), np.asarray(mask, bool)
    qt, qn, scale = np.asarray(query_time), np.asarray(query_node), np.asarray(query_scale)
    order = np.argsort(timestep)
    times = np.asarray(timestep)[order]
    n = len(qt)

    def received(wanted):
        pos = np.searchsorted(times, wanted)
        safe = np.minimum(pos, len(times) - 1)
        idx = order[safe]
        valid = (pos < len(times)) & (times[safe] == wanted) & mask[idx, qn]
        found = np.full(n, np.nan)
        found[valid] = values[idx[valid], qn[valid]]
        if not np.isfinite(found[valid]).all():
            raise ValueError('Observed trajectory values must be finite')
        if rounding_m:
            found[valid] = np.round(found[valid] / rounding_m) * rounding_m
        return found

    recent = np.column_stack([received(qt - age) for age in range(max(TRAJECTORY_WINDOWS))])
    result = np.empty((n, len(OBSERVED_LAGS) * len(TRAJECTORY_WINDOWS) * len(TRAJECTORY_STATS)), np.float32)
    col = 0
    for lag in OBSERVED_LAGS:
        previous = np.column_stack([received(qt - age - lag) for age in range(max(TRAJECTORY_WINDOWS))])
        delta = (recent - previous) / scale[:, None]
        delta = np.round(delta / DELTA_BIN_SIGMA) * DELTA_BIN_SIGMA
        for window in TRAJECTORY_WINDOWS:
            d = delta[:, :window]
            valid = np.isfinite(d)
            count = valid.sum(1)
            signed = _median(d, count)
            absolute = _median(np.abs(d), count)
            mad = _median(np.abs(d - signed[:, None]), count)
            stats = np.column_stack((signed, absolute, mad))
            stats = np.round(stats / DELTA_BIN_SIGMA) * DELTA_BIN_SIGMA
            stats[count < 2] = np.nan
            age = np.where(valid, np.arange(window)[None, :], np.inf).min(1)
            age[count == 0] = np.nan
            result[:, col:col + 5] = np.column_stack((stats, count, age))
            names.extend(f'history_trajectory_{stat}_lag{lag}_w{window}' for stat in TRAJECTORY_STATS)
            col += 5
    return np.column_stack((base, result)), names
