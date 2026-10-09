"""Past received-reading consistency, shared by all existing experts.

No clean readings, event labels or attack parameters are inputs. Changes are
binned at 0.1 reference-noise sigma: the model cannot use exact float equality
as a copy detector. This is an evidence bank, not a family-specific rule.
"""
from __future__ import annotations

import numpy as np

OBSERVED_LAGS = tuple(range(1, 13)) + (24, 48)
DELTA_BIN_SIGMA = .1


def observed_history_features(values, mask, timestep, query_time, query_node,
                              query_scale, rounding_m=0.):
    """Compute signed/absolute changes and support at exact past hours.

    ``rounding_m`` is an optional *feature-path* sensitivity experiment; it does
    not alter the canonical benchmark or claim end-to-end sensor robustness.
    Missing sources stay NaN with availability zero; no imputation or future.
    """
    values = np.asarray(values)
    mask = np.asarray(mask, dtype=bool)
    timestep = np.asarray(timestep)
    qt, qn = np.asarray(query_time), np.asarray(query_node)
    scale = np.asarray(query_scale)
    if values.ndim != 2 or values.shape != mask.shape or len(timestep) != len(values):
        raise ValueError("Observation arrays must align")
    if not np.issubdtype(timestep.dtype, np.integer) or not np.issubdtype(qt.dtype, np.integer):
        raise ValueError("Timesteps must be integer hours")
    if not np.issubdtype(qn.dtype, np.integer) or np.any(qn < 0) or np.any(qn >= values.shape[1]):
        raise ValueError("Invalid sensor indices")
    if len(qt) != len(qn) or scale.shape != qt.shape or np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("Positive per-query reference scales are required")
    if rounding_m < 0:
        raise ValueError("Rounding resolution cannot be negative")
    order = np.argsort(timestep)
    times = timestep[order]
    if len(times) == 0 or np.any(np.diff(times) == 0):
        raise ValueError("Empty or duplicate scenario timesteps")
    current_pos = np.searchsorted(times, qt)
    if np.any(current_pos >= len(times)) or np.any(times[current_pos] != qt):
        raise ValueError("Query time is absent")
    current_idx = order[current_pos]
    if not mask[current_idx, qn].all():
        raise ValueError("Query endpoints must be observed")
    current = values[current_idx, qn].astype(np.float64)
    if not np.isfinite(current).all():
        raise ValueError("Observed values must be finite")
    if rounding_m:
        current = np.round(current / rounding_m) * rounding_m
    result = np.full((len(qt), len(OBSERVED_LAGS) * 3), np.nan, np.float32)
    names = []
    for i, lag in enumerate(OBSERVED_LAGS):
        wanted = qt - lag
        pos = np.searchsorted(times, wanted)
        safe = np.minimum(pos, len(times) - 1)
        previous_idx = order[safe]
        present = (pos < len(times)) & (times[safe] == wanted) & mask[previous_idx, qn]
        previous = values[previous_idx[present], qn[present]].astype(np.float64)
        if not np.isfinite(previous).all():
            raise ValueError("Observed history must be finite")
        if rounding_m:
            previous = np.round(previous / rounding_m) * rounding_m
        delta = (current[present] - previous) / scale[present]
        delta = np.round(delta / DELTA_BIN_SIGMA) * DELTA_BIN_SIGMA
        result[present, 3 * i] = delta
        result[present, 3 * i + 1] = np.abs(delta)
        result[:, 3 * i + 2] = present
        names.extend((f"history_received_change_{lag}", f"history_received_abs_change_{lag}",
                      f"history_received_available_{lag}"))
    return result, names
