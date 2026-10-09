"""Causal, target-group-blind covariates for conditional reference error."""
from __future__ import annotations

import numpy as np


NAMES = ["normal_reference_level_m", "normal_reference_delta_1_m",
         "normal_reference_delta_4_m", "normal_reference_past_mean_4_m",
         "reference_support", "normal_error_scale", "reference_disagreement_m",
         "normal_hour_sin", "normal_hour_cos"]


def normal_context_features(prediction, support, disagreement, scale, timestep):
    """Build full-sequence covariates without a target pressure/residual input.

    ``prediction``, ``support`` and ``disagreement`` must already be produced
    by a reference that excludes the target's complete pressure group.  The
    function uses their current/past values and known clock time only.
    """
    prediction, support, disagreement = map(
        lambda x: np.asarray(x, dtype=float), (prediction, support, disagreement))
    scale, timestep = np.asarray(scale, dtype=float), np.asarray(timestep, dtype=float)
    if (prediction.ndim != 2 or support.shape != prediction.shape
            or disagreement.shape != prediction.shape
            or scale.shape != (prediction.shape[1],)
            or timestep.shape != (prediction.shape[0],)):
        raise ValueError("Expected matching time-sensor reference arrays")
    if (not np.isfinite(prediction).all() or not np.isfinite(support).all()
            or not np.isfinite(disagreement).all() or not np.isfinite(scale).all()
            or not np.isfinite(timestep).all() or np.any(scale <= 0)):
        raise ValueError("Normal context inputs must be finite with positive scale")
    T, N = prediction.shape
    lag1 = prediction[np.maximum(np.arange(T)-1, 0)]
    lag4 = prediction[np.maximum(np.arange(T)-4, 0)]
    past = np.empty_like(prediction)
    for t in range(T):
        start = max(0, t-4)
        past[t] = prediction[start:t].mean(0) if t > start else prediction[t]
    phase = 2*np.pi*(timestep % 24)/24
    features = np.stack([
        prediction, prediction-lag1, prediction-lag4, past,
        support, np.broadcast_to(scale, (T, N)), disagreement,
        np.broadcast_to(np.sin(phase)[:, None], (T, N)),
        np.broadcast_to(np.cos(phase)[:, None], (T, N)),
    ], axis=-1).astype(np.float32)
    if features.shape != (T, N, len(NAMES)) or not np.isfinite(features).all():
        raise FloatingPointError("Invalid normal context features")
    return features, NAMES.copy()
