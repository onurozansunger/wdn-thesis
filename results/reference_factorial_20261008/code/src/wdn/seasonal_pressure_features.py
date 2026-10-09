"""Past-only daily seasonal pressure differences for operational experts."""
from __future__ import annotations

import math
import numpy as np


NAMES = ("seasonal_delta_24", "seasonal_abs_delta_24", "seasonal_support_24",
         "seasonal_delta_48", "seasonal_abs_delta_48", "seasonal_support_48",
         "seasonal_signed_agreement")


def seasonal_pressure_features(values, observed, noise_sigma=.1):
    values = np.asarray(values, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    if values.ndim != 2 or observed.shape != values.shape or noise_sigma <= 0:
        raise ValueError("Expected time-by-sensor values/mask and positive noise sigma")
    if not np.isfinite(values[observed]).all():
        raise ValueError("Observed pressure values must be finite")
    output = np.zeros((*values.shape, len(NAMES)), dtype=np.float32)
    denominator = math.sqrt(2) * noise_sigma
    for lag, column in ((24, 0), (48, 3)):
        if len(values) <= lag:
            continue
        support = observed[lag:] & observed[:-lag]
        delta = np.zeros_like(values[lag:], dtype=np.float32)
        delta[support] = ((values[lag:][support] - values[:-lag][support])
                          / denominator).astype(np.float32)
        output[lag:, :, column] = delta
        output[lag:, :, column + 1] = np.abs(delta)
        output[lag:, :, column + 2] = support
    first, second = output[:, :, 0], output[:, :, 3]
    output[:, :, 6] = np.sign(first * second) * np.minimum(np.abs(first), np.abs(second))
    if not np.isfinite(output).all():
        raise FloatingPointError("Nonfinite seasonal pressure feature")
    return output, list(NAMES)


def endpoint_seasonal_features(arrays, scenario_loader):
    """Align seasonal grids with flattened observed endpoint rows."""
    required = ("scenario", "timestep", "node")
    if any(key not in arrays for key in required):
        raise ValueError("Endpoint metadata is incomplete")
    parts = []
    for uid in sorted(map(int, np.unique(arrays["scenario"]))):
        scenario = scenario_loader(uid)
        timestep = np.asarray(scenario["timestep"])
        if not np.array_equal(timestep, np.arange(len(timestep))):
            raise ValueError("Seasonal expert requires contiguous hourly scenario time")
        features, names = seasonal_pressure_features(scenario["values"], scenario["mask"])
        endpoint = scenario["mask"][15:]
        block = features[15:][endpoint]
        selected = np.asarray(arrays["scenario"]) == uid
        expected_time = np.broadcast_to(timestep[15:, None], endpoint.shape)[endpoint]
        expected_node = np.broadcast_to(np.arange(endpoint.shape[1]), endpoint.shape)[endpoint]
        np.testing.assert_array_equal(np.asarray(arrays["timestep"])[selected], expected_time)
        np.testing.assert_array_equal(np.asarray(arrays["node"])[selected], expected_node)
        parts.append(block)
    result = np.concatenate(parts)
    if result.shape != (len(arrays["scenario"]), len(NAMES)):
        raise ValueError("Seasonal endpoint feature alignment failed")
    return result, names
