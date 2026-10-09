"""Causal, label-blind temporal state for separate drift/noise experts."""
from __future__ import annotations

import numpy as np


HALF_LIVES = (2, 4, 8)
WINDOWS = (3, 5, 8, 12)


def feature_names():
    names = []
    for half_life in HALF_LIVES:
        names += [f"state_signed_ewm_{half_life}", f"state_abs_ewm_{half_life}",
                  f"state_energy_ewm_{half_life}"]
    for offset in (.25, .50):
        suffix = str(offset).replace("0.", "")
        names += [f"state_cusum_positive_{suffix}", f"state_cusum_negative_{suffix}"]
    names += ["state_peak_abs_4", "state_exceedance_1_4", "state_exceedance_2_4"]
    for window in WINDOWS:
        names += [f"state_mean_strength_{window}", f"state_energy_{window}",
                  f"state_sign_consistency_{window}", f"state_slope_strength_{window}",
                  f"state_support_{window}"]
    return names


def family_state_features(arrays, names):
    """Append state computed only from current/past observed endpoint rows.

    The input must contain ``X``, ``scenario``, ``timestep`` and ``node``.
    Scenario/node identifiers reset and index state but never enter output as
    numerical features.
    """
    required = ("X", "scenario", "timestep", "node")
    if any(key not in arrays for key in required):
        raise ValueError("Missing family-state input arrays")
    X = np.asarray(arrays["X"], dtype=float)
    scenario, timestep, node = (np.asarray(arrays[key]) for key in required[1:])
    if (X.ndim != 2 or any(value.shape != (len(X),) for value in (scenario, timestep, node))
            or not np.isfinite(X).all()):
        raise ValueError("Expected finite endpoint feature rows and matching metadata")
    if not all(np.issubdtype(value.dtype, np.integer) for value in (scenario, timestep, node)):
        raise ValueError("Scenario, timestep and node must be integer arrays")
    if len(set(names)) != len(names) or len(names) != X.shape[1]:
        raise ValueError("Feature schema must be unique and match X")
    try:
        innovation_column = names.index("dynamic_innovation")
    except ValueError as error:
        raise ValueError("dynamic_innovation is required") from error
    innovation = np.clip(X[:, innovation_column], -20.0, 20.0)
    output_names = feature_names()
    output = np.empty((len(X), len(output_names)), dtype=np.float32)

    for sid in np.unique(scenario):
        index = np.flatnonzero(scenario == sid)
        index = index[np.lexsort((node[index], timestep[index]))]
        states = {}
        for row in index:
            sensor, clock, z = int(node[row]), int(timestep[row]), float(innovation[row])
            if sensor not in states:
                state = {"last": clock,
                    "signed": {half_life: z for half_life in HALF_LIVES},
                    "absolute": {half_life: abs(z) for half_life in HALF_LIVES},
                    "square": {half_life: z*z for half_life in HALF_LIVES},
                    "cusum_positive": {.25: max(0., z-.25), .50: max(0., z-.50)},
                    "cusum_negative": {.25: max(0., -z-.25), .50: max(0., -z-.50)},
                    "peak": abs(z), "exceed_1": float(abs(z) > 1),
                    "exceed_2": float(abs(z) > 2), "history": [(clock, z)]}
            else:
                state = states[sensor]
                gap = clock-state["last"]
                if gap <= 0:
                    raise ValueError("Duplicate or nonincreasing sensor endpoint")
                for half_life in HALF_LIVES:
                    decay = np.exp(-gap/half_life)
                    state["signed"][half_life] = decay*state["signed"][half_life]+(1-decay)*z
                    state["absolute"][half_life] = decay*state["absolute"][half_life]+(1-decay)*abs(z)
                    state["square"][half_life] = decay*state["square"][half_life]+(1-decay)*z*z
                cusum_decay = .90**gap
                for offset in (.25, .50):
                    state["cusum_positive"][offset] = max(
                        0., cusum_decay*state["cusum_positive"][offset]+z-offset)
                    state["cusum_negative"][offset] = max(
                        0., cusum_decay*state["cusum_negative"][offset]-z-offset)
                decay = np.exp(-gap/4)
                state["peak"] = max(decay*state["peak"], abs(z))
                state["exceed_1"] = decay*state["exceed_1"]+(1-decay)*float(abs(z) > 1)
                state["exceed_2"] = decay*state["exceed_2"]+(1-decay)*float(abs(z) > 2)
                state["history"].append((clock, z))
                state["history"] = [(when, value) for when, value in state["history"]
                                    if when >= clock-max(WINDOWS)+1]
                state["last"] = clock
            states[sensor] = state

            values = []
            for half_life in HALF_LIVES:
                values += [state["signed"][half_life], state["absolute"][half_life],
                           np.sqrt(max(state["square"][half_life], 0.))]
            for offset in (.25, .50):
                values += [state["cusum_positive"][offset], state["cusum_negative"][offset]]
            values += [state["peak"], state["exceed_1"], state["exceed_2"]]
            for window in WINDOWS:
                history = [(when, value) for when, value in state["history"]
                           if when >= clock-window+1]
                times = np.asarray([when for when, _ in history], dtype=float)
                samples = np.asarray([value for _, value in history], dtype=float)
                count = len(samples)
                mean_strength = samples.sum()/np.sqrt(max(count, 1))
                energy = np.sqrt(np.mean(samples**2))
                consistency = abs(np.sign(samples).sum())/max(count, 1)
                centered = times-times.mean()
                denominator = np.sqrt(max(float(np.dot(centered, centered)), 1.0))
                slope_strength = abs(float(np.dot(centered, samples)))/denominator
                values += [mean_strength, energy, consistency, slope_strength, count/window]
            output[row] = values
    if not np.isfinite(output).all():
        raise FloatingPointError("Nonfinite family-state features")
    return output, output_names
