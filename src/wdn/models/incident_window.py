"""Fixed full-window score aggregation for offline incident localisation."""
from __future__ import annotations

import numpy as np


def aggregate_incident_scores(scores, window_id, node, timestep, selected, mode):
    """Assign one complete-window trajectory score to each observed sensor row.

    ``window_id`` and ``selected`` describe an externally supplied inspection window.
    The function does not infer its onset, end, or attack family.
    """
    scores = np.asarray(scores, dtype=float)
    window_id, node, timestep = (np.asarray(value) for value in (window_id, node, timestep))
    selected = np.asarray(selected, dtype=bool)
    if any(value.shape != scores.shape for value in (window_id, node, timestep, selected)):
        raise ValueError("Incident aggregation arrays must have matching one-dimensional shapes")
    if mode not in {"mean", "top2", "logit"} or not np.isfinite(scores).all():
        raise ValueError("Unsupported incident aggregation mode or nonfinite score")
    output = np.zeros(len(scores), dtype=np.float64)
    for incident in np.unique(window_id[selected]):
        if incident < 0:
            raise ValueError("Selected incident rows require a nonnegative window id")
        incident_rows = selected & (window_id == incident)
        for sensor in np.unique(node[incident_rows]):
            rows = np.flatnonzero(incident_rows & (node == sensor))
            rows = rows[np.argsort(timestep[rows], kind="stable")]
            values = scores[rows]
            if mode == "mean":
                value = float(np.mean(values))
            elif mode == "top2":
                value = float(np.mean(np.sort(values)[-min(len(values), 2):]))
            else:
                clipped = np.clip(values, 1e-6, 1-1e-6)
                value = float(np.log(clipped/(1-clipped)).sum()/np.sqrt(len(values)))
            output[rows] = value
    return output
