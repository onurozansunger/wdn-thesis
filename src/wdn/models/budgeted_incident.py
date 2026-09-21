"""Mechanism/rank fusion under a predeclared sparse incident budget."""
from __future__ import annotations

import numpy as np


def budgeted_noise_rank_score(arrays, learned_score, names):
    """Fuse equal sensor ranks of learned evidence and variance-change evidence.

    Returns one score per pressure observation inside noise incident windows. The
    dynamic-innovation difference statistic is invariant to a constant offset and
    directly targets the injected-noise mechanism.
    """
    learned_score = np.asarray(learned_score, dtype=float)
    if learned_score.shape != np.asarray(arrays["labels"]).shape:
        raise ValueError("Learned point score must align with the feature rows")
    innovation = names.index("dynamic_innovation")
    scale = names.index("dynamic_sigma")
    selected = np.asarray(arrays["families"]) == 4
    selected_rows = np.flatnonzero(selected)
    row_position = {int(row): position for position, row in enumerate(selected_rows)}
    output = np.zeros(len(selected_rows), dtype=np.float64)
    sensor_groups = {}
    for incident in np.unique(np.asarray(arrays["event"])[selected]):
        incident_rows = selected & (np.asarray(arrays["event"]) == incident)
        groups = []
        for node in np.unique(np.asarray(arrays["node"])[incident_rows]):
            rows = np.flatnonzero(incident_rows & (np.asarray(arrays["node"]) == node))
            rows = rows[np.argsort(np.asarray(arrays["timestep"])[rows], kind="stable")]
            z = (np.asarray(arrays["X"])[rows, innovation]
                 /np.maximum(np.asarray(arrays["X"])[rows, scale], 1e-4))
            physical = float(np.sqrt(np.mean(np.diff(z)**2))) if len(z) > 1 else 0.0
            learned = float(np.mean(learned_score[rows]))
            groups.append({"node": int(node), "rows": rows,
                           "physical": physical, "learned": learned})
        for field, rank_field in (("physical", "physical_rank"),
                                  ("learned", "learned_rank")):
            values = np.asarray([group[field] for group in groups])
            order = np.argsort(values, kind="stable")
            ranks = np.empty(len(groups), dtype=np.float64)
            ranks[order] = (np.arange(len(groups))+.5)/len(groups)
            for group, rank in zip(groups, ranks):
                group[rank_field] = float(rank)
        for group in groups:
            group["fused_rank"] = 0.5*(group["physical_rank"]+group["learned_rank"])
            for row in group["rows"]:
                output[row_position[int(row)]] = group["fused_rank"]
        sensor_groups[int(incident)] = groups
    return output, selected_rows, sensor_groups


def top_k_sensor_decision(selected_rows, sensor_groups, top_k):
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    row_position = {int(row): position for position, row in enumerate(selected_rows)}
    decision = np.zeros(len(selected_rows), dtype=bool)
    for groups in sensor_groups.values():
        chosen = sorted(groups, key=lambda group: (group["fused_rank"], -group["node"]),
                        reverse=True)[:min(top_k, len(groups))]
        for group in chosen:
            for row in group["rows"]:
                decision[row_position[int(row)]] = True
    return decision
