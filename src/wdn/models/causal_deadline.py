"""Past-only sensor localisation at a fixed post-alarm deadline."""
from __future__ import annotations

import numpy as np


FAMILY_IDS = {"drift": 3, "noise": 4}


def _rank(values, available):
    """Return deterministic within-incident ranks, keeping unseen sensors last."""
    values = np.asarray(values, dtype=float)
    available = np.asarray(available, dtype=bool)
    ranks = np.zeros(len(values), dtype=float)
    if np.any(available):
        local = np.flatnonzero(available)
        order = np.argsort(values[local], kind="stable")
        ranks[local[order]] = (np.arange(len(local)) + .5) / len(local)
    return ranks


def causal_deadline_groups(arrays, learned_score, names, family, deadline_steps,
                           learned_mode, learned_weight):
    """Score known sensors using observations available by a causal deadline.

    The incident onset and family are external inputs. A checkpoint is the earlier
    of ``deadline_steps`` available time points and observed incident closure.
    No score is copied back to rows before the checkpoint.
    """
    if family not in FAMILY_IDS or learned_mode not in {"mean", "top2"}:
        raise ValueError("Unsupported family or aggregation mode")
    if not isinstance(deadline_steps, int) or deadline_steps <= 0:
        raise ValueError("deadline_steps must be a positive integer")
    if not 0 <= learned_weight <= 1:
        raise ValueError("learned_weight must be in [0, 1]")
    learned_score = np.asarray(learned_score, dtype=float)
    labels = np.asarray(arrays["labels"])
    if learned_score.shape != labels.shape or not np.isfinite(learned_score).all():
        raise ValueError("learned_score must be finite and row aligned")

    innovation = names.index("dynamic_innovation")
    scale = names.index("dynamic_sigma")
    family_id = FAMILY_IDS[family]
    selected = np.asarray(arrays["families"]) == family_id
    roster = np.sort(np.unique(np.asarray(arrays["node"])))
    groups = {}
    for incident in np.unique(np.asarray(arrays["event"])[selected]):
        incident_mask = selected & (np.asarray(arrays["event"]) == incident)
        times = np.sort(np.unique(np.asarray(arrays["timestep"])[incident_mask]))
        checkpoint = times[min(deadline_steps - 1, len(times) - 1)]
        prefix = incident_mask & (np.asarray(arrays["timestep"]) <= checkpoint)
        items = []
        for node in roster:
            rows = np.flatnonzero(prefix & (np.asarray(arrays["node"]) == node))
            rows = rows[np.argsort(np.asarray(arrays["timestep"])[rows], kind="stable")]
            available = bool(len(rows))
            if available:
                values = learned_score[rows]
                learned = (float(np.mean(values)) if learned_mode == "mean" else
                           float(np.mean(np.sort(values)[-min(2, len(values)):])))
                z = (np.asarray(arrays["X"])[rows, innovation]
                     / np.maximum(np.asarray(arrays["X"])[rows, scale], 1e-4))
                if family == "noise":
                    physical = float(np.sqrt(np.mean(np.diff(z)**2))) if len(z) > 1 else 0.
                else:
                    time = np.asarray(arrays["timestep"])[rows].astype(float)
                    centered = time - time.mean()
                    slope = abs(np.dot(centered, z))/np.sqrt(max(np.dot(centered, centered), 1.))
                    physical = float(max(abs(z.mean())*np.sqrt(len(z)), slope))
            else:
                learned = physical = 0.
            target_rows = incident_mask & (np.asarray(arrays["node"]) == node)
            items.append({"node": int(node), "rows": rows, "available": available,
                "learned": learned, "physical": physical,
                "target": bool(np.any(labels[target_rows] > 0))})
        available = np.asarray([item["available"] for item in items])
        learned_rank = _rank([item["learned"] for item in items], available)
        physical_rank = _rank([item["physical"] for item in items], available)
        for item, lr, pr in zip(items, learned_rank, physical_rank):
            item["learned_rank"] = float(lr); item["physical_rank"] = float(pr)
            item["fused_rank"] = float(learned_weight*lr+(1-learned_weight)*pr)
        groups[int(incident)] = {"checkpoint": int(checkpoint),
            "available_steps": int(min(deadline_steps, len(times))),
            "incident_steps": int(len(times)), "sensors": items}
    return groups


def sensor_event_metrics(groups, top_k):
    """Evaluate one latched decision per known sensor and incident."""
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be positive")
    labels, decisions = [], []
    for group in groups.values():
        sensors = group["sensors"]
        available = [sensor for sensor in sensors if sensor["available"]]
        chosen = {sensor["node"] for sensor in sorted(available,
            key=lambda sensor: (sensor["fused_rank"], -sensor["node"]), reverse=True)[:top_k]}
        for sensor in sensors:
            labels.append(sensor["target"]); decisions.append(sensor["node"] in chosen)
    labels, decisions = np.asarray(labels, bool), np.asarray(decisions, bool)
    tp = int(np.sum(labels & decisions)); fp = int(np.sum(~labels & decisions))
    fn = int(np.sum(labels & ~decisions))
    return {"f1": float(2*tp/max(2*tp+fp+fn, 1)),
        "precision": float(tp/max(tp+fp, 1)), "recall": float(tp/max(tp+fn, 1)),
        "tp": tp, "fp": fp, "fn": fn}
