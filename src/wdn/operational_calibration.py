"""Exact, calibration-only operating points with explicit family/FPR targets."""
from __future__ import annotations

import numpy as np


def calibrate_threshold(scores, labels, families, *, objective="macro_f1", max_fpr=.001,
                        min_replay_f1=.5):
    """Choose ONE deployable threshold; true families are calibration labels only.

    Bounds apply to both all normal readings and clean-episode readings on
    calibration. They are not guarantees on unseen data. All five attack
    families need positive calibration support. Tied scores move together.
    """
    scores, labels, families = map(np.asarray, (scores, labels, families))
    if scores.ndim != 1 or scores.shape != labels.shape or scores.shape != families.shape or not len(scores):
        raise ValueError("Expected nonempty matching one-dimensional arrays")
    if not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Scores must be finite and labels binary")
    if not np.issubdtype(scores.dtype, np.floating):
        scores = scores.astype(float)
    if objective not in ("macro_f1", "worst_f1", "legacy"):
        raise ValueError("Unknown calibration objective")
    if not 0 <= max_fpr <= 1 or not 0 <= min_replay_f1 <= 1:
        raise ValueError("Invalid constraints")
    pos = labels > .5
    if any(not (pos & (families == f)).any() for f in range(1, 6)):
        raise ValueError("Every attack family needs calibration positives")
    clean = (families == 0) & ~pos
    if not clean.any():
        raise ValueError("Clean calibration observations are required")
    order = np.argsort(-scores, kind="stable")
    s, y, family = scores[order], pos[order], families[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s)-1]
    predicted = ends+1
    tp = np.cumsum(y)[ends]
    overall = 2*tp/(pos.sum()+predicted)
    fpr = (predicted-tp)/(~pos).sum()
    clean_fpr = np.cumsum((family == 0) & ~y)[ends]/clean.sum()
    per_family = []
    for f in range(1, 6):
        mask = family == f
        family_tp = np.cumsum(mask & y)[ends]
        family_pred = np.cumsum(mask)[ends]
        per_family.append(2*family_tp/((mask & y).sum()+family_pred))
    per_family = np.asarray(per_family)
    macro, worst, replay = per_family.mean(0), per_family.min(0), per_family[1]
    if objective == "legacy":
        score = overall-np.maximum(0., .5-replay)
    elif objective == "macro_f1":
        score = macro
    else:
        score = worst
    feasible = (fpr <= max_fpr) & (clean_fpr <= max_fpr) & (replay >= min_replay_f1)
    candidates = np.flatnonzero(feasible)
    if not len(candidates):
        raise ValueError("No feasible calibration operating point")
    # Strict > deployment convention, also for float32 inputs.
    thresholds = np.nextafter(s[ends], np.array(-np.inf, dtype=s.dtype))
    best = candidates[np.lexsort((thresholds[candidates], overall[candidates],
                                  macro[candidates], score[candidates]))[-1]]
    return {"threshold": float(thresholds[best]), "objective": objective,
            "max_calibration_fpr": max_fpr, "min_calibration_replay_f1": min_replay_f1,
            "calibration": {"overall_f1": float(overall[best]), "macro_f1": float(macro[best]),
                "worst_family_f1": float(worst[best]), "replay_f1": float(replay[best]),
                "fpr": float(fpr[best]), "clean_fpr": float(clean_fpr[best])}}


def family_summary(report):
    rows = [report["per_family"][f] for f in ("random", "replay", "stealthy", "noise", "targeted")]
    return {"macro_f1": float(np.mean([r["f1"] for r in rows])),
            "worst_family_f1": min(r["f1"] for r in rows),
            "macro_recall": float(np.mean([r["recall"] for r in rows]))}
