"""Evidence router and expert-feedback utilities for the operational detector.

The router works at graph-time resolution and sees aggregates of observable
sensor evidence. Drift/noise feedback works at the observed sensor row. No
function in this module accepts attack labels or event boundaries at inference.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit


ROUTER_CLASSES = ("clean", "abrupt", "replay", "drift", "noise")

# Fixed before fitting. These are evidence channels rather than family labels.
RAW_EVIDENCE_NAMES = (
    "residual", "abs_residual", "reference_support", "normal_error_scale",
    "lag_advantage", "lag_support", "last_gap", "residual_rate",
    "dynamic_abs_innovation", "dynamic_sigma", "dynamic_level_shift_8",
    "dynamic_innovation_rms_8", "dynamic_level_shift_16",
    "dynamic_innovation_rms_16", "reference_disagreement",
    "seq_abs_innovation", "seq_normal_sigma", "seq_seen",
    "drift_cusum_positive_0.5", "drift_cusum_negative_0.5",
    "noise_state_4.0", "noise_state_9.0", "drift_consistency_3",
    "drift_ramp_strength_3", "drift_mean_strength_3", "noise_energy_3",
    "drift_consistency_8", "drift_ramp_strength_8",
    "drift_mean_strength_8", "noise_energy_8", "seasonal_delta_24",
    "seasonal_abs_delta_24", "seasonal_support_24", "seasonal_delta_48",
    "seasonal_abs_delta_48", "seasonal_support_48",
    "seasonal_signed_agreement",
)

SCORE_EVIDENCE_NAMES = (
    "frozen_drift", "frozen_noise", "maxpool_drift", "maxpool_noise",
    "delayed_drift", "delayed_noise", "final_drift", "final_noise",
)


def feature_columns(names):
    lookup = {name: i for i, name in enumerate(names)}
    missing = [name for name in RAW_EVIDENCE_NAMES if name not in lookup]
    if missing:
        raise ValueError(f"Missing evidence features: {missing}")
    return np.asarray([lookup[name] for name in RAW_EVIDENCE_NAMES], dtype=int)


def local_evidence(X, names, score_parts):
    """Return the fixed local evidence matrix used by router and feedback."""
    missing = [name for name in SCORE_EVIDENCE_NAMES if name not in score_parts]
    if missing:
        raise ValueError(f"Missing score evidence: {missing}")
    raw = np.asarray(X)[:, feature_columns(names)]
    scores = np.column_stack([np.asarray(score_parts[name])
                              for name in SCORE_EVIDENCE_NAMES])
    if len(raw) != len(scores):
        raise ValueError("Raw and score evidence have different row counts")
    return np.column_stack((raw, scores)).astype(np.float32)


def contiguous_groups(arrays):
    """Map rows to contiguous `(source, scenario, timestep)` graph-time groups."""
    n = len(arrays["timestep"])
    if not all(len(arrays[key]) == n for key in ("source", "scenario", "timestep")):
        raise ValueError("Group keys have different lengths")
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    keys = [np.asarray(arrays[key]) for key in ("source", "scenario", "timestep")]
    start = np.r_[True, (keys[0][1:] != keys[0][:-1]) |
                          (keys[1][1:] != keys[1][:-1]) |
                          (keys[2][1:] != keys[2][:-1])]
    starts = np.flatnonzero(start)
    group = np.cumsum(start, dtype=np.int64) - 1
    # A repeated key in disjoint blocks would mix router context incorrectly.
    triples = np.column_stack([key[starts] for key in keys])
    if len(np.unique(triples, axis=0)) != len(triples):
        raise ValueError("Graph-time groups are not contiguous")
    return group, starts


def aggregate_router_evidence(local, group, starts):
    """Aggregate observed-sensor evidence to one router vector per graph-time."""
    local = np.asarray(local, dtype=np.float32)
    if len(local) != len(group):
        raise ValueError("Evidence and group index lengths differ")
    if not len(starts):
        return np.empty((0, local.shape[1] * 3), dtype=np.float32)
    counts = np.diff(np.r_[starts, len(local)]).astype(np.float32)[:, None]
    sums = np.add.reduceat(local, starts, axis=0)
    mean = sums / counts
    sq = np.add.reduceat(local * local, starts, axis=0) / counts
    std = np.sqrt(np.maximum(0.0, sq - mean * mean))
    maximum = np.maximum.reduceat(local, starts, axis=0)
    return np.column_stack((mean, std, maximum)).astype(np.float32)


def router_targets(families):
    """Map benchmark families to evidence states; random/targeted share abrupt."""
    families = np.asarray(families, dtype=int)
    target = families.copy()
    target[families == 5] = 1
    if np.any((target < 0) | (target >= len(ROUTER_CLASSES))):
        raise ValueError("Unknown family id in router target")
    return target


def group_values(values, starts, name="value"):
    values = np.asarray(values)
    if not len(starts):
        return values[:0]
    first = values[starts]
    repeated = np.repeat(first, np.diff(np.r_[starts, len(values)]))
    if not np.array_equal(values, repeated):
        raise ValueError(f"{name} is not constant inside graph-time groups")
    return first


def balanced_weights(target):
    target = np.asarray(target)
    weight = np.ones(len(target), dtype=np.float64)
    classes, counts = np.unique(target, return_counts=True)
    for cls, count in zip(classes, counts):
        weight[target == cls] = len(target) / (len(classes) * count)
    return weight


def clipped_logit(value):
    value = np.clip(np.asarray(value, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(value / (1 - value))


def guarded_specialist_scores(base, router_probs, feedback_probs,
                              router_strength, feedback_strength):
    """Re-rank frozen drift/noise scores with router and feedback evidence.

    Replay is the explicit competing mechanism because the measured deployment
    failure is noise evidence firing on replayed but genuine measurements.
    Threshold calibration absorbs score offsets; only ordering matters here.
    """
    base = np.asarray(base, dtype=float)
    router_probs = np.asarray(router_probs, dtype=float)
    feedback_probs = np.asarray(feedback_probs, dtype=float)
    if base.shape[1:] != (2,) or feedback_probs.shape != base.shape:
        raise ValueError("Base and feedback scores must have shape (n, 2)")
    if router_probs.shape != (len(base), len(ROUTER_CLASSES)):
        raise ValueError("Router probabilities have the wrong shape")
    replay = clipped_logit(router_probs[:, 2])
    adjusted = np.empty_like(base)
    for column, route_column in enumerate((3, 4)):
        route_margin = clipped_logit(router_probs[:, route_column]) - replay
        adjusted[:, column] = expit(np.clip(
            clipped_logit(base[:, column])
            + float(router_strength) * route_margin
            + float(feedback_strength) * clipped_logit(feedback_probs[:, column]),
            -30, 30))
    return adjusted


def specialist_veto_masks(router_probs, feedback_probs, margin, cutoff):
    """Return replay-driven vetoes for drift/noise candidate alarms.

    A veto exists only when replay routing exceeds the candidate mechanism by
    the declared margin *and* that candidate's own feedback rejects it. This is
    deliberately one-sided: the mechanism cannot create a new alarm, and a
    rejected specialist falls back to the always-active mixture branch.
    """
    router_probs = np.asarray(router_probs, dtype=float)
    feedback_probs = np.asarray(feedback_probs, dtype=float)
    if router_probs.ndim != 2 or router_probs.shape[1] != len(ROUTER_CLASSES):
        raise ValueError("Router probabilities have the wrong shape")
    if feedback_probs.shape != (len(router_probs), 2):
        raise ValueError("Feedback probabilities must have shape (n, 2)")
    replay = router_probs[:, 2]
    drift = ((replay > router_probs[:, 3] + float(margin))
             & (feedback_probs[:, 0] < float(cutoff)))
    noise = ((replay > router_probs[:, 4] + float(margin))
             & (feedback_probs[:, 1] < float(cutoff)))
    return np.column_stack((drift, noise))


@dataclass
class FeedbackRouterBundle:
    names: tuple[str, ...]
    router: object
    feedback_drift: object
    feedback_noise: object

    def predict(self, local, arrays):
        group, starts = contiguous_groups(arrays)
        aggregate = aggregate_router_evidence(local, group, starts)
        group_probs = self.router.predict_proba(aggregate)
        if not np.array_equal(self.router.classes_, np.arange(len(ROUTER_CLASSES))):
            raise RuntimeError("Router was not fitted on all declared classes")
        router = group_probs[group]
        feedback_X = np.column_stack((local, router)).astype(np.float32)
        feedback = np.column_stack((
            self.feedback_drift.predict_proba(feedback_X)[:, 1],
            self.feedback_noise.predict_proba(feedback_X)[:, 1],
        ))
        return {"router": router, "feedback": feedback,
                "router_group": group_probs, "group_starts": starts}
