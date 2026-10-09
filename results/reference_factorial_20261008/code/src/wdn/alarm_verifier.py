"""A per-branch verifier for candidate specialist alarms.

The Stage A audit found that the pooled-F1 deficit is not spread evenly across
the detector: on 99 calibration scenarios the noise specialist alone raises
6,164 of 8,705 false alarms and the general mixture alone raises 139. So this
verifier attacks specialist precision, and it is deliberately one-directional.

* It can only **remove** a candidate specialist alarm. It can never create one.
* It is never applied to the general mixture, whose immediate alarm path stays
  protected.
* It is not a second replay guard. The existing router/feedback veto handles the
  replay-period failure; this head is trained against *all* negatives, and the
  two are composed once each, never stacked as two copies of the same rule.

The verifier runs on the decision clock: the decision for hour ``t`` is due at
``t + 3``, so bounded-future evidence already inside that window is admissible.
Nothing here reads a label, an event boundary, an attack parameter, a target
identity, or anything after ``t + 3``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wdn.evidence_feedback import RAW_EVIDENCE_NAMES, ROUTER_CLASSES, feature_columns

BRANCHES = ("drift", "noise")
BRANCH_FAMILY = {"drift": 3, "noise": 4}

#: Score channels every verifier variant sees, in this order.
SCORE_CHANNELS = ("causal_drift", "causal_noise", "maxpool_drift", "maxpool_noise",
                  "delayed_drift", "delayed_noise", "final_drift", "final_noise")

#: Extra channels the early-warning variant adds, aligned to the endpoint's own
#: graph-time hour and its two preceding hours.
EARLY_CHANNELS = (tuple(f"early_p_{name}" for name in ROUTER_CLASSES)
                  + ("early_confidence", "early_abstain",
                     "early_warned_persistence", "early_p_drift_lag1",
                     "early_p_noise_lag1", "early_p_drift_lag2", "early_p_noise_lag2"))


def score_matrix(scores):
    """Stack the eight score channels in the declared order."""
    missing = [name for name in SCORE_CHANNELS if name not in scores]
    if missing:
        raise ValueError(f"Missing verifier score channels: {missing}")
    return np.column_stack([np.asarray(scores[name], dtype=float)
                            for name in SCORE_CHANNELS])


def early_row_channels(early, arrays):
    """Broadcast graph-time early-warning evidence down to endpoint rows.

    ``early`` is the output of :meth:`wdn.early_warning.EarlyWarningHead.predict_groups`
    computed on the *same* arrays, so its ``group`` index maps rows to hours.
    """
    from wdn.early_warning import past_lag_index

    group = early["group"]
    probabilities = early["probabilities"]
    keys = early["keys"]
    lag = past_lag_index(keys, (1, 2))
    warned = (early["prediction"] != 0) & ~early["abstain"]

    # Persistence: how many consecutive earlier hours also carried a warning.
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    persistence = np.zeros(len(keys), dtype=np.float32)
    previous = {}
    for position in order:
        series = (int(keys[position, 0]), int(keys[position, 1]))
        last_time, run = previous.get(series, (None, 0))
        hour = int(keys[position, 2])
        run = run + 1 if (warned[position] and last_time == hour - 1) else int(warned[position])
        persistence[position] = run
        previous[series] = (hour, run)

    columns = [probabilities, early["confidence"][:, None],
               early["abstain"].astype(np.float32)[:, None], persistence[:, None]]
    for column in range(2):
        present = lag[:, column] >= 0
        block = np.full((len(keys), 2), np.nan, dtype=np.float32)
        block[present] = probabilities[lag[present, column]][:, [3, 4]]
        columns.append(block)
    grouped = np.column_stack(columns).astype(np.float32)
    if grouped.shape[1] != len(EARLY_CHANNELS):
        raise AssertionError("Early channel matrix and name list disagree")
    return grouped[group]


def verifier_features(X, names, scores, early=None, arrays=None):
    """Assemble the verifier's design matrix and its column names."""
    raw = np.asarray(X)[:, feature_columns(names)]
    blocks = [raw, score_matrix(scores)]
    columns = list(RAW_EVIDENCE_NAMES) + list(SCORE_CHANNELS)
    if early is not None:
        if arrays is None:
            raise ValueError("Early channels need the arrays they were grouped on")
        blocks.append(early_row_channels(early, arrays))
        columns += list(EARLY_CHANNELS)
    return np.column_stack(blocks).astype(np.float32), columns


def candidate_rows(score, clean, quantile):
    """Rows plausible enough to be worth verifying, plus the pre-threshold used.

    The pre-threshold is set on *negatives only* so that the candidate region is
    defined by how unusual a score is, not by where the positives happen to sit.
    """
    values = np.sort(np.asarray(score, dtype=float)[clean])
    if not len(values):
        raise ValueError("A candidate region needs negative rows")
    position = min(max(int(np.floor((1 - quantile) * len(values))), 0), len(values) - 1)
    pre_threshold = float(values[position])
    return np.asarray(score) > pre_threshold, pre_threshold


@dataclass
class AlarmVerifier:
    """Two per-branch heads plus the candidate region they were trained on."""

    columns: tuple[str, ...]
    models: dict
    pre_thresholds: dict
    uses_early: bool

    def verify(self, features, scores):
        """Verification probability per branch; rows outside the candidate region get 1.0.

        Returning 1.0 outside the candidate region is what keeps the verifier
        one-directional: a row the verifier was never trained to judge is left
        exactly as the specialist left it.
        """
        result = {}
        for branch in BRANCHES:
            probability = np.ones(len(features), dtype=float)
            inside = np.asarray(scores[f"final_{branch}"]) > self.pre_thresholds[branch]
            if inside.any():
                probability[inside] = self.models[branch].predict_proba(
                    features[inside])[:, 1]
            result[branch] = probability
        return result

    def keep(self, features, scores, cutoffs):
        """Boolean keep-mask per branch at the given cutoffs."""
        probability = self.verify(features, scores)
        return {branch: probability[branch] >= float(cutoffs[branch])
                for branch in BRANCHES}
