"""Heterogeneous residual experts and out-of-fold anomaly-score fusion."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from wdn.probe_residual_experts import feature_profiles


MECHANISMS = ("general", "abrupt", "replay", "drift", "noise")


def training_weights(labels, families, events, scenarios, selected):
    """Equal family/event risk for positives, equal scenario risk for normals."""
    y, f, e, sid = (np.asarray(a)[selected] for a in (labels, families, events, scenarios))
    weights = np.zeros(len(selected))
    pos_families = np.unique(f[y > 0])
    for family in pos_families:
        event_ids = np.unique(e[(y > 0) & (f == family)])
        for event in event_ids:
            mask = (y > 0) & (f == family) & (e == event)
            weights[mask] = .5/(len(pos_families)*len(event_ids)*mask.sum())
    normal_scenarios = np.unique(sid[y == 0])
    for scenario in normal_scenarios:
        mask = (y == 0) & (sid == scenario)
        weights[mask] = .5/(len(normal_scenarios)*mask.sum())
    return weights*len(weights)


def signed_log(X):
    return np.sign(X)*np.log1p(np.abs(X))


class EvidenceReadout:
    def __init__(self, C=.1):
        self.C = C

    def fit(self, X, y, sample_weight):
        self.scaler = StandardScaler().fit(signed_log(X), sample_weight=sample_weight)
        self.model = LogisticRegression(C=self.C, max_iter=1000, solver="lbfgs")
        self.model.fit(self.scaler.transform(signed_log(X)), y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(signed_log(X)))


class ResidualHybrid:
    def __init__(self, names, kind="mechanism", C=.1, seed=821):
        self.names, self.kind, self.C, self.seed = names, kind, C, seed
        profiles = feature_profiles(names)
        dynamic = [i for i, n in enumerate(names) if n.startswith("dynamic_")]
        reliability = [i for i, n in enumerate(names) if n.startswith("reference_") or n == "normal_error_scale"]
        for i in (1, 3, 4):
            profiles[i] = np.unique(np.r_[profiles[i], dynamic, reliability]).astype(int)
        if kind == "mechanism":
            common = {"residual", "abs_residual", "last_gap", "normal_error_scale", "dynamic_innovation", "dynamic_abs_innovation"}
            for i, prefix in ((3, "drift_"), (4, "noise_")):
                profiles[i] = np.array([j for j, n in enumerate(names) if n in common or n.startswith((prefix, "seq_", "reference_"))])
        elif kind != "trees":
            raise ValueError("Unknown expert recipe")
        self.profiles = profiles

    def fit(self, a):
        y, family = a["labels"], a["families"]
        rng = np.random.default_rng(self.seed)
        negatives = np.flatnonzero(y == 0)
        subset = np.r_[np.flatnonzero(y > 0), rng.choice(negatives, min(60000, len(negatives)), replace=False)]
        self.experts = []
        for i, cols in enumerate(self.profiles):
            domain = np.ones(len(subset), dtype=bool) if i == 0 else ((y[subset] == 0) | np.isin(family[subset], {1: (1,5), 2: (2,), 3: (3,), 4: (4,)}[i]))
            selected = subset[domain]
            weights = training_weights(y, family, a["event"], a["scenario"], selected)
            if self.kind == "mechanism" and i in (3, 4):
                model = EvidenceReadout(self.C)
            else:
                model = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15,
                    min_samples_leaf=30, learning_rate=.08, l2_regularization=10.,
                    early_stopping=False, random_state=self.seed)
            model.fit(a["X"][selected][:, cols], y[selected], sample_weight=weights)
            self.experts.append(model)
        return self

    def predict(self, X):
        return np.column_stack([m.predict_proba(X[:, cols])[:, 1] for m, cols in zip(self.experts, self.profiles)])


class EvidenceFusion:
    def __init__(self, names, C=.1):
        self.names, self.C = names, C
        self.columns = np.array([i for i, name in enumerate(names) if name in (
            "reference_support", "reference_disagreement", "last_gap", "seq_seen",
            "dynamic_abs_innovation", "coverage_4", "coverage_16")])

    def transform(self, scores, X):
        clipped = np.clip(scores, 1e-6, 1-1e-6)
        logits = np.log(clipped/(1-clipped))
        return np.column_stack([logits, signed_log(X[:, self.columns])])

    def fit(self, scores, a):
        # Caller supplies only predictions from models/reference that did not
        # train on the corresponding scenario, never in-sample fitted scores.
        selected = np.arange(len(a["labels"]))
        weights = training_weights(a["labels"], a["families"], a["event"], a["scenario"], selected)
        X = self.transform(scores, a["X"])
        self.scaler = StandardScaler().fit(X, sample_weight=weights)
        self.model = LogisticRegression(C=self.C, max_iter=1000)
        self.model.fit(self.scaler.transform(X), a["labels"], sample_weight=weights)
        return self

    def predict(self, scores, X):
        return self.model.predict_proba(self.scaler.transform(self.transform(scores, X)))[:, 1]
