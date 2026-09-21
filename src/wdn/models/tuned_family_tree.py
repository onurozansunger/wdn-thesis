"""Configurable nonlinear drift/noise trees for bounded TRAIN-only tuning."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from wdn.models.residual_hybrid import training_weights
from wdn.probe_residual_experts import feature_profiles


@dataclass(frozen=True)
class TunedFamilyTreeConfig:
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 30
    learning_rate: float = .08
    l2_regularization: float = 10.
    max_iter: int = 120
    seed: int = 2421

    def validate(self):
        if (self.max_leaf_nodes < 2 or self.min_samples_leaf < 2
                or self.learning_rate <= 0 or self.l2_regularization < 0
                or self.max_iter < 1):
            raise ValueError("Invalid tuned family-tree configuration")


class TunedFamilyTrees:
    """Only the drift/noise heads from the nonlinear residual tree recipe."""

    def __init__(self, names, config=TunedFamilyTreeConfig()):
        config.validate()
        self.names, self.config = tuple(names), config
        profiles = feature_profiles(names)
        dynamic = [i for i, name in enumerate(names) if name.startswith("dynamic_")]
        reliability = [i for i, name in enumerate(names)
                       if name.startswith("reference_") or name == "normal_error_scale"]
        self.profiles = {family_id: np.unique(np.r_[profiles[family_id], dynamic, reliability]).astype(int)
                         for family_id in (3, 4)}

    def fit(self, arrays):
        y, family = np.asarray(arrays["labels"]), np.asarray(arrays["families"])
        rng = np.random.default_rng(self.config.seed)
        negatives = np.flatnonzero(y == 0)
        subset = np.r_[np.flatnonzero(y > 0),
                       rng.choice(negatives, min(60_000, len(negatives)), replace=False)]
        self.models = {}
        for family_id in (3, 4):
            domain = (y[subset] == 0) | (family[subset] == family_id)
            selected = subset[domain]
            weights = training_weights(y, family, arrays["event"], arrays["scenario"], selected)
            model = HistGradientBoostingClassifier(
                max_iter=self.config.max_iter,
                max_leaf_nodes=self.config.max_leaf_nodes,
                min_samples_leaf=self.config.min_samples_leaf,
                learning_rate=self.config.learning_rate,
                l2_regularization=self.config.l2_regularization,
                early_stopping=False, random_state=self.config.seed)
            model.fit(arrays["X"][selected][:, self.profiles[family_id]], y[selected],
                      sample_weight=weights)
            self.models[family_id] = model
        return self

    def predict(self, X):
        if set(getattr(self, "models", {})) != {3, 4}:
            raise RuntimeError("TunedFamilyTrees must be fitted first")
        return np.column_stack([self.models[family_id].predict_proba(
            X[:, self.profiles[family_id]])[:, 1] for family_id in (3, 4)])

    def metadata(self):
        return {"families": ["drift", "noise"], "config": asdict(self.config),
                "feature_counts": {str(key): len(value) for key, value in self.profiles.items()}}
