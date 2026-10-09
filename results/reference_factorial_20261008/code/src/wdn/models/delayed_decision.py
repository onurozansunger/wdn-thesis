"""Weak-family experts that may read a declared number of future hours.

Same fixed LightGBM recipe as the promoted seasonal experts, same balanced
family weighting, same feature bank — plus the forward-window evidence of
`wdn.delayed_decision_features`. The only architectural change is that the
decision for hour ``t`` is finalised at ``t + delta``, which `metadata`
declares so an audit cannot mistake this for a strictly online detector.
"""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier

from wdn.models.family_specific import _balanced_weights


class DelayedDecisionExperts:
    def __init__(self, names, delta, seed=5100, category_mass=None, other_sample=30000):
        if int(delta) < 1:
            raise ValueError("A delayed decision needs at least one hour of latency")
        self.names, self.delta, self.seed = tuple(names), int(delta), int(seed)
        self.category_mass, self.other_sample = category_mass, int(other_sample)
        if not any(name.startswith("future_") for name in names):
            raise ValueError("The bank carries no forward-window evidence")
        if not any(name.startswith("seasonal_") for name in names):
            raise ValueError("The bank carries no daily seasonal evidence")

    @staticmethod
    def _model(seed):
        return LGBMClassifier(n_estimators=400, num_leaves=31, min_child_samples=20,
            learning_rate=.03, reg_lambda=20., reg_alpha=1., colsample_bytree=.85,
            random_state=seed, n_jobs=1, verbosity=-1, deterministic=True,
            force_col_wise=True)

    def fit(self, arrays):
        if arrays["X"].shape[1] != len(self.names):
            raise ValueError("Feature matrix does not match the declared bank")
        self.models = {}
        for name, family_id, seed_offset in (("drift", 3, 0), ("noise", 4, 100)):
            selected, weights = _balanced_weights(
                arrays, family_id, 2., self.seed,
                category_mass=self.category_mass, other_sample=self.other_sample)
            target = ((arrays["families"][selected] == family_id)
                      & (arrays["labels"][selected] > 0)).astype(int)
            model = self._model(self.seed + seed_offset)
            model.fit(arrays["X"][selected], target, sample_weight=weights)
            self.models[name] = model
        return self

    def predict(self, X):
        if set(getattr(self, "models", {})) != {"drift", "noise"}:
            raise RuntimeError("DelayedDecisionExperts must be fitted first")
        return {name: model.predict_proba(X)[:, 1] for name, model in self.models.items()}

    def metadata(self):
        return {"architecture": "seasonal bank plus bounded forward-window evidence",
            "uses_future": True, "decision_latency_hours": self.delta,
            "uses_event_boundaries": False, "uses_labels_at_inference": False,
            "category_mass": self.category_mass, "other_sample": self.other_sample,
            "feature_count": len(self.names),
            "forward_feature_count": sum(name.startswith("future_") or "_t+" in name
                                         for name in self.names)}
