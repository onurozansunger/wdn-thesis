"""Full drift and fast-reset noise experts with daily seasonal evidence."""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier

from wdn.models.family_specific import _balanced_weights, _profile


class SeasonalFamilyExperts:
    def __init__(self, names, seed=4600, category_mass=None, other_sample=30000):
        self.names, self.seed = tuple(names), int(seed)
        self.category_mass, self.other_sample = category_mass, int(other_sample)
        seasonal = np.asarray([i for i, name in enumerate(names) if name.startswith("seasonal_")])
        if len(seasonal) != 7:
            raise ValueError("All seven seasonal pressure features are required")
        base_names = [name for name in names if not name.startswith("seasonal_")]
        fast = _profile(base_names, "noise", "fast")
        self.noise_fast_columns = np.unique(np.r_[fast, seasonal]).astype(int)
        self.all_columns = np.arange(len(names))

    @staticmethod
    def _model(seed, *, full):
        return LGBMClassifier(n_estimators=400 if full else 350,
            num_leaves=31 if full else 23, min_child_samples=20,
            learning_rate=.03 if full else .035, reg_lambda=20., reg_alpha=1.,
            colsample_bytree=.85 if full else .90, random_state=seed, n_jobs=1,
            verbosity=-1, deterministic=True, force_col_wise=True)

    def fit(self, arrays):
        self.models = {}
        for name, family_id, columns, full, seed_offset in (
                ("drift", 3, self.all_columns, True, 0),
                ("noise_full", 4, self.all_columns, True, 0),
                ("noise_fast", 4, self.noise_fast_columns, False, 200)):
            selected, weights = _balanced_weights(
                arrays, family_id, 2., self.seed,
                category_mass=self.category_mass, other_sample=self.other_sample)
            target = ((arrays["families"][selected] == family_id)
                      & (arrays["labels"][selected] > 0)).astype(int)
            model = self._model(self.seed + seed_offset, full=full)
            # Avoid two consecutive advanced-indexing copies for the full models.
            # A fold contains about one million endpoints, so ``X[selected][:,
            # columns]`` can transiently duplicate the entire ~450 MB matrix and
            # make LightGBM die in native code under memory pressure.
            if len(columns) == arrays["X"].shape[1]:
                fit_X = arrays["X"][selected]
            else:
                fit_X = arrays["X"][np.ix_(selected, columns)]
            model.fit(fit_X, target, sample_weight=weights)
            self.models[name] = model
        return self

    def predict(self, X):
        if set(getattr(self, "models", {})) != {"drift", "noise_full", "noise_fast"}:
            raise RuntimeError("SeasonalFamilyExperts must be fitted first")
        return {"drift": self.models["drift"].predict_proba(X)[:, 1],
            "noise_full": self.models["noise_full"].predict_proba(X)[:, 1],
            "noise_fast": self.models["noise_fast"].predict_proba(
                X[:, self.noise_fast_columns])[:, 1]}

    def metadata(self):
        return {"architecture": "daily seasonal full drift plus fast-reset noise trees",
            "category_mass": self.category_mass, "other_sample": self.other_sample,
            "uses_future": False, "uses_event_boundaries": False,
            "feature_count": len(self.names),
            "noise_fast_feature_count": len(self.noise_fast_columns)}
