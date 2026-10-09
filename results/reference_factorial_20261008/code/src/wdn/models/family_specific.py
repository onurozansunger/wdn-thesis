"""Separate fast/persistent experts for drift and injected-noise attacks."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


FAMILY_IDS = {"drift": 3, "noise": 4}
CATEGORY_MASS = {"positive": .45, "hard": .30, "clean": .15, "other": .10}


def _balanced_weights(arrays, family_id, early_multiplier, seed=911, category_mass=None,
                      other_sample=30000):
    """Training weights for one family expert.

    ``category_mass`` splits the total weight between this family's positives,
    its own event-period negatives, clean rows and *other mechanisms' rows*.
    The default keeps the promoted experts' behaviour. Raising the ``other``
    share teaches an expert to stay quiet inside another mechanism's event,
    which is what a mixture of experts needs and what a shared false-alarm
    budget punishes hardest.
    """
    mass = dict(CATEGORY_MASS if category_mass is None else category_mass)
    if set(mass) != set(CATEGORY_MASS) or abs(sum(mass.values()) - 1.) > 1e-9:
        raise ValueError("Category mass must cover every category and sum to one")
    labels, families = arrays["labels"], arrays["families"]
    event, scenario, early = arrays["event"], arrays["scenario"], arrays["early"]
    positive = np.flatnonzero((families == family_id) & (labels > 0))
    hard = np.flatnonzero((families == family_id) & (labels == 0))
    clean_all = np.flatnonzero((families == 0) & (labels == 0))
    other_all = np.flatnonzero((families != 0) & (families != family_id))
    if not len(positive) or not len(hard) or not len(clean_all) or not len(other_all):
        raise ValueError("Every family expert requires positive, hard, clean and other-family rows")
    rng = np.random.default_rng(seed+family_id)
    clean = rng.choice(clean_all, min(60000, len(clean_all)), replace=False)
    other = rng.choice(other_all, min(int(other_sample), len(other_all)), replace=False)
    selected = np.concatenate((positive, hard, clean, other))
    weights = np.zeros(len(selected))
    sections = {"positive": slice(0, len(positive)),
        "hard": slice(len(positive), len(positive)+len(hard)),
        "clean": slice(len(positive)+len(hard), len(positive)+len(hard)+len(clean)),
        "other": slice(len(positive)+len(hard)+len(clean), len(selected))}

    pos_events = np.unique(event[positive])
    if np.any(pos_events < 0):
        raise ValueError("Positive family rows require event identity")
    for event_id in pos_events:
        local = sections["positive"]
        loc = event[selected[local]] == event_id
        raw = np.where(early[selected[local]][loc], early_multiplier, 1.0)
        weights[np.arange(len(selected))[local][loc]] = (
            mass["positive"]/len(pos_events))*raw/raw.sum()

    hard_events = np.unique(event[hard])
    if np.any(hard_events < 0):
        raise ValueError("Hard negatives require active event identity")
    for event_id in hard_events:
        local = sections["hard"]
        loc = event[selected[local]] == event_id
        weights[np.arange(len(selected))[local][loc]] = (
            mass["hard"]/len(hard_events))/loc.sum()

    clean_scenarios = np.unique(scenario[clean])
    for sid in clean_scenarios:
        local = sections["clean"]
        loc = scenario[selected[local]] == sid
        weights[np.arange(len(selected))[local][loc]] = (
            mass["clean"]/len(clean_scenarios))/loc.sum()

    local = sections["other"]
    groups = np.column_stack((families[selected[local]], event[selected[local]]))
    unique_groups = np.unique(groups, axis=0)
    for group in unique_groups:
        loc = np.all(groups == group, axis=1)
        weights[np.arange(len(selected))[local][loc]] = (
            mass["other"]/len(unique_groups))/loc.sum()
    if not np.isclose(weights.sum(), 1.0) or np.any(weights <= 0):
        raise FloatingPointError("Invalid family expert training weights")
    return selected, weights*len(selected)


def _profile(names, family, head):
    exact = {
        ("drift", "fast"): {"residual", "abs_residual", "residual_rate", "normal_error_scale",
            "last_gap", "reference_support", "dynamic_innovation", "dynamic_abs_innovation",
            "dynamic_cusum_positive", "dynamic_cusum_negative", "seq_innovation", "seq_abs_innovation"},
        ("noise", "fast"): {"abs_residual", "normal_error_scale", "last_gap", "reference_support",
            "dynamic_innovation", "dynamic_abs_innovation", "seq_abs_innovation"},
    }.get((family, head), set())
    prefixes = {
        ("drift", "fast"): ("state_signed_ewm_", "state_abs_ewm_", "state_peak_",
            "state_exceedance_", "state_cusum_"),
        ("drift", "persistent"): ("mean_", "slope_", "sign_consistency_", "drift_",
            "dynamic_level_shift_", "dynamic_innovation_mean_", "dynamic_innovation_rms_",
            "state_mean_strength_", "state_sign_consistency_", "state_slope_strength_",
            "state_cusum_", "state_support_"),
        ("noise", "fast"): ("noise_state_", "state_abs_ewm_2", "state_energy_ewm_2",
            "state_peak_", "state_exceedance_"),
        ("noise", "persistent"): ("std_", "rms_", "noise_energy_", "noise_state_",
            "dynamic_innovation_rms_", "dynamic_past_std_", "state_energy_",
            "state_energy_ewm_", "state_exceedance_", "state_support_"),
    }[(family, head)]
    columns = np.asarray([index for index, name in enumerate(names)
                          if name in exact or name.startswith(prefixes)], dtype=int)
    if not len(columns):
        raise ValueError(f"Empty {family}/{head} feature profile")
    return columns


class FamilySpecificExpert:
    """One family's fast and persistent nonlinear subexperts."""

    def __init__(self, names, family, seed=911):
        if family not in FAMILY_IDS:
            raise ValueError("family must be drift or noise")
        self.names, self.family, self.family_id, self.seed = tuple(names), family, FAMILY_IDS[family], seed
        if len(set(self.names)) != len(self.names):
            raise ValueError("Feature schema must be unique")
        self.fast_columns = _profile(self.names, family, "fast")
        self.persistent_columns = _profile(self.names, family, "persistent")

    def fit(self, arrays):
        self.models = []
        for head, columns, early_multiplier, leaves, minimum in (
                ("fast", self.fast_columns, 4.0, 7, 15),
                ("persistent", self.persistent_columns, 1.0, 15, 20)):
            selected, weights = _balanced_weights(
                arrays, self.family_id, early_multiplier, self.seed)
            target = ((arrays["families"][selected] == self.family_id)
                      & (arrays["labels"][selected] > 0))
            model = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=leaves,
                min_samples_leaf=minimum, learning_rate=.05, l2_regularization=20.,
                early_stopping=False, random_state=self.seed)
            model.fit(arrays["X"][selected][:, columns], target, sample_weight=weights)
            self.models.append(model)
        return self

    def predict_heads(self, X):
        if len(getattr(self, "models", [])) != 2:
            raise RuntimeError("FamilySpecificExpert must be fitted before prediction")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.names) or not np.isfinite(X).all():
            raise ValueError("X does not match the family expert feature schema")
        return np.column_stack([model.predict_proba(X[:, columns])[:, 1]
            for model, columns in zip(self.models, (self.fast_columns, self.persistent_columns))])

    def metadata(self):
        return {"family": self.family, "family_id": self.family_id,
            "fast_features": [self.names[index] for index in self.fast_columns],
            "persistent_features": [self.names[index] for index in self.persistent_columns],
            "fixed_model": {"iterations": 120, "learning_rate": .05,
                "l2_regularization": 20., "fast_leaves": 7, "fast_min_leaf": 15,
                "persistent_leaves": 15, "persistent_min_leaf": 20},
            "category_mass": CATEGORY_MASS, "fast_early_multiplier": 4.0}


class MonotoneFamilyStacker:
    """Nonnegative logistic combination of fast and persistent head scores."""

    def __init__(self, family, regularisation=.1, seed=911):
        if family not in FAMILY_IDS or regularisation <= 0:
            raise ValueError("Invalid family stacker configuration")
        self.family, self.family_id = family, FAMILY_IDS[family]
        self.regularisation, self.seed = regularisation, seed

    @staticmethod
    def _logits(scores):
        scores = np.asarray(scores, dtype=float)
        if scores.ndim != 2 or scores.shape[1] != 2 or not np.isfinite(scores).all():
            raise ValueError("Stacker requires two finite head scores")
        clipped = np.clip(scores, 1e-6, 1-1e-6)
        return np.log(clipped/(1-clipped))

    def fit(self, scores, arrays):
        selected, weights = _balanced_weights(arrays, self.family_id, 1.0, self.seed)
        X = self._logits(scores[selected])
        y = ((arrays["families"][selected] == self.family_id)
             & (arrays["labels"][selected] > 0)).astype(float)
        weights = weights/weights.sum()
        self.scaler = StandardScaler().fit(X, sample_weight=weights)
        X = self.scaler.transform(X)

        def objective(theta):
            linear = X@theta[:2]+theta[2]
            error = weights*(expit(linear)-y)
            loss = (np.dot(weights, np.logaddexp(0., linear)-y*linear)
                    +.5*self.regularisation*np.dot(theta[:2], theta[:2]))
            gradient = np.r_[X.T@error+self.regularisation*theta[:2], error.sum()]
            return loss, gradient

        fitted = minimize(objective, np.zeros(3), jac=True, method="L-BFGS-B",
            bounds=[(0., None), (0., None), (None, None)],
            options={"maxiter": 700, "ftol": 1e-11})
        if not fitted.success:
            raise RuntimeError(f"Family stacker failed: {fitted.message}")
        self.coef_, self.intercept_, self.iterations_ = fitted.x[:2], float(fitted.x[2]), fitted.nit
        return self

    def predict(self, scores):
        if not hasattr(self, "coef_"):
            raise RuntimeError("MonotoneFamilyStacker must be fitted before prediction")
        X = self.scaler.transform(self._logits(scores))
        return expit(X@self.coef_+self.intercept_)

    def metadata(self):
        return {"family": self.family, "regularisation": self.regularisation,
            "nonnegative_coefficients": self.coef_.tolist(),
            "intercept": self.intercept_, "iterations": int(self.iterations_)}
