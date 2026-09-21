"""Nonlinear weak-family specialists and monotone, causal score stacking."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from wdn.models.residual_hybrid import signed_log, training_weights


def phase_weights(a, selected, multiplier):
    weights = training_weights(a["labels"], a["families"], a["event"], a["scenario"], selected)
    if multiplier <= 0:
        raise ValueError("Positive early-phase weight required")
    for event in np.unique(a["event"][selected][a["labels"][selected] > 0]):
        loc = (a["event"][selected] == event) & (a["labels"][selected] > 0)
        total = weights[loc].sum()
        weights[loc] *= np.where(a["early"][selected][loc], multiplier, 1.)
        weights[loc] *= total/weights[loc].sum()
    return weights


class WeakExperts:
    def __init__(self, names, leaves=15, regularisation=10., early_weight=1., seed=831):
        self.names, self.leaves = names, leaves
        self.regularisation, self.early_weight, self.seed = regularisation, early_weight, seed
        self.profiles = []
        common = {"residual", "abs_residual", "normal_error_scale", "reference_support", "reference_disagreement", "last_gap", "residual_rate"}
        for family in (3, 4):
            prefixes = ("dynamic_", "seq_", "coverage_") + (("drift_", "mean_", "slope_", "sign_consistency_") if family == 3 else ("noise_", "std_", "rms_", "max_abs_", "mean_"))
            columns = []
            for i, name in enumerate(names):
                base = name.removeprefix("joint_")
                if base in common or base.startswith(prefixes) or name.startswith("context_"):
                    columns.append(i)
            self.profiles.append(np.asarray(columns))

    def fit(self, a):
        rng = np.random.default_rng(self.seed)
        negatives = np.flatnonzero(a["labels"] == 0)
        sampled = rng.choice(negatives, min(60000, len(negatives)), replace=False)
        self.experts = []
        for family, cols in zip((3, 4), self.profiles):
            positives = np.flatnonzero((a["labels"] > 0) & (a["families"] == family))
            if not len(positives):
                raise ValueError("Missing weak-family training positives")
            selected = np.r_[positives, sampled]
            weights = phase_weights(a, selected, self.early_weight)
            model = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=self.leaves,
                min_samples_leaf=25, learning_rate=.07, l2_regularization=self.regularisation,
                early_stopping=False, random_state=self.seed)
            model.fit(a["X"][selected][:, cols], a["labels"][selected], sample_weight=weights)
            self.experts.append(model)
        return self

    def predict(self, X):
        return np.column_stack([model.predict_proba(X[:, cols])[:, 1]
                                for model, cols in zip(self.experts, self.profiles)])


def score_memory(scores, scenario, timestep, node):
    """Current + decaying maximum + EWMA; all updates use observed endpoints.

    Scenario IDs only reset state, never become numerical model inputs.
    Missing hours decay the state; no unseen observation or true event reset.
    """
    scores = np.asarray(scores, dtype=float)
    scenario, timestep, node = map(np.asarray, (scenario, timestep, node))
    if scores.ndim != 2 or any(len(x) != len(scores) for x in (scenario, timestep, node)):
        raise ValueError("Matching endpoint arrays required")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Finite probability scores required")
    output = np.empty((len(scores), scores.shape[1]*3))
    for sid in np.unique(scenario):
        indices = np.flatnonzero(scenario == sid)
        indices = indices[np.lexsort((node[indices], timestep[indices]))]
        states = {}
        for index in indices:
            sensor, t = int(node[index]), int(timestep[index])
            if sensor in states:
                last, peak, mean = states[sensor]
                if t <= last:
                    raise ValueError("Duplicate or nonincreasing sensor endpoint")
                gap = t-last
                peak = np.maximum(peak*np.exp(-gap/6.), scores[index])
                mean = .65*mean*np.exp(-(gap-1)/6.)+.35*scores[index]
            else:
                peak, mean = scores[index].copy(), scores[index].copy()
            output[index] = np.r_[scores[index], peak, mean]
            states[sensor] = (t, peak, mean)
    return output


class MonotoneFusion:
    def __init__(self, names, C=.1):
        self.names, self.C = names, C
        self.columns = np.asarray([i for i, name in enumerate(names) if name in
            ("reference_support", "reference_disagreement", "last_gap", "coverage_4", "coverage_16", "dynamic_abs_innovation")
            or name.startswith("context_")])

    def transform(self, scores, a):
        memory = score_memory(scores, a["scenario"], a["timestep"], a["node"])
        clipped = np.clip(memory, 1e-6, 1-1e-6)
        logits = np.log(clipped/(1-clipped))
        return np.column_stack([logits, signed_log(a["X"][:, self.columns])])

    def fit(self, scores, a):
        x = self.transform(scores, a)
        selected = np.arange(len(a["labels"]))
        weights = training_weights(a["labels"], a["families"], a["event"], a["scenario"], selected)
        self.scaler = StandardScaler().fit(x, sample_weight=weights)
        x = self.scaler.transform(x)
        y = a["labels"]
        weights = weights/weights.sum()
        penalty = 1./(self.C*len(y))
        def objective(theta):
            score = x@theta[:-1]+theta[-1]
            error = weights*(expit(score)-y)
            loss = np.dot(weights, np.logaddexp(0., score)-y*score)+.5*penalty*np.dot(theta[:-1], theta[:-1])
            gradient = np.r_[x.T@error+penalty*theta[:-1], error.sum()]
            return loss, gradient
        monotone_count = scores.shape[1]*3
        fit = minimize(objective, np.zeros(x.shape[1]+1), jac=True, method="L-BFGS-B",
                       bounds=[(0., None)]*monotone_count+[(None, None)]*(x.shape[1]+1-monotone_count),
                       options={"maxiter": 700, "ftol": 1e-10})
        if not fit.success:
            raise RuntimeError(f"Monotone fusion did not converge: {fit.message}")
        self.coef_, self.intercept_ = fit.x[:-1], float(fit.x[-1])
        self.monotone_count_, self.iterations_ = monotone_count, fit.nit
        return self

    def predict(self, scores, a):
        return expit(self.scaler.transform(self.transform(scores, a))@self.coef_+self.intercept_)
