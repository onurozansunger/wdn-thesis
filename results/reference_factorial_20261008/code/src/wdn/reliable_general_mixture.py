"""OOF-fitted probability calibration and replacement internal General router.

Inference still combines precisely five existing component scores. No family
label, event information, or additional anomaly detector enters inference.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, softmax


def logits(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class ReliableCombination:
    def fit(self, experts, routing, labels, weights):
        p, r, y = np.asarray(experts, float), np.asarray(routing, float), np.asarray(labels, float)
        w = np.asarray(weights, float)
        w = w / w.sum()
        if p.shape != r.shape or p.shape != (len(y), 5) or set(np.unique(y)) != {0., 1.}:
            raise ValueError('Five component and routing columns with both classes required')
        z = logits(p)
        self.calibration = []
        self.optimization = []
        for j in range(5):
            def loss(ab):
                linear = ab[0] * z[:, j] + ab[1]
                error = expit(linear) - y
                value = np.sum(w * (np.logaddexp(0, linear) - y * linear)) + 1e-5 * ((ab[0]-1)**2 + ab[1]**2)
                grad = np.array([np.sum(w * error * z[:, j]), np.sum(w * error)]) + 2e-5 * np.array([ab[0]-1, ab[1]])
                return value, grad
            result = minimize(loss, [1., 0.], jac=True, method='L-BFGS-B', bounds=[(.05, 5.), (-20., 20.)], options={'maxiter': 300, 'ftol': 1e-12})
            if not result.success: raise RuntimeError('Component calibration optimizer failed: ' + result.message)
            self.calibration.append(result.x)
            self.optimization.append(str(result.message))
        self.calibration = np.asarray(self.calibration)
        calibrated = expit(z * self.calibration[:, 0] + self.calibration[:, 1])
        raw_context = np.column_stack((logits(p), np.log(np.maximum(r, 1e-6))))
        self.center = np.sum(w[:, None] * raw_context, axis=0)
        self.scale = np.maximum(np.sqrt(np.sum(w[:, None] * (raw_context - self.center)**2, axis=0)), .1)
        x = np.column_stack((np.ones(len(y)), np.clip((raw_context - self.center) / self.scale, -8, 8)))
        offset = np.log(np.maximum(r, 1e-6))
        def loss(flat):
            coef = flat.reshape(x.shape[1], 5)
            gate = softmax(offset + x @ coef, axis=1)
            score = np.clip(np.sum(gate * calibrated, axis=1), 1e-9, 1-1e-9)
            value = -np.sum(w * (y * np.log(score) + (1-y) * np.log1p(-score))) + .001 * np.sum(coef**2)
            ds = w * (score-y) / (score * (1-score))
            grad = x.T @ (ds[:, None] * gate * (calibrated-score[:, None])) + .002 * coef
            return value, grad.ravel()
        result = minimize(loss, np.zeros(x.shape[1] * 5), jac=True, method='L-BFGS-B', options={'maxiter': 300, 'ftol': 1e-11})
        if not result.success: raise RuntimeError('Reliability router optimizer failed: ' + result.message)
        self.coef = result.x.reshape(x.shape[1], 5)
        self.optimization.append(str(result.message))
        return self

    def transform(self, prediction):
        p, r = prediction['experts'], prediction['routing']
        calibrated = expit(logits(p) * self.calibration[:, 0] + self.calibration[:, 1])
        context = np.column_stack((logits(p), np.log(np.maximum(r, 1e-6))))
        x = np.column_stack((np.ones(len(p)), np.clip((context-self.center)/self.scale, -8, 8)))
        gate = softmax(np.log(np.maximum(r, 1e-6)) + x @ self.coef, axis=1)
        return dict(experts=calibrated, routing=gate, mixture=np.sum(calibrated * gate, axis=1),
                    general=calibrated[:, 0], uniform=calibrated.mean(1))


class ReliableGeneralMixture:
    def __init__(self, base, combination):
        self.base, self.combination = base, combination
        self.names, self.seed, self.profiles = base.names, base.seed, base.profiles
        self.experts = base.experts

    def predict(self, X):
        return self.combination.transform(self.base.predict(X))
