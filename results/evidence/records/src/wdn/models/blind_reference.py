"""Train-only normal reference whose prediction cannot see its target group.

This is a diagnostic pressure model, not the production GNN. Sensor groups
are fixed independently of values/labels. Each prediction uses only observed
sensors outside its group, including during robust reweighting.
"""
from __future__ import annotations

import numpy as np


class BlindPressureReference:
    def __init__(self, rank=4, groups=4, iterations=12, ridge=1e-3, robust_steps=3):
        self.rank, self.groups, self.iterations = rank, groups, iterations
        self.ridge, self.robust_steps = ridge, robust_steps

    def fit(self, values, observed):
        values = np.asarray(values, dtype=np.float64)
        observed = np.asarray(observed, dtype=bool)
        if values.ndim != 2 or values.shape != observed.shape:
            raise ValueError("Expected matching time-by-sensor arrays")
        if not 1 <= self.rank < min(values.shape) or not 2 <= self.groups <= values.shape[1]:
            raise ValueError("Invalid rank or sensor groups")
        if np.any(observed.sum(0) < 2):
            raise ValueError("Every sensor needs at least two normal training observations")
        safe = np.where(observed, values, 0.)
        count = observed.sum(0)
        self.mean_ = safe.sum(0)/count
        centered = np.where(observed, values-self.mean_, 0.)
        spread = np.sqrt((centered**2).sum(0)/count)
        self.scale_ = np.maximum(spread, max(float(np.median(spread))*.01, 1e-6))
        z = centered/self.scale_
        filled = z.copy()
        # Iterative low-rank completion fits only normal TRAIN observations.
        # The observed entries are never replaced with simulator clean truth.
        for _ in range(self.iterations):
            _, _, vt = np.linalg.svd(filled, full_matrices=False)
            self.basis_ = vt[:self.rank].T
            reconstructed = (filled@self.basis_)@self.basis_.T
            filled = np.where(observed, z, reconstructed)
        self.group_ = np.arange(values.shape[1]) % self.groups
        self.noise_scale_ = None
        initial, _ = self.predict(values, observed)
        errors = np.where(observed, np.abs(values-initial), np.nan)
        sigma = np.nanmedian(errors, axis=0)/.67448975
        floor = max(float(np.median(sigma))*.1, 1e-6)
        self.noise_scale_ = np.maximum(sigma, floor)
        return self

    def predict(self, values, observed):
        values = np.asarray(values, dtype=np.float64)
        observed = np.asarray(observed, dtype=bool)
        if values.ndim != 2 or values.shape != observed.shape or values.shape[1] != len(self.mean_):
            raise ValueError("Expected matching arrays on fitted sensor order")
        z = np.where(observed, values-self.mean_, 0.)/self.scale_
        prediction, support = np.empty_like(values), np.zeros_like(values)
        for group in range(self.groups):
            target = self.group_ == group
            visible = observed & ~target[None, :]
            weights = visible.astype(float)
            for iteration in range(self.robust_steps+1):
                gram = np.einsum("tn,nr,ns->trs", weights, self.basis_, self.basis_)
                gram += np.eye(self.rank)[None]*self.ridge
                rhs = (weights*z)@self.basis_
                latent = np.linalg.solve(gram, rhs[..., None])[..., 0]
                fitted = (latent@self.basis_.T)*self.scale_+self.mean_
                if self.noise_scale_ is None or iteration == self.robust_steps:
                    break
                error = np.abs(np.where(visible, values-fitted, 0.))
                weights = visible*np.minimum(1., 2.5*self.noise_scale_/np.maximum(error, 1e-12))
            prediction[:, target] = fitted[:, target]
            support[:, target] = visible.sum(1)[:, None]/max(1, (~target).sum())
        return prediction, support
