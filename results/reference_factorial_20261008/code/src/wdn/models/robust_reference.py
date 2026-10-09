"""Multi-view robust pressure reference with strict target-group blindness."""
from __future__ import annotations

import numpy as np


class RobustBlindReference:
    def __init__(self, reference, views=3, steps=6, cutoff=4.0, seed=821):
        self.reference = reference
        self.views, self.steps, self.cutoff, self.seed = views, steps, cutoff, seed
        self.noise_scale_ = reference.noise_scale_.copy()
        if views < 1 or steps < 1 or cutoff <= 0:
            raise ValueError("Invalid robust reference settings")
        self.subsets_ = []
        rng = np.random.default_rng(seed)
        for group in range(reference.groups):
            visible = reference.group_ != group
            subsets = [visible.copy()]
            for _ in range(views-1):
                keep = visible & (rng.random(len(visible)) < .8)
                if keep.sum() <= reference.rank:
                    keep = visible.copy()
                subsets.append(keep)
            self.subsets_.append(subsets)

    def predict_details(self, values, observed):
        ref = self.reference
        values = np.asarray(values, dtype=float)
        observed = np.asarray(observed, dtype=bool)
        if values.ndim != 2 or observed.shape != values.shape or values.shape[1] != len(ref.mean_):
            raise ValueError("Expected matching arrays in fitted sensor order")
        if not np.isfinite(values[observed]).all():
            raise ValueError("Observed values must be finite")
        # Whiten physical observation errors, rather than weighting sensors by
        # the unrelated range of their hydraulic variation.
        sigma = ref.noise_scale_
        design = ref.basis_*ref.scale_[:, None]/sigma[:, None]
        response = np.where(observed, values-ref.mean_, 0.)/sigma
        prediction, spread, support = (np.zeros_like(values) for _ in range(3))
        for group, subsets in enumerate(self.subsets_):
            target = ref.group_ == group
            estimates = []
            for subset in subsets:
                visible = observed & subset[None, :]
                weights = visible.astype(float)
                for step in range(self.steps):
                    gram = np.einsum("tn,nr,ns->trs", weights, design, design, optimize=True)
                    gram += np.eye(ref.rank)[None]*ref.ridge
                    latent = np.linalg.solve(gram, ((weights*response)@design)[..., None])[..., 0]
                    fitted = latent@design.T
                    if step+1 < self.steps:
                        # Studentise residuals for leverage. Every input to
                        # this computation excludes the complete target group.
                        inverse = np.linalg.inv(gram)
                        leverage = np.einsum("nr,trs,ns->tn", design, inverse, design, optimize=True)*weights
                        error = np.abs(np.where(visible, response-fitted, 0.))/np.sqrt(np.maximum(1-leverage, .1))
                        if step < 2:
                            robust = np.minimum(1., 2.5/np.maximum(error, 1e-12))
                        else:
                            robust = np.maximum(0., 1-(error/self.cutoff)**2)**2
                            robust = np.maximum(robust, .01)
                        weights = visible*robust
                estimate = fitted[:, target]*sigma[target]+ref.mean_[target]
                estimates.append(estimate)
            estimates = np.asarray(estimates)
            center = np.median(estimates, axis=0)
            prediction[:, target] = center
            spread[:, target] = np.median(np.abs(estimates-center[None]), axis=0)
            support[:, target] = (observed & ~target[None]).sum(1)[:, None]/max(1, (~target).sum())
        return prediction, support, spread

    def calibrate_scale(self, normal_values, normal_observed):
        predicted, _, _ = self.predict_details(normal_values, normal_observed)
        errors = np.where(normal_observed, np.abs(normal_values-predicted), np.nan)
        scale = np.nanmedian(errors, axis=0)/.67448975
        if not np.isfinite(scale).all():
            raise ValueError("Every sensor requires normal training observations")
        self.noise_scale_ = np.maximum(scale, max(float(np.median(scale))*.1, 1e-6))
        return self

    def predict(self, values, observed):
        predicted, support, _ = self.predict_details(values, observed)
        return predicted, support
