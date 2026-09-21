"""Strongly pooled conditional normal-error model for cross-fitted residuals.

The model is deliberately small.  It estimates a conditional location and a
robust conditional scale from *normal* residual rows whose reference and
covariates were produced out of scenario.  It never accepts labels or attack
families, and its covariates are restricted to a reviewed, target-value-blind
allow-list.  Scenario identifiers are used only to give every training
scenario equal total weight; node identifiers only index shrunken offsets.

This class does not construct cross-fitted rows.  The caller is responsible
for fitting every upstream reference/feature transform without the scenario
represented by each row.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


# These names were reviewed against the feature construction code.  They are
# functions of masks, fixed TRAIN scale, or target-group-blind reference
# diagnostics.  In particular, no current/past target value or residual is in
# this list.  Keeping an explicit allow-list is safer than trying to blacklist
# names such as residual, mean, CUSUM, or noise after the fact.
SAFE_COVARIATES = (
    "normal_reference_level_m",
    "normal_reference_delta_1_m",
    "normal_reference_delta_4_m",
    "normal_reference_past_mean_4_m",
    "reference_support",
    "reference_disagreement_m",
    "normal_error_scale",
    "normal_hour_sin",
    "normal_hour_cos",
    "coverage_4",
    "coverage_8",
    "coverage_16",
    "lag_support",
    "last_gap",
    "dynamic_past_support",
    "dynamic_forecast_support_8",
    "dynamic_forecast_support_16",
    "dynamic_forecast_support_32",
    "dynamic_forecast_support_48",
    "reference_disagreement",
    "seq_seen",
    "seq_support_3",
    "seq_support_5",
    "seq_support_8",
    "seq_support_12",
    "seq_support_16",
    "seq_physical_weight",
    "seq_physical_anchors",
    "seq_physical_sigma",
    "seq_physical_jackknife",
)


DEFAULT_COVARIATES = (
    "normal_reference_level_m",
    "normal_reference_delta_1_m",
    "normal_reference_delta_4_m",
    "normal_reference_past_mean_4_m",
    "reference_support",
    "normal_error_scale",
    "reference_disagreement_m",
    "normal_hour_sin",
    "normal_hour_cos",
)


@dataclass(frozen=True)
class ConditionalNormalConfig:
    """Transparent regularisation and clipping choices.

    Weights have total mass equal to the number of training scenarios, rather
    than the number of correlated sensor-hour rows.  Consequently the ridge
    and node penalties below express deliberately strong pooling.
    """

    covariates: tuple[str, ...] = DEFAULT_COVARIATES
    feature_l2: float = 5.0
    mean_node_shrinkage: float = 0.25
    scale_node_shrinkage: float = 0.50
    irls_iterations: int = 8
    huber_cutoff: float = 1.5
    covariate_clip: float = 4.0
    residual_clip: float = 6.0
    mean_clip_in_scales: float = 5.0
    maximum_scale_ratio: float = 10.0
    minimum_scale: float = 1e-4
    fit_mean: bool = True
    conditional_scale: bool = False


def _weighted_median(values, weights):
    order = np.argsort(values, kind="stable")
    x, w = values[order], weights[order]
    return float(x[np.searchsorted(np.cumsum(w), 0.5*w.sum(), side="left")])


def _robust_scale(values, weights, floor):
    center = _weighted_median(values, weights)
    mad = _weighted_median(np.abs(values-center), weights)/0.6744897501960817
    if not np.isfinite(mad) or mad < floor:
        variance = np.dot(weights, (values-center)**2)/weights.sum()
        mad = np.sqrt(max(float(variance), floor**2))
    return max(float(mad), floor)


def _zero_centered_scale(values, weights, floor):
    """Robust Gaussian scale around the supplied predictive mean (zero)."""
    scale = _weighted_median(np.abs(values), weights)/0.6744897501960817
    if not np.isfinite(scale) or scale < floor:
        scale = np.sqrt(max(float(np.dot(weights, values**2)/weights.sum()), floor**2))
    return max(float(scale), floor)


class ConditionalNormalError:
    """Conditional mean/scale with ridge effects and shrunken node offsets.

    Parameters
    ----------
    feature_names:
        Complete ordered schema for ``X``.  Selected names must belong to the
        reviewed ``SAFE_COVARIATES`` tuple.
    config:
        Frozen, serialisable model choices.

    Notes
    -----
    ``fit`` intentionally has no label or family argument.  ``scenario`` is a
    grouping variable used only for equal-risk weights and is never available
    to ``predict``.  Unknown nodes at prediction time receive the pooled
    offset zero.
    """

    def __init__(self, feature_names, config=ConditionalNormalConfig()):
        self.feature_names = tuple(feature_names)
        self.config = config
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be nonempty and unique")
        if not config.covariates or len(set(config.covariates)) != len(config.covariates):
            raise ValueError("covariates must be nonempty and unique")
        unsafe = sorted(set(config.covariates)-set(SAFE_COVARIATES))
        missing = sorted(set(config.covariates)-set(self.feature_names))
        if unsafe:
            raise ValueError(f"Unreviewed target-value covariates are forbidden: {unsafe}")
        if missing:
            raise ValueError(f"Covariates absent from feature schema: {missing}")
        positive = (config.feature_l2, config.mean_node_shrinkage,
                    config.scale_node_shrinkage, config.huber_cutoff,
                    config.covariate_clip, config.residual_clip,
                    config.mean_clip_in_scales, config.maximum_scale_ratio,
                    config.minimum_scale)
        if any(not np.isfinite(x) or x <= 0 for x in positive):
            raise ValueError("All regularisation, clipping and scale settings must be positive")
        if config.irls_iterations < 1:
            raise ValueError("irls_iterations must be positive")
        if not isinstance(config.fit_mean, bool) or not isinstance(config.conditional_scale, bool):
            raise ValueError("fit_mean and conditional_scale must be boolean")
        self.columns = np.asarray([self.feature_names.index(name)
                                   for name in config.covariates], dtype=int)

    @property
    def covariate_names(self):
        return self.config.covariates

    def _check_fit_inputs(self, X, residual, node, scenario, sample_weight):
        X = np.asarray(X, dtype=float)
        residual = np.asarray(residual, dtype=float)
        node, scenario = np.asarray(node), np.asarray(scenario)
        n = len(residual)
        if (X.ndim != 2 or X.shape != (n, len(self.feature_names))
                or residual.ndim != 1 or node.shape != (n,) or scenario.shape != (n,)
                or not n):
            raise ValueError("Expected nonempty matching feature, residual, node and scenario rows")
        if not np.issubdtype(node.dtype, np.integer) or not np.issubdtype(scenario.dtype, np.integer):
            raise ValueError("node and scenario identifiers must be integers")
        selected = X[:, self.columns]
        if not np.isfinite(selected).all() or not np.isfinite(residual).all():
            raise ValueError("Selected covariates and residuals must be finite")
        if sample_weight is None:
            sample_weight = np.ones(n)
        else:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if sample_weight.shape != (n,) or not np.isfinite(sample_weight).all() or np.any(sample_weight <= 0):
                raise ValueError("sample_weight must contain matching positive finite values")
        # Equal total mass per independent scenario; custom weights only alter
        # the distribution within a scenario.
        weights = np.empty(n)
        for sid in np.unique(scenario):
            loc = scenario == sid
            weights[loc] = sample_weight[loc]/sample_weight[loc].sum()
        return X, residual, node.astype(np.int64), scenario.astype(np.int64), weights

    def _standardise(self, selected, weights):
        centers, scales = [], []
        for column in selected.T:
            center = _weighted_median(column, weights)
            scale = _robust_scale(column, weights, 1e-8)
            centers.append(center)
            scales.append(scale)
        centers, scales = np.asarray(centers), np.asarray(scales)
        z = np.clip((selected-centers)/scales, -self.config.covariate_clip,
                    self.config.covariate_clip)
        return z, centers, scales

    def _node_indices(self, node):
        positions = np.searchsorted(self.node_ids_, node)
        valid = positions < len(self.node_ids_)
        valid[valid] &= self.node_ids_[positions[valid]] == node[valid]
        return positions, valid

    def _fit_component(self, z, target, node_index, base_weight, node_shrinkage):
        design = np.column_stack([np.ones(len(z)), z])
        coefficients = np.zeros(design.shape[1])
        offsets = np.zeros(len(self.node_ids_))
        robust_weight = base_weight.copy()
        floor = max(self.config.minimum_scale, self.pooled_scale_*1e-3)
        for _ in range(self.config.irls_iterations):
            adjusted = target-offsets[node_index]
            gram = design.T@(robust_weight[:, None]*design)
            gram[1:, 1:] += np.eye(design.shape[1]-1)*self.config.feature_l2
            rhs = design.T@(robust_weight*adjusted)
            coefficients = np.linalg.solve(gram, rhs)
            remainder = target-design@coefficients
            numerator = np.bincount(node_index, weights=robust_weight*remainder,
                                    minlength=len(offsets))
            denominator = np.bincount(node_index, weights=robust_weight,
                                      minlength=len(offsets))+node_shrinkage
            offsets = numerator/denominator
            error = remainder-offsets[node_index]
            scale = _robust_scale(error, base_weight, floor)
            robust_weight = base_weight*np.minimum(
                1.0, self.config.huber_cutoff*scale/np.maximum(np.abs(error), 1e-12))
        return coefficients, offsets

    def fit(self, X, residual, *, node, scenario, sample_weight=None):
        """Fit from cross-fitted *normal-only* endpoint rows.

        ``node`` supplies sensor IDs for partial pooling. ``scenario`` only
        equalises scenario risk. Neither is a numerical prediction covariate.
        """
        X, residual, node, _, weights = self._check_fit_inputs(
            X, residual, node, scenario, sample_weight)
        self.node_ids_ = np.unique(node)
        node_index = np.searchsorted(self.node_ids_, node)
        selected = X[:, self.columns]
        z, self.feature_center_, self.feature_scale_ = self._standardise(selected, weights)

        center = _weighted_median(residual, weights)
        self.pooled_scale_ = _robust_scale(residual-center, weights,
                                           self.config.minimum_scale)
        self.mean_clip_m_ = self.config.mean_clip_in_scales*self.pooled_scale_
        clipped = np.clip(residual, center-self.config.residual_clip*self.pooled_scale_,
                          center+self.config.residual_clip*self.pooled_scale_)
        if self.config.fit_mean:
            self.mean_coef_, self.mean_node_offset_ = self._fit_component(
                z, clipped, node_index, weights, self.config.mean_node_shrinkage)
        else:
            self.mean_coef_ = np.zeros(z.shape[1]+1)
            self.mean_node_offset_ = np.zeros(len(self.node_ids_))

        fitted_mean = np.column_stack([np.ones(len(z)), z])@self.mean_coef_
        fitted_mean += self.mean_node_offset_[node_index]
        fitted_mean = np.clip(fitted_mean, -self.mean_clip_m_, self.mean_clip_m_)
        error = residual-fitted_mean
        self.pooled_scale_ = _zero_centered_scale(
            error, weights, self.config.minimum_scale)
        self.scale_coef_ = np.zeros(z.shape[1]+1)
        if self.config.conditional_scale:
            absolute = np.abs(error)
            gaussian_median = 0.6744897501960817*self.pooled_scale_
            floor = max(self.config.minimum_scale, gaussian_median/20)
            log_ratio = np.log(np.maximum(absolute, floor)/gaussian_median)
            limit = np.log(self.config.maximum_scale_ratio)
            log_ratio = np.clip(log_ratio, -limit, limit)
            scale_coef, self.scale_node_offset_ = self._fit_component(
                z, log_ratio, node_index, weights, self.config.scale_node_shrinkage)
            # Correct the Gaussian E[log|Z|] offset for a log-scale regression.
            self.scale_coef_[0] = scale_coef[0]+0.2415644752704905
            self.scale_coef_[1:] = scale_coef[1:]
        else:
            self.scale_node_offset_ = np.zeros(len(self.node_ids_))
            for index in range(len(self.node_ids_)):
                loc = node_index == index
                mass = weights[loc].sum()
                local = _zero_centered_scale(
                    error[loc], weights[loc], self.config.minimum_scale)
                shrink = mass/(mass+self.config.scale_node_shrinkage)
                ratio = np.clip(local/self.pooled_scale_,
                                1/self.config.maximum_scale_ratio,
                                self.config.maximum_scale_ratio)
                self.scale_node_offset_[index] = shrink*np.log(ratio)
        self.fitted_ = True
        return self

    def _prediction_design(self, X):
        if not getattr(self, "fitted_", False):
            raise RuntimeError("ConditionalNormalError must be fitted before prediction")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError("X does not match the fitted feature schema")
        selected = X[:, self.columns]
        if not np.isfinite(selected).all():
            raise ValueError("Selected covariates must be finite")
        z = np.clip((selected-self.feature_center_)/self.feature_scale_,
                    -self.config.covariate_clip, self.config.covariate_clip)
        return np.column_stack([np.ones(len(z)), z])

    def predict(self, X, *, node):
        """Return finite conditional ``(mean, scale)`` in residual units."""
        design = self._prediction_design(X)
        node = np.asarray(node)
        if node.shape != (len(design),) or not np.issubdtype(node.dtype, np.integer):
            raise ValueError("node must provide one integer identifier per row")
        positions, known = self._node_indices(node.astype(np.int64))
        mean_offset, scale_offset = np.zeros(len(design)), np.zeros(len(design))
        mean_offset[known] = self.mean_node_offset_[positions[known]]
        scale_offset[known] = self.scale_node_offset_[positions[known]]
        mean = design@self.mean_coef_+mean_offset
        mean = np.clip(mean, -self.mean_clip_m_, self.mean_clip_m_)
        log_scale = design@self.scale_coef_+scale_offset
        limit = np.log(self.config.maximum_scale_ratio)
        scale = self.pooled_scale_*np.exp(np.clip(log_scale, -limit, limit))
        scale = np.clip(scale, self.config.minimum_scale,
                        self.pooled_scale_*self.config.maximum_scale_ratio)
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise FloatingPointError("Nonfinite conditional normal prediction")
        return mean, scale

    def standardise(self, X, residual, *, node):
        """Return conditional z, mean and scale for explicitly supplied residuals."""
        residual = np.asarray(residual, dtype=float)
        if residual.shape != (len(X),) or not np.isfinite(residual).all():
            raise ValueError("residual must be a matching finite vector")
        mean, scale = self.predict(X, node=node)
        return (residual-mean)/scale, mean, scale

    def metadata(self):
        """Return a JSON-serialisable audit description (no fitted values)."""
        return {
            "model": type(self).__name__,
            "config": asdict(self.config),
            "feature_schema": list(self.feature_names),
            "selected_covariates": list(self.config.covariates),
            "selected_columns": self.columns.tolist(),
            "fit_contract": "cross-fitted normal-only residual rows",
            "scenario_usage": "equal-risk weighting only; never a covariate",
            "node_usage": "strongly shrunken offset only; unseen nodes use pooled offset",
            "forbidden_inputs": ["label", "attack family", "current target residual as covariate"],
        }
