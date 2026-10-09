"""Scenario-balanced causal AR(1) model for normal reference errors.

The caller supplies normal-only residual rows produced by an out-of-scenario
reference.  The model has no label/family argument.  Scenario IDs equalise fit
risk and node IDs only index a shrunken innovation scale.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class LatentARConfig:
    phi_maximum: float = 0.95
    irls_iterations: int = 8
    huber_cutoff: float = 1.5
    residual_clip: float = 6.0
    innovation_scale_floor_m: float = 0.05
    node_scale_shrinkage: float = 0.50
    maximum_scale_ratio: float = 10.0


def _weighted_median(values, weights):
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    return float(values[np.searchsorted(np.cumsum(weights), .5*weights.sum(), side="left")])


def _zero_scale(values, weights, floor):
    scale = _weighted_median(np.abs(values), weights)/0.6744897501960817
    if not np.isfinite(scale) or scale < floor:
        scale = np.sqrt(max(float(np.dot(weights, values**2)/weights.sum()), floor**2))
    return max(float(scale), floor)


class LatentARNormalError:
    """Fit one global AR coefficient and shrunken per-node innovation scales."""

    def __init__(self, config=LatentARConfig()):
        self.config = config
        positive = (config.phi_maximum, config.huber_cutoff, config.residual_clip,
                    config.innovation_scale_floor_m, config.node_scale_shrinkage,
                    config.maximum_scale_ratio)
        if any(not np.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("All AR configuration values must be positive and finite")
        if config.phi_maximum >= 1:
            raise ValueError("phi_maximum must be below one")
        if not isinstance(config.irls_iterations, int) or config.irls_iterations < 1:
            raise ValueError("irls_iterations must be a positive integer")

    @staticmethod
    def _validate_rows(residual, scenario, timestep, node):
        residual = np.asarray(residual, dtype=float)
        scenario, timestep, node = map(np.asarray, (scenario, timestep, node))
        n = len(residual)
        if (not n or residual.shape != (n,) or scenario.shape != (n,)
                or timestep.shape != (n,) or node.shape != (n,)):
            raise ValueError("Expected matching nonempty residual/scenario/timestep/node rows")
        if (not np.isfinite(residual).all()
                or not all(np.issubdtype(value.dtype, np.integer)
                           for value in (scenario, timestep, node))):
            raise ValueError("Residuals must be finite and identifiers must be integers")
        order = np.lexsort((timestep, node, scenario))
        scenario, timestep, node = (value[order].astype(np.int64)
                                    for value in (scenario, timestep, node))
        residual = residual[order]
        duplicate = ((scenario[1:] == scenario[:-1]) & (node[1:] == node[:-1])
                     & (timestep[1:] == timestep[:-1]))
        if np.any(duplicate):
            raise ValueError("AR fit rows must have unique scenario/timestep/node keys")
        return residual, scenario, timestep, node

    def fit(self, residual, *, scenario, timestep, node):
        residual, scenario, timestep, node = self._validate_rows(
            residual, scenario, timestep, node)
        adjacent = ((scenario[1:] == scenario[:-1]) & (node[1:] == node[:-1])
                    & (timestep[1:]-timestep[:-1] == 1))
        if not np.any(adjacent):
            raise ValueError("AR fitting requires consecutive within-sensor pairs")
        previous, current = residual[:-1][adjacent], residual[1:][adjacent]
        pair_scenario, pair_node = scenario[1:][adjacent], node[1:][adjacent]

        weights = np.empty(len(previous))
        for sid in np.unique(pair_scenario):
            selected = pair_scenario == sid
            weights[selected] = 1/selected.sum()
        pooled_residual_scale = _zero_scale(
            np.concatenate((previous, current)), np.concatenate((weights, weights)),
            self.config.innovation_scale_floor_m)
        limit = self.config.residual_clip*pooled_residual_scale
        previous, current = np.clip(previous, -limit, limit), np.clip(current, -limit, limit)

        denominator = np.dot(weights, previous**2)
        if denominator <= 0 or not np.isfinite(denominator):
            raise FloatingPointError("Degenerate AR predecessor values")
        phi = np.clip(np.dot(weights, previous*current)/denominator,
                      0.0, self.config.phi_maximum)
        robust_weight = weights.copy()
        for _ in range(self.config.irls_iterations):
            denominator = np.dot(robust_weight, previous**2)
            if denominator <= 0 or not np.isfinite(denominator):
                raise FloatingPointError("Degenerate robust AR fit")
            phi = float(np.clip(np.dot(robust_weight, previous*current)/denominator,
                                0.0, self.config.phi_maximum))
            innovation = current-phi*previous
            scale = _zero_scale(innovation, weights,
                                self.config.innovation_scale_floor_m)
            robust_weight = weights*np.minimum(
                1.0, self.config.huber_cutoff*scale/np.maximum(np.abs(innovation), 1e-12))

        innovation = current-phi*previous
        pooled = _zero_scale(innovation, weights,
                             self.config.innovation_scale_floor_m)
        node_ids = np.unique(node)
        offsets = np.zeros(len(node_ids))
        for index, node_id in enumerate(node_ids):
            selected = pair_node == node_id
            if not np.any(selected):
                continue
            local = _zero_scale(innovation[selected], weights[selected],
                                self.config.innovation_scale_floor_m)
            mass = weights[selected].sum()
            shrink = mass/(mass+self.config.node_scale_shrinkage)
            ratio = np.clip(local/pooled, 1/self.config.maximum_scale_ratio,
                            self.config.maximum_scale_ratio)
            offsets[index] = shrink*np.log(ratio)

        self.phi_ = phi
        self.pooled_innovation_scale_ = pooled
        self.node_ids_ = node_ids
        self.node_log_scale_offset_ = offsets
        self.fit_scenarios_ = np.unique(scenario)
        self.fit_pair_count_ = int(len(previous))
        self.fitted_ = True
        return self

    def _node_scales(self, node):
        node = np.asarray(node)
        if node.ndim != 1 or not np.issubdtype(node.dtype, np.integer):
            raise ValueError("node must be a one-dimensional integer array")
        positions = np.searchsorted(self.node_ids_, node)
        known = positions < len(self.node_ids_)
        known[known] &= self.node_ids_[positions[known]] == node[known]
        offsets = np.zeros(len(node))
        offsets[known] = self.node_log_scale_offset_[positions[known]]
        limit = np.log(self.config.maximum_scale_ratio)
        return np.maximum(self.config.innovation_scale_floor_m,
                          self.pooled_innovation_scale_*np.exp(np.clip(offsets, -limit, limit)))

    def predict_sequence(self, residual, observed, *, timestep=None, node=None):
        """Return causal conditional mean/scale before each observation update."""
        if not getattr(self, "fitted_", False):
            raise RuntimeError("LatentARNormalError must be fitted before prediction")
        residual, observed = np.asarray(residual, dtype=float), np.asarray(observed, dtype=bool)
        if residual.ndim != 2 or observed.shape != residual.shape:
            raise ValueError("residual and observed must be matching time-by-node arrays")
        if not np.isfinite(residual[observed]).all():
            raise ValueError("Observed residuals must be finite")
        t_count, n_nodes = residual.shape
        if timestep is None:
            timestep = np.arange(t_count)
        timestep = np.asarray(timestep)
        if (timestep.shape != (t_count,) or not np.issubdtype(timestep.dtype, np.integer)
                or np.any(np.diff(timestep) <= 0)):
            raise ValueError("timestep must be a strictly increasing integer vector")
        if node is None:
            node = np.arange(n_nodes)
        node = np.asarray(node)
        if node.shape != (n_nodes,):
            raise ValueError("node must provide one identifier per sensor column")
        innovation_scale = self._node_scales(node)
        stationary_factor = 1/np.sqrt(1-self.phi_**2)
        mean = np.zeros_like(residual)
        scale = np.empty_like(residual)
        last_value = np.zeros(n_nodes)
        last_time = np.zeros(n_nodes, dtype=np.int64)
        seen = np.zeros(n_nodes, dtype=bool)
        for row, clock in enumerate(timestep.astype(np.int64)):
            if np.any(seen):
                gap = clock-last_time[seen]
                if np.any(gap <= 0):
                    raise ValueError("Observation gaps must be positive")
                rho = self.phi_**gap
                mean[row, seen] = rho*last_value[seen]
                scale[row, seen] = innovation_scale[seen]*np.sqrt(
                    (1-rho**2)/(1-self.phi_**2))
            scale[row, ~seen] = innovation_scale[~seen]*stationary_factor
            update = observed[row]
            last_value[update] = residual[row, update]
            last_time[update] = clock
            seen[update] = True
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise FloatingPointError("Nonfinite AR sequence prediction")
        return mean, scale

    def metadata(self):
        if not getattr(self, "fitted_", False):
            raise RuntimeError("LatentARNormalError must be fitted before metadata")
        return {"model": type(self).__name__, "config": asdict(self.config),
            "phi": self.phi_, "pooled_innovation_scale_m": self.pooled_innovation_scale_,
            "fit_pair_count": self.fit_pair_count_,
            "fit_scenarios": self.fit_scenarios_.tolist(),
            "fit_contract": "cross-fitted family-zero consecutive residual pairs",
            "scenario_usage": "equal-risk fit weighting only",
            "node_usage": "shrunken innovation scale only",
            "prediction_contract": "last observed residual only; update after emission"}
