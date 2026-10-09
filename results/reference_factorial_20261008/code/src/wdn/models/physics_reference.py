"""Target-group-blind head estimation from measured flow and static pipes.

No demands, hydraulic truth or event metadata enter this estimator. Unanchored
or unstable estimates fall back to the supplied TRAIN-fitted pressure model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class PhysicsConfig:
    pressure_sigma: float = .1
    flow_sigma: float = .0001
    steps: int = 4
    cutoff: float = 2.5
    minimum_anchors: int = 2


class HazenWilliamsReference:
    def __init__(self, graph, baseline, config=PhysicsConfig()):
        self.baseline, self.config = baseline, config
        self.group_ = np.asarray(baseline.reference.group_).copy()
        self.reservoir = np.asarray(graph.node_types) == 1
        self.elevation = np.asarray(graph.node_elevations, dtype=float).copy()
        self.edges = np.asarray(graph.edge_index, dtype=int).copy()
        length, diameter, roughness = (np.asarray(x, dtype=float) for x in
            (graph.edge_lengths, graph.edge_diameters, graph.edge_roughness))
        if (np.any(np.asarray(graph.edge_types) != 0) or
                np.any(np.asarray(graph.node_types) > 1)):
            raise ValueError("This reference supports pipes, junctions and fixed reservoirs only")
        if (self.edges.shape != (2, len(length)) or len(self.group_) != len(self.elevation)
                or np.any(length <= 0) or np.any(diameter <= 0) or np.any(roughness <= 0)):
            raise ValueError("Invalid static graph or non-SI pipe geometry")
        if (min(config.pressure_sigma, config.flow_sigma, config.cutoff) <= 0
                or config.steps < 1 or config.minimum_anchors < 1):
            raise ValueError("Invalid physical uncertainty settings")
        self.resistance = 10.666829500036352*length*roughness**-1.852*diameter**-4.871
        self.noise_scale_ = baseline.noise_scale_.copy()

    def predict_details(self, values, observed, flow, flow_observed):
        p, q = np.asarray(values, dtype=float), np.asarray(flow, dtype=float)
        pm, fm = np.asarray(observed, dtype=bool), np.asarray(flow_observed, dtype=bool)
        n = len(self.elevation)
        if p.ndim != 2 or p.shape != pm.shape or p.shape[1] != n:
            raise ValueError("Matching pressure arrays on the graph required")
        if q.shape != fm.shape or q.shape != (len(p), self.edges.shape[1]):
            raise ValueError("Matching original directed-edge flow arrays required")
        if not np.isfinite(p[pm]).all() or not np.isfinite(q[fm]).all():
            raise ValueError("Observed measurements must be finite")
        fallback, support, spread = self.baseline.predict_details(p, pm)
        prediction = fallback.copy()
        weight = np.zeros_like(p)
        anchors_count = np.zeros_like(p)
        sigma = np.broadcast_to(self.baseline.noise_scale_, p.shape).copy()
        instability = np.zeros_like(p)
        raw = fallback.copy()
        u, v = self.edges
        cfg = self.config
        base_variance = np.maximum(self.baseline.noise_scale_**2-cfg.pressure_sigma**2, .0025)
        for t in range(len(p)):
            present = np.flatnonzero(fm[t])
            rows = np.r_[u[present], v[present]]
            cols = np.r_[v[present], u[present]]
            adjacency = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
            _, components = connected_components(adjacency, directed=False)
            safe_flow = np.where(fm[t], q[t], 0.)
            loss = self.resistance*np.sign(safe_flow)*np.abs(safe_flow)**1.852
            # The second term keeps uncertainty nonzero around zero flow.
            flow_sigma = np.hypot(1.852*self.resistance*np.abs(safe_flow)**.852*cfg.flow_sigma,
                                  self.resistance*cfg.flow_sigma**1.852)
            flow_sigma = np.maximum(flow_sigma, .002)
            for group in np.unique(self.group_):
                target_group = (self.group_ == group) & ~self.reservoir
                # The target group's values AND masks cannot affect weights.
                anchors = (pm[t] & (self.group_ != group)) | self.reservoir
                for component in np.unique(components[target_group]):
                    nodes = np.flatnonzero(components == component)
                    targets = nodes[target_group[nodes]]
                    measured = nodes[anchors[nodes]]
                    anchors_count[t, targets] = len(measured)
                    if len(measured) < cfg.minimum_anchors:
                        continue
                    edges = present[components[u[present]] == component]
                    if not len(edges):
                        continue
                    local = np.full(n, -1, dtype=int)
                    local[nodes] = np.arange(len(nodes))
                    a = np.zeros((len(edges)+len(measured), len(nodes)))
                    a[np.arange(len(edges)), local[u[edges]]] = 1.
                    a[np.arange(len(edges)), local[v[edges]]] = -1.
                    a[len(edges)+np.arange(len(measured)), local[measured]] = 1.
                    anchor_heads = self.elevation[measured].copy()
                    junction = ~self.reservoir[measured]
                    anchor_heads[junction] += p[t, measured[junction]]
                    b = np.r_[loss[edges], anchor_heads]
                    error_sigma = np.r_[flow_sigma[edges],
                        np.where(self.reservoir[measured], 1e-4, cfg.pressure_sigma)]
                    trusted = np.r_[np.zeros(len(edges), dtype=bool), self.reservoir[measured]]
                    precision = 1/error_sigma**2
                    weights = precision.copy()
                    for step in range(cfg.steps):
                        gram = a.T@(weights[:, None]*a)
                        inverse = np.linalg.inv(gram)
                        estimate = inverse@(a.T@(weights*b))
                        residual = a@estimate-b
                        influence = inverse@a.T
                        leverage = weights*np.einsum("ij,ji->i", a, influence)
                        if step+1 < cfg.steps:
                            student = np.abs(residual)/error_sigma/np.sqrt(np.maximum(1-leverage, .05))
                            robust = np.minimum(1., cfg.cutoff/np.maximum(student, 1e-12))
                            weights = precision*np.where(trusted, 1., np.maximum(robust, .01))
                    target_local = local[targets]
                    target_prediction = estimate[target_local]-self.elevation[targets]
                    # Linearised leave-one-measurement-out sensitivity. Rows
                    # with leverage one identify a bridge/unidentifiable LOO;
                    # those targets receive an explicit uncertainty penalty.
                    influence = influence[target_local]
                    critical = (~trusted) & (1-leverage < 1e-5)
                    fragile = (np.abs(influence[:, critical]) > 1e-6).any(axis=1)
                    delta = np.abs(influence*(weights*residual/np.maximum(1-leverage, 1e-5))[None])
                    delta[:, trusted] = 0.
                    jackknife = np.max(delta, axis=1)
                    variance = np.maximum(np.diag(inverse)[target_local], 0.)+jackknife**2
                    variance += fragile*base_variance[targets]
                    unstable = jackknife > np.maximum(.1, 2*self.baseline.noise_scale_[targets])
                    disagreement = target_prediction-fallback[t, targets]
                    compatible = np.abs(disagreement) <= 3*np.sqrt(variance+base_variance[targets])
                    eligible = (variance < base_variance[targets]) & ~unstable & compatible
                    blend = np.where(eligible, base_variance[targets]/(base_variance[targets]+variance), 0.)
                    raw[t, targets] = target_prediction
                    weight[t, targets] = blend
                    prediction[t, targets] += blend*disagreement
                    sigma[t, targets] = np.sqrt(cfg.pressure_sigma**2+variance)
                    instability[t, targets] = jackknife
        diagnostics = {"physical_weight": weight, "anchor_count": anchors_count,
                       "physical_sigma": sigma, "jackknife": instability,
                       "raw_prediction": raw, "fallback_prediction": fallback}
        return prediction, support, spread, diagnostics

    def calibrate_scale(self, values, observed, flow, flow_observed):
        prediction, _, _, _ = self.predict_details(values, observed, flow, flow_observed)
        error = np.where(observed, np.abs(values-prediction), np.nan)
        scale = np.nanmedian(error, axis=0)/.67448975
        if not np.isfinite(scale).all():
            raise ValueError("Every sensor needs normal TRAIN observations")
        self.noise_scale_ = np.maximum(scale, max(float(np.median(scale))*.1, 1e-6))
        return self
