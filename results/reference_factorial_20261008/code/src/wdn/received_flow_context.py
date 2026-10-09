"""Received-flow context for the existing General mixture; no detector here.

All fitting inputs must be normal observations from the current TRAIN scope.
Missing values contribute zero weight, never a measured zero. Inference is
instantaneous or backward-looking, and receives no labels or hydraulic truth.
"""
from __future__ import annotations
import numpy as np

FLOW_RANK = 16
FLOW_LAGS = (1, 3, 6, 12, 24)


def quantize(x):
    return np.rint(np.asarray(x) / .1) * .1


class ReceivedFlowReference:
    def fit(self, pressure, pressure_mask, flow, flow_mask):
        p, pm = np.asarray(pressure, float), np.asarray(pressure_mask, bool)
        q, qm = np.asarray(flow, float), np.asarray(flow_mask, bool)
        if p.shape != pm.shape or q.shape != qm.shape or len(p) != len(q):
            raise ValueError('Matching normal received arrays required')
        if not np.isfinite(p[pm]).all() or not np.isfinite(q[qm]).all():
            raise ValueError('Nonfinite received observation')
        if min(pm.sum(0).min(), qm.sum(0).min()) < 2:
            raise ValueError('Insufficient normal TRAIN support')
        self.mean = np.where(qm, q, 0).sum(0) / qm.sum(0)
        centered = np.where(qm, q - self.mean, 0.)
        scale = np.sqrt((centered ** 2).sum(0) / qm.sum(0))
        self.scale = np.maximum(scale, max(float(np.median(scale)) * .01, 1e-8))
        z = centered / self.scale
        filled = z.copy()
        rank = min(FLOW_RANK, min(q.shape) - 1)
        for _ in range(12):
            _, _, vt = np.linalg.svd(filled, full_matrices=False)
            self.basis = vt[:rank].T
            filled = np.where(qm, z, (filled @ self.basis) @ self.basis.T)
        latent, _, _ = self.encode(q, qm)
        self.latent_scale = np.maximum(np.std(latent, axis=0), .01)
        x = np.column_stack((np.ones(len(q)), latent / self.latent_scale))
        self.coef = np.empty((x.shape[1], p.shape[1]))
        for j in range(p.shape[1]):
            seen = pm[:, j]
            reg = np.eye(x.shape[1]) * 10.
            reg[0, 0] = 0.
            self.coef[:, j] = np.linalg.solve(x[seen].T @ x[seen] + reg, x[seen].T @ p[seen, j])
        error = np.where(pm, np.abs(p - x @ self.coef), np.nan)
        scale = np.nanmedian(error, axis=0) / .67448975
        self.pressure_scale = np.maximum(scale, max(float(np.median(scale)) * .1, .01))
        return self

    def encode(self, flow, mask):
        q, seen = np.asarray(flow, float), np.asarray(mask, bool)
        if q.shape != seen.shape or q.shape[1] != len(self.mean):
            raise ValueError('Flow schema mismatch')
        if not np.isfinite(q[seen]).all():
            raise ValueError('Nonfinite received flow')
        z = np.where(seen, q - self.mean, 0.) / self.scale
        weights = seen.astype(float)
        for step in range(4):
            gram = np.einsum('tn,nr,ns->trs', weights, self.basis, self.basis, optimize=True)
            gram += np.eye(self.basis.shape[1])[None] * .01
            latent = np.linalg.solve(gram, ((weights * z) @ self.basis)[..., None])[..., 0]
            error = np.abs(z - latent @ self.basis.T)
            if step < 3:
                weights = seen * np.minimum(1., 2.5 / np.maximum(error, 1e-12))
        coverage = seen.mean(1)
        mismatch = np.sum(np.where(seen, np.minimum(error, 10.), 0.), axis=1) / np.maximum(seen.sum(1), 1)
        return latent, coverage, mismatch

    def predict(self, flow, mask):
        latent, coverage, mismatch = self.encode(flow, mask)
        context = latent / self.latent_scale
        predicted = np.column_stack((np.ones(len(latent)), context)) @ self.coef
        unavailable = coverage == 0
        predicted[unavailable] = np.nan
        context[unavailable] = np.nan
        mismatch[unavailable] = np.nan
        return predicted, context, coverage, mismatch


def flow_context_features(reference, pressure, pressure_mask, flow, flow_mask,
                          times, query_times, nodes, pressure_scale, rounding_m=0.):
    p, pm = np.asarray(pressure, float), np.asarray(pressure_mask, bool)
    times, qt, nodes = np.asarray(times), np.asarray(query_times), np.asarray(nodes)
    if p.shape != pm.shape or len(times) != len(p) or np.any(np.diff(times) <= 0):
        raise ValueError('Unique increasing timestamps and matching pressure masks required')
    row = np.searchsorted(times, qt)
    if np.any(row >= len(times)) or not np.array_equal(times[row], qt) or not pm[row, nodes].all():
        raise ValueError('Query must be an actually received pressure at its exact timestamp')
    if rounding_m:
        p = np.rint(p / rounding_m) * rounding_m
    predicted, context, coverage, mismatch = reference.predict(flow, flow_mask)
    scale = np.asarray(pressure_scale, float)
    if scale.shape != qt.shape or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Positive TRAIN pressure reference scales required')
    fs = reference.pressure_scale[nodes]
    residual = p[row, nodes] - predicted[row, nodes]
    parts = [quantize(residual / scale), quantize(np.abs(residual) / scale),
             quantize(residual / fs), quantize(np.abs(residual) / fs),
             quantize(fs / scale), coverage[row], quantize(mismatch[row]), (coverage[row] > 0).astype(float)]
    names = ['residual', 'abs_residual', 'own_scaled_residual', 'own_scaled_abs_residual',
             'uncertainty_ratio', 'coverage', 'mismatch', 'available']
    for j in range(context.shape[1]):
        parts.append(quantize(context[row, j])); names.append(f'latent_{j}')
    for lag in FLOW_LAGS:
        pos = np.searchsorted(times, qt - lag)
        safe = np.minimum(pos, len(times) - 1)
        valid = (pos < len(times)) & (times[safe] == qt - lag) & pm[safe, nodes]
        valid &= (coverage[safe] > 0) & (coverage[row] > 0)
        pressure_delta = (p[row, nodes] - p[safe, nodes]) / scale
        flow_delta = (predicted[row, nodes] - predicted[safe, nodes]) / scale
        for value, name in ((flow_delta, 'predicted_change'), (pressure_delta - flow_delta, 'change_error'),
                            (np.abs(pressure_delta - flow_delta), 'abs_change_error')):
            parts.append(np.where(valid, quantize(value), np.nan)); names.append(f'{name}_{lag}')
        parts.append(valid.astype(float)); names.append(f'pair_available_{lag}')
    return np.column_stack(parts).astype(np.float32), ['history_flow_' + n for n in names]
