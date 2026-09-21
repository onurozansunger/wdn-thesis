"""Causal residual-context features for small drift and burst noise."""
from __future__ import annotations

import numpy as np


def dynamic_residual_features(values, observed, predicted, scale, window=16):
    values, observed = np.asarray(values), np.asarray(observed, dtype=bool)
    residual = np.where(observed, (values-predicted)/scale, 0.)
    T, N = residual.shape
    baseline, variance = np.zeros(N), np.ones(N)
    seen = np.zeros(N, dtype=int)
    positive, negative = np.zeros(N), np.zeros(N)
    online = []
    # Every state update uses current/past observations only and resets at a
    # scenario boundary. No attack labels identify a supposedly clean past.
    for t in range(T):
        valid = observed[t]
        sigma = np.sqrt(np.maximum(variance, .01))
        innovation = np.where(valid & (seen > 0), (residual[t]-baseline)/sigma, 0.)
        positive = np.where(valid, np.maximum(0., .9*positive+np.clip(innovation, -20., 20.)-.5), positive)
        negative = np.where(valid, np.maximum(0., .9*negative-np.clip(innovation, -20., 20.)-.5), negative)
        online.append(np.column_stack([innovation, np.abs(innovation), positive, negative,
                                       baseline.copy(), sigma, np.minimum(seen, 32)/32]))
        first = valid & (seen == 0)
        delta = residual[t]-baseline
        baseline = np.where(first, residual[t], np.where(valid, baseline+.02*np.clip(delta, -2*sigma, 2*sigma), baseline))
        variance = np.where(valid & ~first, .98*variance+.02*np.minimum(delta**2, 9*variance), variance)
        seen += valid
    online = np.asarray(online)
    names = [f"dynamic_{n}" for n in ("innovation", "abs_innovation", "cusum_positive",
             "cusum_negative", "baseline", "sigma", "past_support")]
    for length in (8, 16, 32, 48):
        names.extend([f"dynamic_{n}_{length}" for n in ("forecast_error", "forecast_support", "past_std",
                                                       "level_shift", "innovation_rms", "innovation_mean")])
    rows = []
    for t in range(window-1, T):
        cols = list(online[t].T)
        for length in (8, 16, 32, 48):
            start = max(0, t-length)
            r, w = residual[start:t], observed[start:t].astype(float)
            count = w.sum(0)
            mean = r.sum(0)/np.maximum(count, 1)
            times = np.arange(start, t)[:, None]
            tm = (times*w).sum(0)/np.maximum(count, 1)
            centered = times-tm
            slope = (centered*r).sum(0)/np.maximum((centered**2*w).sum(0), 1.)
            forecast = mean+slope*(t-tm)
            error = np.where(observed[t] & (count >= 3), residual[t]-forecast, 0.)
            std = np.sqrt(((r-mean)**2*w).sum(0)/np.maximum(count, 1))
            # Recent four samples versus the earlier context. Both means
            # ignore missing placeholders, and the baseline excludes now.
            recent_start = max(0, t-3)
            recent_w = observed[recent_start:t+1]
            recent_count = recent_w.sum(0)
            recent_mean = residual[recent_start:t+1].sum(0)/np.maximum(recent_count, 1)
            old_end = max(start, t-3)
            old_count = observed[start:old_end].sum(0)
            old_mean = residual[start:old_end].sum(0)/np.maximum(old_count, 1)
            shift = np.where((recent_count >= 2) & (old_count >= 3), recent_mean-old_mean, 0.)
            iw = observed[start:t+1]
            innovation = online[start:t+1, :, 0]
            inum = iw.sum(0).clip(min=1)
            cols.extend([error, count/max(1, t-start), std, shift,
                         np.sqrt((innovation**2*iw).sum(0)/inum), (innovation*iw).sum(0)/inum])
        rows.append(np.column_stack(cols))
    return np.asarray(rows, dtype=np.float32), names
