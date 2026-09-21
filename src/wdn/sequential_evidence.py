"""Causal drift/ramp and variance-state evidence, with explicit missing data."""
from __future__ import annotations

import numpy as np


def sequential_evidence(values, observed, predicted, scale, window=16):
    values = np.asarray(values, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    predicted, scale = np.asarray(predicted), np.asarray(scale)
    if values.ndim != 2 or observed.shape != values.shape or predicted.shape != values.shape:
        raise ValueError("Expected matching time-by-sensor arrays")
    if not np.isfinite(values[observed]).all() or not np.isfinite(predicted).all() or np.any(scale <= 0):
        raise ValueError("Finite observations/reference and positive scale required")
    if window < 2 or len(values) < window:
        raise ValueError("Insufficient history")
    residual = np.where(observed, (values-predicted)/scale, 0.)
    T, N = values.shape
    mean, var, seen = np.zeros(N), np.ones(N), np.zeros(N, dtype=int)
    shifts = np.array([.25, .5, 1.])
    ratios = np.array([2., 4., 9.])
    positive, negative = np.zeros((3, N)), np.zeros((3, N))
    noise_probability = np.full((3, N), .005)
    z_history = np.zeros_like(values)
    outputs = []
    names = ["seq_innovation", "seq_abs_innovation", "seq_normal_sigma", "seq_seen"]
    names += [f"drift_cusum_{sign}_{delta}" for sign in ("positive", "negative") for delta in shifts]
    names += [f"noise_state_{ratio}" for ratio in ratios]
    for length in (3, 5, 8, 12, 16):
        names += [f"drift_ramp_{length}", f"drift_mean_{length}", f"noise_energy_{length}",
                  f"seq_support_{length}", f"drift_consistency_{length}",
                  f"drift_ramp_strength_{length}", f"drift_mean_strength_{length}"]
    for t in range(T):
        valid = observed[t]
        sigma = np.sqrt(np.maximum(var, .25))
        innovation = np.where(valid & (seen > 0), (residual[t]-mean)/sigma, 0.)
        z = np.clip(innovation, -12., 12.)
        z_history[t] = z
        llr = shifts[:, None]*z[None]-.5*shifts[:, None]**2
        negative_llr = -shifts[:, None]*z[None]-.5*shifts[:, None]**2
        positive = np.maximum(0., .98*positive+np.where(valid[None], llr, 0.))
        negative = np.maximum(0., .98*negative+np.where(valid[None], negative_llr, 0.))
        # Fault duration/entry priors are fixed modelling assumptions. The
        # real event start/end/family is not an input, including at reset.
        prior = noise_probability*.90+(1-noise_probability)*.0005
        evidence = .5*((1-1/ratios[:, None])*z[None]**2-np.log(ratios[:, None]))
        log_odds = np.log(prior/(1-prior))+np.where(valid[None], evidence, 0.)
        noise_probability = 1/(1+np.exp(-np.clip(log_odds, -30, 30)))
        if t >= window-1:
            columns = [innovation, np.abs(innovation), sigma, np.minimum(seen, 32)/32]
            columns += list(positive)+list(negative)+list(noise_probability)
            for length in (3, 5, 8, 12, 16):
                start = max(0, t-length+1)
                w = observed[start:t+1]
                zs = z_history[start:t+1]
                count = w.sum(0)
                age = np.arange(1, t-start+2)[:, None]
                ramp = (zs*age).sum(0)/np.sqrt(np.maximum((w*age**2).sum(0), 1.))
                mean_score = zs.sum(0)/np.sqrt(np.maximum(count, 1.))
                energy = ((zs**2*w).sum(0)+2.)/(count+2.)
                consistency = np.abs(np.sign(zs).sum(0))/np.maximum(count, 1.)
                columns += [ramp, mean_score, energy, count/length, consistency, .5*ramp**2, .5*mean_score**2]
            outputs.append(np.column_stack(columns))
        # Slow, clipped normal-state adaptation cannot use clean-event labels.
        first = valid & (seen == 0)
        delta = residual[t]-mean
        mean = np.where(first, residual[t], np.where(valid, mean+.01*np.clip(delta, -2*sigma, 2*sigma), mean))
        var = np.where(valid & ~first, .99*var+.01*np.minimum(delta**2, 4*var), var)
        seen += valid
    return np.asarray(outputs, dtype=np.float32), names
