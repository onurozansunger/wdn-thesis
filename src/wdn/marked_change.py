"""Causal marked run-length filter with unknown slope and excess variance.

Finite-dimensional sufficient statistics integrate the segment parameters.
Missing emissions have likelihood one; duration still advances in clock time.
These model posteriors are not claimed to be empirically calibrated scores.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammainc, gammaln, logsumexp


@dataclass(frozen=True)
class ChangeConfig:
    entry_probability: float = .001
    survival_probability: float = .90
    offset_sigma: float = .5
    slope_sigma: float = .35
    variance_shape: float = 3.
    variance_scale: float = 8.
    mean_precision: float = 4.


def _log_lower_gamma(a, b):
    return np.log(np.maximum(gammainc(a, b), np.finfo(float).tiny))


def variance_log_evidence(count, total, square, config):
    """Normal/inverse-gamma marginal, truncated to variance >= one."""
    k0, a0, b0 = config.mean_precision, config.variance_shape, config.variance_scale
    k, a = k0+count, a0+.5*count
    b = b0+.5*np.maximum(square-total**2/k, 0.)
    evidence = (gammaln(a)-gammaln(a0)+a0*np.log(b0)-a*np.log(b)
                -.5*count*np.log(2*np.pi)+.5*np.log(k0/k)
                +_log_lower_gamma(a, b)-_log_lower_gamma(a0, b0))
    mean_variance = b/(a-1)*np.exp(_log_lower_gamma(a-1, b)-_log_lower_gamma(a, b))
    return evidence, mean_variance


def slope_log_evidence(count, total, square, age_sum, age_square, age_value, config):
    """Gaussian integrated likelihood for uncertain intercept and signed slope."""
    p0, p1 = 1/config.offset_sigma**2, 1/config.slope_sigma**2
    aa, cc = count+p0, age_square+p1
    determinant = aa*cc-age_sum**2
    quadratic = (cc*total**2-2*age_sum*total*age_value+aa*age_value**2)/determinant
    evidence = -.5*(count*np.log(2*np.pi)+square+np.log(determinant/(p0*p1))-quadratic)
    slope = (aa*age_value-age_sum*total)/determinant
    return evidence, slope, aa/determinant


class MarkedChangeFilter:
    def __init__(self, config=ChangeConfig()):
        self.config = config
        if (not 0 < config.entry_probability < 1 or not 0 < config.survival_probability < 1
                or min(config.offset_sigma, config.slope_sigma, config.variance_scale,
                       config.mean_precision) <= 0 or config.variance_shape <= 1):
            raise ValueError("Invalid changepoint prior")

    def transform(self, values, observed, predicted, scale, window=16):
        values, predicted = np.asarray(values, dtype=float), np.asarray(predicted, dtype=float)
        mask, scale = np.asarray(observed, dtype=bool), np.asarray(scale, dtype=float)
        if (values.ndim != 2 or values.shape != predicted.shape or values.shape != mask.shape
                or scale.shape != (values.shape[1],) or window < 1 or len(values) < window):
            raise ValueError("Matching time-sensor arrays and sufficient history required")
        if (not np.isfinite(values[mask]).all() or not np.isfinite(predicted).all()
                or not np.isfinite(scale).all() or np.any(scale <= 0)):
            raise ValueError("Finite observations and positive normal scale required")
        z = np.where(mask, (values-predicted)/scale, 0.)
        t_count, n_sensors = z.shape
        cfg = self.config
        stats = [np.zeros_like(z) for _ in range(6)]
        count, total, square, age_sum, age_square, age_value = stats
        prior_d = np.full_like(z, -np.inf)
        prior_n = np.full_like(z, -np.inf)
        old_d, old_n = np.zeros_like(z), np.zeros_like(z)
        normal_probability = np.ones(n_sensors)
        outputs = []
        names = ["drift_run_probability", "noise_run_probability", "drift_run_logodds",
                 "noise_run_logodds", "drift_run_slope", "drift_run_abs_slope",
                 "noise_run_variance", "seq_run_drift_age", "seq_run_noise_age",
                 "seq_run_change_probability", "drift_run_slope_uncertainty"]
        for t in range(t_count):
            end = t+1
            ages = (t-np.arange(end)+1)[:, None]
            valid = mask[t].astype(float)[None]
            x = z[t][None]
            count[:end] += valid
            total[:end] += valid*x
            square[:end] += valid*x*x
            age_sum[:end] += valid*ages
            age_square[:end] += valid*ages*ages
            age_value[:end] += valid*ages*x
            log_d, slope, slope_variance = slope_log_evidence(
                *(s[:end] for s in stats), cfg)
            log_n, variance = variance_log_evidence(count[:end], total[:end], square[:end], cfg)
            normal_prior = (normal_probability*(1-cfg.entry_probability)
                            +(1-normal_probability)*(1-cfg.survival_probability))
            prior_d[:t] += np.log(cfg.survival_probability)
            prior_n[:t] += np.log(cfg.survival_probability)
            start = np.log(np.maximum(normal_probability, 1e-300))+np.log(cfg.entry_probability/2)
            prior_d[t] = start
            prior_n[t] = start
            normal_log = np.log(np.maximum(normal_prior, 1e-300))
            normal_log += np.where(mask[t], -.5*(np.log(2*np.pi)+z[t]**2), 0.)
            # Evidence differences are exactly zero for missing emissions.
            prior_d[:end] += log_d-old_d[:end]
            prior_n[:end] += log_n-old_n[:end]
            old_d[:end], old_n[:end] = log_d, log_n
            norm = logsumexp(np.vstack([normal_log[None], prior_d[:end], prior_n[:end]]), axis=0)
            normal_probability = np.exp(normal_log-norm)
            prior_d[:end] -= norm
            prior_n[:end] -= norm
            pd, pn = np.exp(prior_d[:end]), np.exp(prior_n[:end])
            drift, noise = pd.sum(axis=0), pn.sum(axis=0)
            if t >= window-1:
                def conditional_mean(probability, value, mass):
                    return (probability*value).sum(axis=0)/np.maximum(mass, 1e-300)
                safe_d, safe_n = np.clip(drift, 1e-12, 1-1e-12), np.clip(noise, 1e-12, 1-1e-12)
                outputs.append(np.column_stack([drift, noise,
                    np.log(safe_d/(1-safe_d)), np.log(safe_n/(1-safe_n)),
                    conditional_mean(pd, slope, drift), conditional_mean(pd, np.abs(slope), drift),
                    conditional_mean(pn, variance, noise), conditional_mean(pd, ages, drift),
                    conditional_mean(pn, ages, noise), drift+noise,
                    np.sqrt(conditional_mean(pd, slope_variance, drift))]))
        result = np.asarray(outputs, dtype=np.float32)
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite marked change evidence")
        return result, names
