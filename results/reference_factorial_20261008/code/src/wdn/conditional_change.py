"""Causal marked change filter with a shared conditional normal model.

The null, drift, and noise branches all use the caller's frozen nuisance mean
and scale.  Drift adds only an onset-specific intercept and slope; noise adds
only zero-mean excess measurement variance.  No labels, attack boundaries, or
future observations enter the recursion.

Rows are assumed to be one hour apart, so slopes and posterior ages are
reported in metres/hour and hours respectively.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class ConditionalChangeConfig:
    """Frozen physical priors for the conditional change hypotheses."""

    entry_probability: float = 0.001
    survival_probability: float = 0.90
    intercept_scale_multiplier: float = 0.50
    slope_scale_multiplier: float = 0.35
    noise_extra_std_m: tuple[float, ...] = (0.20, 0.40, 0.80)


class ConditionalChangeFilter:
    """Online posterior over normal, drift, and excess-noise states."""

    FEATURE_NAMES = (
        "conditional_drift_probability",
        "conditional_noise_probability",
        "conditional_drift_logodds",
        "conditional_noise_logodds",
        "conditional_drift_slope_m_per_h",
        "conditional_noise_excess_std_m",
        "conditional_drift_age_h",
        "conditional_noise_age_h",
        "conditional_change_probability",
    )

    def __init__(self, config=ConditionalChangeConfig()):
        self.config = config
        if config != ConditionalChangeConfig():
            raise ValueError("Conditional-change priors are frozen for this experiment")
        numbers = (
            config.entry_probability,
            config.survival_probability,
            config.intercept_scale_multiplier,
            config.slope_scale_multiplier,
            *config.noise_extra_std_m,
        )
        if not all(np.isfinite(value) for value in numbers):
            raise ValueError("Conditional-change priors must be finite")
        if not 0 < config.entry_probability < 1:
            raise ValueError("entry_probability must lie strictly between zero and one")
        if not 0 < config.survival_probability < 1:
            raise ValueError("survival_probability must lie strictly between zero and one")
        if min(config.intercept_scale_multiplier, config.slope_scale_multiplier) <= 0:
            raise ValueError("Drift prior scales must be positive")
        if not config.noise_extra_std_m or min(config.noise_extra_std_m) <= 0:
            raise ValueError("At least one positive excess-noise scale is required")

    @staticmethod
    def _validate(values, observed, reference_prediction, nuisance_mean,
                  nuisance_scale, prior_scale, window):
        values = np.asarray(values, dtype=float)
        observed = np.asarray(observed, dtype=bool)
        prediction = np.asarray(reference_prediction, dtype=float)
        mean = np.asarray(nuisance_mean, dtype=float)
        scale = np.asarray(nuisance_scale, dtype=float)
        prior_scale = np.asarray(prior_scale, dtype=float)
        if (values.ndim != 2 or observed.shape != values.shape
                or prediction.shape != values.shape or mean.shape != values.shape
                or scale.shape != values.shape):
            raise ValueError("All time-by-sensor inputs must have matching shapes")
        if prior_scale.shape != (values.shape[1],):
            raise ValueError("prior_scale must provide one value per sensor")
        if not isinstance(window, (int, np.integer)) or window < 1 or len(values) < window:
            raise ValueError("window must be a positive integer no larger than the sequence")
        if (not np.isfinite(values[observed]).all()
                or not np.isfinite(prediction).all()
                or not np.isfinite(mean).all()
                or not np.isfinite(scale).all()
                or not np.isfinite(prior_scale).all()):
            raise ValueError("Observed values and all reference/nuisance inputs must be finite")
        if np.any(scale <= 0) or np.any(prior_scale <= 0):
            raise ValueError("nuisance_scale and prior_scale must be strictly positive")
        return values, observed, prediction, mean, scale, prior_scale

    @staticmethod
    def _drift_evidence(sum_w, sum_w_age, sum_w_age2, sum_w_y,
                        sum_w_age_y, intercept_std, slope_std):
        """Bayes factor and slope posterior for a heteroscedastic segment."""
        p00 = sum_w + 1/intercept_std[None]**2
        p01 = sum_w_age
        p11 = sum_w_age2 + 1/slope_std[None]**2
        determinant = p00*p11-p01**2
        if np.any(determinant <= 0) or not np.isfinite(determinant).all():
            raise FloatingPointError("Invalid drift posterior precision")
        quadratic = (p11*sum_w_y**2-2*p01*sum_w_y*sum_w_age_y
                     +p00*sum_w_age_y**2)/determinant
        prior_logdet = -2*np.log(intercept_std)[None]-2*np.log(slope_std)[None]
        evidence = 0.5*(prior_logdet-np.log(determinant)+quadratic)
        slope = (p00*sum_w_age_y-p01*sum_w_y)/determinant
        return evidence, slope

    def transform(self, values, observed, reference_prediction, nuisance_mean,
                  nuisance_scale, prior_scale, window=16):
        """Return causal conditional-change features for every observed prefix.

        Missing observations have likelihood one under every branch.  State
        transitions and onset ages still advance in clock time.
        """
        (values, observed, prediction, nuisance_mean,
         nuisance_scale, prior_scale) = self._validate(
            values, observed, reference_prediction, nuisance_mean,
            nuisance_scale, prior_scale, window)
        t_count, n_sensors = values.shape
        cfg = self.config
        residual = np.where(observed, values-prediction-nuisance_mean, 0.0)

        # One row per possible onset.  Only rows through the current hour are
        # active.  The five arrays are sufficient statistics for Bayesian
        # linear regression with a segment-specific intercept and slope.
        sum_w = np.zeros((t_count, n_sensors))
        sum_w_age = np.zeros_like(sum_w)
        sum_w_age2 = np.zeros_like(sum_w)
        sum_w_y = np.zeros_like(sum_w)
        sum_w_age_y = np.zeros_like(sum_w)
        old_drift_evidence = np.zeros_like(sum_w)

        extra_std = np.asarray(cfg.noise_extra_std_m, dtype=float)
        noise_evidence = np.zeros((t_count, n_sensors, len(extra_std)))
        old_noise_marginal = np.zeros_like(sum_w)

        drift_log_mass = np.full_like(sum_w, -np.inf)
        noise_log_mass = np.full_like(sum_w, -np.inf)
        normal_probability = np.ones(n_sensors)
        intercept_std = prior_scale*cfg.intercept_scale_multiplier
        slope_std = prior_scale*cfg.slope_scale_multiplier
        log_survival = np.log(cfg.survival_probability)
        log_entry_branch = np.log(cfg.entry_probability/2)
        outputs = []

        for t in range(t_count):
            end = t+1
            ages = (t-np.arange(end)+1.0)[:, None]
            valid = observed[t].astype(float)[None]
            variance = nuisance_scale[t]**2
            precision = valid/variance[None]
            y = residual[t][None]

            sum_w[:end] += precision
            sum_w_age[:end] += precision*ages
            sum_w_age2[:end] += precision*ages**2
            sum_w_y[:end] += precision*y
            sum_w_age_y[:end] += precision*ages*y
            drift_evidence, drift_slope = self._drift_evidence(
                sum_w[:end], sum_w_age[:end], sum_w_age2[:end],
                sum_w_y[:end], sum_w_age_y[:end], intercept_std, slope_std)

            total_variance = variance[None, :, None]+extra_std[None, None]**2
            noise_increment = 0.5*(
                np.log(variance)[None, :, None]-np.log(total_variance)
                +y[:, :, None]**2*(1/variance[None, :, None]-1/total_variance)
            )
            noise_evidence[:end] += valid[:, :, None]*noise_increment
            noise_marginal = logsumexp(noise_evidence[:end], axis=2)-np.log(len(extra_std))

            # Existing faults survive; a new drift/noise onset can start only
            # from the preceding normal state.  Faults that end return to the
            # null before the current emission.
            fault_probability = np.clip(1-normal_probability, 0.0, 1.0)
            normal_prior = (normal_probability*(1-cfg.entry_probability)
                            +fault_probability*(1-cfg.survival_probability))
            drift_log_mass[:t] += log_survival
            noise_log_mass[:t] += log_survival
            start = np.log(np.maximum(normal_probability, 1e-300))+log_entry_branch
            drift_log_mass[t] = start
            noise_log_mass[t] = start

            # The null likelihood is common and has been divided out.  These
            # increments are therefore Bayes factors against the same frozen
            # nuisance mean/scale, not separately normalised residual scores.
            drift_log_mass[:end] += drift_evidence-old_drift_evidence[:end]
            noise_log_mass[:end] += noise_marginal-old_noise_marginal[:end]
            old_drift_evidence[:end] = drift_evidence
            old_noise_marginal[:end] = noise_marginal

            normal_log_mass = np.log(np.maximum(normal_prior, 1e-300))
            normaliser = logsumexp(np.vstack([
                normal_log_mass[None], drift_log_mass[:end], noise_log_mass[:end]
            ]), axis=0)
            normal_probability = np.exp(normal_log_mass-normaliser)
            drift_log_mass[:end] -= normaliser
            noise_log_mass[:end] -= normaliser
            pd = np.exp(drift_log_mass[:end])
            pn = np.exp(noise_log_mass[:end])
            drift_probability = np.clip(pd.sum(axis=0), 0.0, 1.0)
            noise_probability = np.clip(pn.sum(axis=0), 0.0, 1.0)

            if t >= window-1:
                def conditional_average(mass, quantity, total):
                    return (mass*quantity).sum(axis=0)/np.maximum(total, 1e-300)

                safe_d = np.clip(drift_probability, 1e-12, 1-1e-12)
                safe_n = np.clip(noise_probability, 1e-12, 1-1e-12)
                noise_grid_probability = np.exp(
                    noise_evidence[:end]
                    -logsumexp(noise_evidence[:end], axis=2, keepdims=True))
                onset_noise_std = (noise_grid_probability*extra_std[None, None]).sum(axis=2)
                change_probability = np.clip(drift_probability+noise_probability, 0.0, 1.0)
                outputs.append(np.column_stack([
                    drift_probability,
                    noise_probability,
                    np.log(safe_d/(1-safe_d)),
                    np.log(safe_n/(1-safe_n)),
                    conditional_average(pd, drift_slope, drift_probability),
                    conditional_average(pn, onset_noise_std, noise_probability),
                    conditional_average(pd, ages, drift_probability),
                    conditional_average(pn, ages, noise_probability),
                    change_probability,
                ]))

        result = np.asarray(outputs, dtype=np.float32)
        if (result.shape != (t_count-window+1, n_sensors, len(self.FEATURE_NAMES))
                or not np.isfinite(result).all()):
            raise FloatingPointError("Nonfinite conditional-change output")
        return result, list(self.FEATURE_NAMES)
