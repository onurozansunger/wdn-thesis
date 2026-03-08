"""Corruption pipeline: apply missing data, noise, and attacks to clean snapshots.

Takes ground-truth pressure/flow values and produces corrupted observations
with masks indicating which values are observed.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass

from wdn.config import CorruptionConfig


@dataclass
class CorruptedSnapshot:
    """Corrupted observations for a single snapshot.

    These get combined with the clean Snapshot to form the model input.
    """

    # Observed values (with noise; missing values set to 0)
    pressure_obs: torch.Tensor         # (N,)
    flow_obs: torch.Tensor             # (NE,)

    # Binary masks: 1 = observed, 0 = missing
    pressure_mask: torch.Tensor        # (N,)
    flow_mask: torch.Tensor            # (NE,)

    # Anomaly labels: 1 = attacked, 0 = clean (for Phase 5+)
    pressure_anomaly: torch.Tensor     # (N,)
    flow_anomaly: torch.Tensor         # (NE,)


def corrupt_snapshot(
    pressure_true: torch.Tensor,
    flow_true: torch.Tensor,
    cfg: CorruptionConfig,
    rng: np.random.Generator,
) -> CorruptedSnapshot:
    """Apply corruption to a single snapshot's ground truth values.

    Pipeline:
        1. Generate missing-data masks (Bernoulli)
        2. Add Gaussian noise to observed values
        3. (Optional) Apply adversarial attacks

    Args:
        pressure_true: (N,) ground truth pressures.
        flow_true: (NE,) ground truth flows.
        cfg: Corruption parameters.
        rng: NumPy random generator for reproducibility.

    Returns:
        CorruptedSnapshot with observations, masks, and anomaly labels.
    """
    N = pressure_true.shape[0]
    NE = flow_true.shape[0]

    # ------------------------------------------------------------------
    # Step 1: Missing data masks
    # mask=1 means observed, mask=0 means missing
    # ------------------------------------------------------------------
    p_mask = torch.tensor(
        rng.random(N) >= cfg.missing_rate_pressure,
        dtype=torch.float32,
    )
    q_mask = torch.tensor(
        rng.random(NE) >= cfg.missing_rate_flow,
        dtype=torch.float32,
    )

    # ------------------------------------------------------------------
    # Step 2: Add Gaussian noise to observed values
    # ------------------------------------------------------------------
    p_obs = pressure_true.clone()
    q_obs = flow_true.clone()

    if cfg.noise_sigma_pressure > 0:
        p_noise = torch.tensor(
            rng.normal(0, cfg.noise_sigma_pressure, size=N),
            dtype=torch.float32,
        )
        p_obs = p_obs + p_noise * p_mask  # only add noise to observed values

    if cfg.noise_sigma_flow > 0:
        q_noise = torch.tensor(
            rng.normal(0, cfg.noise_sigma_flow, size=NE),
            dtype=torch.float32,
        )
        q_obs = q_obs + q_noise * q_mask

    # ------------------------------------------------------------------
    # Step 3: Anomaly labels (clean by default, Phase 5 adds attacks)
    # ------------------------------------------------------------------
    p_anomaly = torch.zeros(N, dtype=torch.float32)
    q_anomaly = torch.zeros(NE, dtype=torch.float32)

    if cfg.attack_enabled:
        p_obs, q_obs, p_anomaly, q_anomaly = _apply_attacks(
            p_obs, q_obs, p_mask, q_mask, cfg, rng,
        )

    # ------------------------------------------------------------------
    # Step 4: Zero out missing values
    # ------------------------------------------------------------------
    p_obs = p_obs * p_mask
    q_obs = q_obs * q_mask

    return CorruptedSnapshot(
        pressure_obs=p_obs,
        flow_obs=q_obs,
        pressure_mask=p_mask,
        flow_mask=q_mask,
        pressure_anomaly=p_anomaly,
        flow_anomaly=q_anomaly,
    )


def _apply_attacks(
    p_obs: torch.Tensor,
    q_obs: torch.Tensor,
    p_mask: torch.Tensor,
    q_mask: torch.Tensor,
    cfg: CorruptionConfig,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply adversarial data falsification to observed values.

    Only attacks *observed* sensors (can't attack what's not there).
    Falsification: obs_new = obs * scale + bias

    Returns updated (p_obs, q_obs, p_anomaly, q_anomaly).
    """
    N = p_obs.shape[0]
    NE = q_obs.shape[0]

    p_anomaly = torch.zeros(N, dtype=torch.float32)
    q_anomaly = torch.zeros(NE, dtype=torch.float32)

    # Select which observed sensors to attack
    observed_p = torch.where(p_mask > 0)[0].numpy()
    observed_q = torch.where(q_mask > 0)[0].numpy()

    n_attack_p = max(1, int(len(observed_p) * cfg.attack_fraction))
    n_attack_q = max(1, int(len(observed_q) * cfg.attack_fraction))

    if len(observed_p) > 0:
        attacked_p = rng.choice(observed_p, size=min(n_attack_p, len(observed_p)), replace=False)
        for idx in attacked_p:
            p_obs[idx] = p_obs[idx] * cfg.attack_scale + cfg.attack_bias
            p_anomaly[idx] = 1.0

    if len(observed_q) > 0:
        attacked_q = rng.choice(observed_q, size=min(n_attack_q, len(observed_q)), replace=False)
        for idx in attacked_q:
            q_obs[idx] = q_obs[idx] * cfg.attack_scale + cfg.attack_bias
            q_anomaly[idx] = 1.0

    return p_obs, q_obs, p_anomaly, q_anomaly


def corrupt_all_snapshots(
    snapshots: list,
    cfg: CorruptionConfig,
    seed: int = 42,
) -> list[CorruptedSnapshot]:
    """Apply corruption to all snapshots.

    Args:
        snapshots: List of Snapshot objects with .pressure_true and .flow_true.
        cfg: Corruption configuration.
        seed: Random seed.

    Returns:
        List of CorruptedSnapshot objects (same order as input).
    """
    rng = np.random.default_rng(seed)
    corrupted = []

    for snap in snapshots:
        c = corrupt_snapshot(snap.pressure_true, snap.flow_true, cfg, rng)
        corrupted.append(c)

    return corrupted
