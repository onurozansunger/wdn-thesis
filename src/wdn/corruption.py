"""Corruption pipeline: apply missing data, noise, and attacks to clean snapshots.

Takes ground-truth pressure/flow values and produces corrupted observations
with masks indicating which values are observed.

Attack types (from WDN cyber-security literature):
    1. Random falsification: scale + bias on random sensors
    2. Replay attack: replace current reading with a past value
    3. Stealthy bias injection: small gradual drift that's hard to detect
    4. Targeted attack: attack sensors with highest impact on the network
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

    # Anomaly labels: 1 = attacked, 0 = clean
    pressure_anomaly: torch.Tensor     # (N,)
    flow_anomaly: torch.Tensor         # (NE,)


def corrupt_snapshot(
    pressure_true: torch.Tensor,
    flow_true: torch.Tensor,
    cfg: CorruptionConfig,
    rng: np.random.Generator,
    replay_buffer: dict | None = None,
    snapshot_idx: int = 0,
) -> CorruptedSnapshot:
    """Apply corruption to a single snapshot's ground truth values.

    Pipeline:
        1. Generate missing-data masks (Bernoulli)
        2. Add Gaussian noise to observed values
        3. (Optional) Apply adversarial attacks
        4. Zero out missing values

    Args:
        pressure_true: (N,) ground truth pressures.
        flow_true: (NE,) ground truth flows.
        cfg: Corruption parameters.
        rng: NumPy random generator for reproducibility.
        replay_buffer: Dict with past observations for replay attacks.
        snapshot_idx: Current snapshot index (for stealthy drift).

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
    # Step 3: Anomaly labels (clean by default)
    # ------------------------------------------------------------------
    p_anomaly = torch.zeros(N, dtype=torch.float32)
    q_anomaly = torch.zeros(NE, dtype=torch.float32)

    if cfg.attack_enabled:
        p_obs, q_obs, p_anomaly, q_anomaly = _apply_attacks(
            p_obs, q_obs, p_mask, q_mask,
            pressure_true, flow_true,
            cfg, rng,
            replay_buffer=replay_buffer,
            snapshot_idx=snapshot_idx,
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


# -----------------------------------------------------------------------
# Attack implementations
# -----------------------------------------------------------------------

def _select_targets(
    mask: torch.Tensor,
    fraction: float,
    rng: np.random.Generator,
    high_impact_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Select which observed sensors to attack.

    Args:
        mask: (K,) binary observation mask.
        fraction: Fraction of observed sensors to attack.
        rng: Random generator.
        high_impact_indices: If provided, preferentially attack these indices.

    Returns:
        Array of indices to attack.
    """
    observed = torch.where(mask > 0)[0].numpy()
    if len(observed) == 0:
        return np.array([], dtype=int)

    n_attack = max(1, int(len(observed) * fraction))
    n_attack = min(n_attack, len(observed))

    if high_impact_indices is not None:
        # Prefer high-impact sensors (targeted attack)
        candidates = np.intersect1d(observed, high_impact_indices)
        if len(candidates) >= n_attack:
            return rng.choice(candidates, size=n_attack, replace=False)
        # Fill remaining from other observed sensors
        remaining = np.setdiff1d(observed, candidates)
        n_extra = n_attack - len(candidates)
        if len(remaining) > 0 and n_extra > 0:
            extra = rng.choice(remaining, size=min(n_extra, len(remaining)), replace=False)
            return np.concatenate([candidates, extra])
        return candidates

    return rng.choice(observed, size=n_attack, replace=False)


def _attack_random_falsification(
    obs: torch.Tensor,
    targets: np.ndarray,
    scale: float,
    bias: float,
) -> torch.Tensor:
    """Random falsification: obs_new = obs * scale + bias.

    Simple but effective — mimics a compromised sensor sending
    scaled/offset readings.
    """
    out = obs.clone()
    for idx in targets:
        out[idx] = out[idx] * scale + bias
    return out


def _attack_replay(
    obs: torch.Tensor,
    targets: np.ndarray,
    replay_values: torch.Tensor | None,
) -> torch.Tensor:
    """Replay attack: replace current reading with a past value.

    The attacker records legitimate sensor readings and replays them
    later to mask real changes in the network state. This is particularly
    dangerous because individual replayed values look realistic.
    """
    out = obs.clone()
    if replay_values is None:
        return out  # no history yet, skip
    for idx in targets:
        if idx < len(replay_values):
            out[idx] = replay_values[idx]
    return out


def _attack_stealthy_bias(
    obs: torch.Tensor,
    targets: np.ndarray,
    rng: np.random.Generator,
    snapshot_idx: int,
    max_drift: float = 5.0,
    ramp_steps: int = 20,
) -> torch.Tensor:
    """Stealthy bias injection: small gradual drift over time.

    Instead of a sudden large change, the attacker slowly shifts
    readings. By the time the drift is large enough to matter,
    operators have adjusted to the "new normal".

    drift(t) = max_drift * min(t / ramp_steps, 1.0) * direction
    """
    out = obs.clone()
    # How far along the ramp are we?
    ramp_factor = min(snapshot_idx / max(ramp_steps, 1), 1.0)
    for idx in targets:
        direction = rng.choice([-1.0, 1.0])
        drift = max_drift * ramp_factor * direction
        out[idx] = out[idx] + drift
    return out


def _attack_gaussian_noise_injection(
    obs: torch.Tensor,
    targets: np.ndarray,
    rng: np.random.Generator,
    noise_multiplier: float = 5.0,
) -> torch.Tensor:
    """Noise injection: add large Gaussian noise to readings.

    Simulates a malfunctioning or jammed sensor producing noisy output.
    The readings fluctuate wildly around the true value.
    """
    out = obs.clone()
    for idx in targets:
        noise = rng.normal(0, abs(out[idx].item()) * noise_multiplier * 0.1 + 1.0)
        out[idx] = out[idx] + noise
    return out


def _apply_attacks(
    p_obs: torch.Tensor,
    q_obs: torch.Tensor,
    p_mask: torch.Tensor,
    q_mask: torch.Tensor,
    p_true: torch.Tensor,
    q_true: torch.Tensor,
    cfg: CorruptionConfig,
    rng: np.random.Generator,
    replay_buffer: dict | None = None,
    snapshot_idx: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply adversarial attacks to observed values.

    Only attacks *observed* sensors (can't attack what's not there).

    Supported attack types:
        - "random": Random falsification (scale + bias)
        - "replay": Replay past legitimate readings
        - "stealthy": Gradual bias drift over time
        - "noise": Inject large random noise
        - "mixed": Randomly pick from all attack types per snapshot

    Returns updated (p_obs, q_obs, p_anomaly, q_anomaly).
    """
    N = p_obs.shape[0]
    NE = q_obs.shape[0]

    p_anomaly = torch.zeros(N, dtype=torch.float32)
    q_anomaly = torch.zeros(NE, dtype=torch.float32)

    # Determine high-impact nodes for targeted attacks
    high_impact_p = None
    high_impact_q = None
    if cfg.attack_type == "targeted":
        # Attack nodes with highest pressure variance (most informative)
        p_vals = p_true.numpy()
        top_k = max(1, int(N * 0.3))
        high_impact_p = np.argsort(np.abs(p_vals - np.mean(p_vals)))[-top_k:]
        q_vals = q_true.numpy()
        top_k_q = max(1, int(NE * 0.3))
        high_impact_q = np.argsort(np.abs(q_vals))[-top_k_q:]

    # Select targets
    targets_p = _select_targets(p_mask, cfg.attack_fraction, rng, high_impact_p)
    targets_q = _select_targets(q_mask, cfg.attack_fraction, rng, high_impact_q)

    # Resolve attack type (for "mixed", pick one randomly per snapshot)
    attack = cfg.attack_type
    if attack == "mixed":
        attack = rng.choice(["random", "replay", "stealthy", "noise"])
    elif attack == "targeted":
        attack = "random"  # targeted just changes sensor selection, uses random falsification

    # Apply the chosen attack
    replay_p = replay_buffer.get("pressure") if replay_buffer else None
    replay_q = replay_buffer.get("flow") if replay_buffer else None

    if attack == "random":
        p_obs = _attack_random_falsification(p_obs, targets_p, cfg.attack_scale, cfg.attack_bias)
        q_obs = _attack_random_falsification(q_obs, targets_q, cfg.attack_scale, cfg.attack_bias)
    elif attack == "replay":
        p_obs = _attack_replay(p_obs, targets_p, replay_p)
        q_obs = _attack_replay(q_obs, targets_q, replay_q)
    elif attack == "stealthy":
        p_obs = _attack_stealthy_bias(p_obs, targets_p, rng, snapshot_idx,
                                       max_drift=cfg.attack_bias)
        q_obs = _attack_stealthy_bias(q_obs, targets_q, rng, snapshot_idx,
                                       max_drift=cfg.attack_bias * 0.1)
    elif attack == "noise":
        p_obs = _attack_gaussian_noise_injection(p_obs, targets_p, rng)
        q_obs = _attack_gaussian_noise_injection(q_obs, targets_q, rng)

    # Mark attacked sensors
    for idx in targets_p:
        p_anomaly[idx] = 1.0
    for idx in targets_q:
        q_anomaly[idx] = 1.0

    return p_obs, q_obs, p_anomaly, q_anomaly


# -----------------------------------------------------------------------
# Batch corruption
# -----------------------------------------------------------------------

def corrupt_all_snapshots(
    snapshots: list,
    cfg: CorruptionConfig,
    seed: int = 42,
) -> list[CorruptedSnapshot]:
    """Apply corruption to all snapshots.

    Maintains a replay buffer for replay attacks (stores the previous
    snapshot's observations to use as replay values).

    Args:
        snapshots: List of Snapshot objects with .pressure_true and .flow_true.
        cfg: Corruption configuration.
        seed: Random seed.

    Returns:
        List of CorruptedSnapshot objects (same order as input).
    """
    rng = np.random.default_rng(seed)
    corrupted = []
    replay_buffer = None

    for i, snap in enumerate(snapshots):
        c = corrupt_snapshot(
            snap.pressure_true, snap.flow_true, cfg, rng,
            replay_buffer=replay_buffer,
            snapshot_idx=i,
        )
        corrupted.append(c)

        # Update replay buffer with current clean observations (pre-attack)
        # Attacker records legitimate readings for future replay
        if cfg.attack_enabled and cfg.attack_type in ("replay", "mixed"):
            replay_buffer = {
                "pressure": snap.pressure_true.clone(),
                "flow": snap.flow_true.clone(),
            }

    return corrupted
