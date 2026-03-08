from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class CorruptionResult:
    p_obs: np.ndarray
    q_obs: np.ndarray
    p_mask: np.ndarray
    q_mask: np.ndarray
    p_anom: np.ndarray
    q_anom: np.ndarray


def _apply_missing(rng: np.random.Generator, size: int, missing_rate: float) -> np.ndarray:
    mask = rng.random(size) > missing_rate
    return mask


def _apply_noise(rng: np.random.Generator, values: np.ndarray, sigma: float, mask: np.ndarray) -> np.ndarray:
    noisy = values.copy()
    noise = rng.normal(0.0, sigma, size=values.shape)
    noisy[mask] = noisy[mask] + noise[mask]
    return noisy


def apply_corruption(
    rng: np.random.Generator,
    p_true: np.ndarray,
    q_true: np.ndarray,
    missing_p: float,
    missing_q: float,
    noise_sigma_p: float,
    noise_sigma_q: float,
    attack_enabled: bool = False,
    attack_fraction: float = 0.0,
    attack_bias_p: float = 0.0,
    attack_bias_q: float = 0.0,
    attack_scale_p: float = 1.0,
    attack_scale_q: float = 1.0,
    targeted: bool = False,
    target_node_idx: Optional[List[int]] = None,
    target_edge_idx: Optional[List[int]] = None,
    time_index: Optional[int] = None,
    time_window: Optional[Tuple[int, int]] = None,
) -> CorruptionResult:
    p_mask = _apply_missing(rng, p_true.size, missing_p)
    q_mask = _apply_missing(rng, q_true.size, missing_q)

    p_obs = _apply_noise(rng, p_true, noise_sigma_p, p_mask)
    q_obs = _apply_noise(rng, q_true, noise_sigma_q, q_mask)

    p_anom = np.zeros_like(p_true, dtype=bool)
    q_anom = np.zeros_like(q_true, dtype=bool)

    if attack_enabled and attack_fraction > 0.0:
        if time_window is not None and time_index is not None:
            if not (time_window[0] <= time_index <= time_window[1]):
                return CorruptionResult(p_obs, q_obs, p_mask, q_mask, p_anom, q_anom)

        if targeted and target_node_idx:
            node_candidates = np.array(target_node_idx, dtype=int)
        else:
            node_candidates = np.where(p_mask)[0]
        if targeted and target_edge_idx:
            edge_candidates = np.array(target_edge_idx, dtype=int)
        else:
            edge_candidates = np.where(q_mask)[0]

        num_node_attacks = max(1, int(len(node_candidates) * attack_fraction)) if len(node_candidates) > 0 else 0
        num_edge_attacks = max(1, int(len(edge_candidates) * attack_fraction)) if len(edge_candidates) > 0 else 0

        if num_node_attacks > 0:
            attacked_nodes = rng.choice(node_candidates, size=num_node_attacks, replace=False)
            p_obs[attacked_nodes] = p_obs[attacked_nodes] * attack_scale_p + attack_bias_p
            p_anom[attacked_nodes] = True

        if num_edge_attacks > 0:
            attacked_edges = rng.choice(edge_candidates, size=num_edge_attacks, replace=False)
            q_obs[attacked_edges] = q_obs[attacked_edges] * attack_scale_q + attack_bias_q
            q_anom[attacked_edges] = True

    p_obs_masked = p_obs.copy()
    q_obs_masked = q_obs.copy()
    p_obs_masked[~p_mask] = np.nan
    q_obs_masked[~q_mask] = np.nan

    return CorruptionResult(p_obs_masked, q_obs_masked, p_mask, q_mask, p_anom, q_anom)
