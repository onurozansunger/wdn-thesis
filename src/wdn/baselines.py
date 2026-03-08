from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class BaselineConfig:
    pinv_rcond: float = 1e-4
    wls_alpha: float = 1e-2
    wls_beta: float = 1e-2
    diag_eps: float = 1e-6


def incidence_matrix(num_nodes: int, edge_index: np.ndarray) -> np.ndarray:
    """Build incidence matrix B (N x E) from edge_index (2, E)."""
    num_edges = edge_index.shape[1]
    B = np.zeros((num_nodes, num_edges), dtype=np.float32)
    for e in range(num_edges):
        i = int(edge_index[0, e])
        j = int(edge_index[1, e])
        B[i, e] = -1.0
        B[j, e] = 1.0
    return B


def analytical_reconstruct_flows(
    q_obs: np.ndarray,
    q_mask: np.ndarray,
    B: np.ndarray,
    rcond: float = 1e-4,
) -> np.ndarray:
    """Reconstruct missing flows via pseudo-inverse on incidence matrix."""
    known = q_mask.astype(bool)
    unknown = ~known
    q_hat = q_obs.copy()
    if np.all(known):
        return q_hat

    B_K = B[:, known]
    B_U = B[:, unknown]
    if B_U.size == 0:
        return q_hat

    rhs = B_K @ q_obs[known]
    pinv = np.linalg.pinv(B_U, rcond=rcond)
    q_u_hat = pinv @ rhs
    q_hat[unknown] = q_u_hat
    return q_hat


def wls_reconstruct_flows(
    q_obs: np.ndarray,
    q_mask: np.ndarray,
    B: np.ndarray,
    alpha: float,
    diag_eps: float,
) -> np.ndarray:
    """Weighted least squares with mass conservation penalty."""
    num_edges = q_obs.shape[0]
    W = np.diag(q_mask.astype(np.float32))
    BtB = B.T @ B
    A = W + alpha * BtB + diag_eps * np.eye(num_edges, dtype=np.float32)
    b = W @ q_obs
    q_hat = np.linalg.solve(A, b)
    return q_hat


def graph_laplacian(num_nodes: int, edge_index: np.ndarray) -> np.ndarray:
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for e in range(edge_index.shape[1]):
        i = int(edge_index[0, e])
        j = int(edge_index[1, e])
        A[i, j] = 1.0
        A[j, i] = 1.0
    D = np.diag(np.sum(A, axis=1))
    return D - A


def wls_reconstruct_pressures(
    p_obs: np.ndarray,
    p_mask: np.ndarray,
    L: np.ndarray,
    beta: float,
    diag_eps: float,
) -> np.ndarray:
    """Smooth pressures via Laplacian-regularized least squares."""
    num_nodes = p_obs.shape[0]
    W = np.diag(p_mask.astype(np.float32))
    A = W + beta * L + diag_eps * np.eye(num_nodes, dtype=np.float32)
    b = W @ p_obs
    p_hat = np.linalg.solve(A, b)
    return p_hat


def baseline_reconstruct(
    p_obs: np.ndarray,
    q_obs: np.ndarray,
    p_mask: np.ndarray,
    q_mask: np.ndarray,
    edge_index: np.ndarray,
    config: BaselineConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    return analytical_baseline(p_obs, q_obs, p_mask, q_mask, edge_index, config)


def analytical_baseline(
    p_obs: np.ndarray,
    q_obs: np.ndarray,
    p_mask: np.ndarray,
    q_mask: np.ndarray,
    edge_index: np.ndarray,
    config: BaselineConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Baseline A: pseudo-inverse flow reconstruction + mean-impute pressures."""
    num_nodes = p_obs.shape[0]
    B = incidence_matrix(num_nodes, edge_index)
    q_hat = analytical_reconstruct_flows(q_obs, q_mask, B, rcond=config.pinv_rcond)
    p_hat = p_obs.copy()
    if np.any(p_mask):
        mean_p = float(np.mean(p_obs[p_mask]))
    else:
        mean_p = 0.0
    p_hat[~p_mask] = mean_p
    return p_hat, q_hat


def wls_baseline(
    p_obs: np.ndarray,
    q_obs: np.ndarray,
    p_mask: np.ndarray,
    q_mask: np.ndarray,
    edge_index: np.ndarray,
    config: BaselineConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Baseline B: WLS with mass conservation and Laplacian pressure smoothing."""
    num_nodes = p_obs.shape[0]
    B = incidence_matrix(num_nodes, edge_index)
    L = graph_laplacian(num_nodes, edge_index)
    q_hat = wls_reconstruct_flows(q_obs, q_mask, B, alpha=config.wls_alpha, diag_eps=config.diag_eps)
    p_hat = wls_reconstruct_pressures(p_obs, p_mask, L, beta=config.wls_beta, diag_eps=config.diag_eps)
    return p_hat, q_hat


def residual_anomaly_scores(
    obs: np.ndarray,
    recon: np.ndarray,
    mask: np.ndarray,
    mad_scale: float = 3.5,
) -> Tuple[np.ndarray, float]:
    resid = (obs - recon)[mask]
    if resid.size == 0:
        return np.zeros_like(obs), 0.0
    median = np.median(resid)
    mad = np.median(np.abs(resid - median)) + 1e-8
    threshold = mad_scale * mad
    scores = np.zeros_like(obs, dtype=np.float32)
    scores[mask] = np.abs(obs[mask] - recon[mask]) / threshold
    return scores, threshold
