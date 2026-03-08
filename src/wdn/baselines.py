"""Analytical baseline methods for WDN state reconstruction.

Implements the pseudo-inverse and WLS methods from:
    Cattai et al., "GraphSmart: A Method for Green and Accurate IoT Water Monitoring"
    (referenced in Locatelli et al., Section 3.1)

These serve as comparison baselines for the GNN models.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass

from wdn.data_generation import WDNGraph


@dataclass
class BaselineResult:
    """Result from a baseline reconstruction method."""
    pressure_pred: torch.Tensor    # (N,) reconstructed pressures
    flow_pred: torch.Tensor        # (NE,) reconstructed flows


# ---------------------------------------------------------------------------
# 1. Pseudo-inverse baseline (from the paper, Section 3.1)
# ---------------------------------------------------------------------------

def pseudoinverse_baseline(
    graph: WDNGraph,
    pressure_obs: torch.Tensor,
    flow_obs: torch.Tensor,
    pressure_mask: torch.Tensor,
    flow_mask: torch.Tensor,
    rcond: float = 1e-6,
) -> BaselineResult:
    """Reconstruct missing flows using the incidence matrix pseudo-inverse.

    From the paper:
        f_U = B_U^+ · B_K · f_K

    Where:
        B is the incidence matrix (N × NE)
        B_K = columns of B for known (observed) edges
        B_U = columns of B for unknown (missing) edges
        f_K = observed flow values
        f_U = estimated flow values for unobserved edges

    For pressure: simple mean imputation of missing values.

    Args:
        graph: Static WDN graph with incidence matrix.
        pressure_obs: (N,) observed pressures (0 where missing).
        flow_obs: (NE,) observed flows (0 where missing).
        pressure_mask: (N,) binary mask (1=observed).
        flow_mask: (NE,) binary mask (1=observed).
        rcond: Cutoff for pseudo-inverse singular values.

    Returns:
        BaselineResult with reconstructed pressure and flow.
    """
    B = graph.incidence_matrix  # (N, NE)
    q_obs = flow_obs.numpy()
    q_mask = flow_mask.numpy().astype(bool)

    # --- Flow reconstruction ---
    known_idx = np.where(q_mask)[0]
    unknown_idx = np.where(~q_mask)[0]

    q_pred = q_obs.copy()

    if len(unknown_idx) > 0 and len(known_idx) > 0:
        B_K = B[:, known_idx]      # (N, n_known)
        B_U = B[:, unknown_idx]    # (N, n_unknown)
        f_K = q_obs[known_idx]     # (n_known,)

        # B_U^+ · B_K · f_K
        B_U_pinv = np.linalg.pinv(B_U, rcond=rcond)  # (n_unknown, N)
        f_U = B_U_pinv @ B_K @ f_K                    # (n_unknown,)

        q_pred[unknown_idx] = f_U

    # --- Pressure reconstruction: mean imputation ---
    p_obs = pressure_obs.numpy()
    p_mask = pressure_mask.numpy().astype(bool)

    p_pred = p_obs.copy()
    if p_mask.sum() > 0 and (~p_mask).sum() > 0:
        mean_p = p_obs[p_mask].mean()
        p_pred[~p_mask] = mean_p

    return BaselineResult(
        pressure_pred=torch.tensor(p_pred, dtype=torch.float32),
        flow_pred=torch.tensor(q_pred, dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# 2. Weighted Least Squares (WLS) baseline
# ---------------------------------------------------------------------------

def wls_baseline(
    graph: WDNGraph,
    pressure_obs: torch.Tensor,
    flow_obs: torch.Tensor,
    pressure_mask: torch.Tensor,
    flow_mask: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.1,
    diag_eps: float = 1e-6,
) -> BaselineResult:
    """Reconstruct using Weighted Least Squares with physics constraints.

    Flow reconstruction:
        Minimize: ||W_q (q - q_obs)||^2 + alpha * ||B^T q||^2
        Where W_q is a diagonal weight matrix (1 for observed, 0 for missing)
        and B^T q ≈ 0 enforces mass conservation (Kirchhoff's law).

        Closed-form: q = (W_q + alpha * B^T B + eps*I)^{-1} W_q q_obs

    Pressure reconstruction:
        Minimize: ||W_p (p - p_obs)||^2 + beta * ||L p||^2
        Where L is the graph Laplacian (smoothness prior).

        Closed-form: p = (W_p + beta * L + eps*I)^{-1} W_p p_obs

    Args:
        graph: Static WDN graph.
        pressure_obs, flow_obs: Observed values.
        pressure_mask, flow_mask: Binary masks.
        alpha: Weight for mass conservation constraint.
        beta: Weight for Laplacian smoothness.
        diag_eps: Small diagonal for numerical stability.

    Returns:
        BaselineResult with reconstructed pressure and flow.
    """
    B = graph.incidence_matrix     # (N, NE)
    A = graph.adjacency_matrix     # (N, N)
    N = graph.num_nodes
    NE = graph.num_edges

    # --- Flow reconstruction with mass conservation ---
    q_obs = flow_obs.numpy()
    q_mask_np = flow_mask.numpy()

    W_q = np.diag(q_mask_np)                          # (NE, NE)
    BtB = B.T @ B                                      # (NE, NE) — mass conservation
    M_q = W_q + alpha * BtB + diag_eps * np.eye(NE)    # (NE, NE)
    rhs_q = W_q @ q_obs                                # (NE,)

    q_pred = np.linalg.solve(M_q, rhs_q)

    # --- Pressure reconstruction with Laplacian smoothness ---
    p_obs = pressure_obs.numpy()
    p_mask_np = pressure_mask.numpy()

    W_p = np.diag(p_mask_np)                           # (N, N)
    # Graph Laplacian: L = D - A
    degree = A.sum(axis=1)
    L = np.diag(degree) - A                            # (N, N)
    M_p = W_p + beta * L + diag_eps * np.eye(N)        # (N, N)
    rhs_p = W_p @ p_obs                                # (N,)

    p_pred = np.linalg.solve(M_p, rhs_p)

    return BaselineResult(
        pressure_pred=torch.tensor(p_pred, dtype=torch.float32),
        flow_pred=torch.tensor(q_pred, dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# 3. Residual-based anomaly detection (for baselines)
# ---------------------------------------------------------------------------

def residual_anomaly_scores(
    obs: torch.Tensor,
    pred: torch.Tensor,
    mask: torch.Tensor,
    mad_scale: float = 3.0,
) -> torch.Tensor:
    """Compute anomaly scores based on reconstruction residuals.

    Anomaly score = |obs - pred| / (MAD * mad_scale)
    Where MAD = Median Absolute Deviation of residuals on observed values.

    Values > 1.0 are flagged as anomalous.

    Args:
        obs: (K,) observed values.
        pred: (K,) reconstructed values.
        mask: (K,) binary mask (1=observed).
        mad_scale: Threshold multiplier for MAD.

    Returns:
        (K,) anomaly scores (higher = more anomalous). 0 for missing values.
    """
    residuals = torch.abs(obs - pred) * mask

    # Compute MAD on observed residuals only
    observed_residuals = residuals[mask > 0]
    if len(observed_residuals) == 0:
        return torch.zeros_like(obs)

    mad = torch.median(observed_residuals)
    if mad < 1e-8:
        mad = torch.tensor(1e-8)

    scores = residuals / (mad * mad_scale)
    return scores


def residual_anomaly_predict(
    obs: torch.Tensor,
    pred: torch.Tensor,
    mask: torch.Tensor,
    mad_scale: float = 3.0,
) -> torch.Tensor:
    """Binary anomaly prediction: 1 if anomaly score > 1.0, else 0."""
    scores = residual_anomaly_scores(obs, pred, mask, mad_scale)
    return (scores > 1.0).float()
