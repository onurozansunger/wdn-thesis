"""Evaluation metrics for reconstruction and anomaly detection.

Reconstruction: MAE, MSE, RMSE (computed on all values or only unobserved).
Anomaly detection: Precision, Recall, F1, AUROC.
"""

from __future__ import annotations

import torch
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from dataclasses import dataclass


@dataclass
class ReconMetrics:
    """Reconstruction quality metrics."""
    mae: float
    mse: float
    rmse: float

    def __repr__(self) -> str:
        return f"ReconMetrics(MAE={self.mae:.4f}, MSE={self.mse:.6f}, RMSE={self.rmse:.4f})"


@dataclass
class AnomalyMetrics:
    """Anomaly detection metrics."""
    precision: float
    recall: float
    f1: float
    auroc: float

    def __repr__(self) -> str:
        return (f"AnomalyMetrics(P={self.precision:.3f}, R={self.recall:.3f}, "
                f"F1={self.f1:.3f}, AUROC={self.auroc:.3f})")


# ---------------------------------------------------------------------------
# Reconstruction metrics
# ---------------------------------------------------------------------------

def compute_recon_metrics(
    pred: torch.Tensor,
    true: torch.Tensor,
    mask: torch.Tensor | None = None,
    only_unobserved: bool = False,
) -> ReconMetrics:
    """Compute reconstruction metrics.

    Args:
        pred: (K,) predicted values.
        true: (K,) ground truth values.
        mask: (K,) observation mask (1=observed, 0=missing).
            If None, compute on all values.
        only_unobserved: If True and mask is provided, compute only on
            unobserved (missing) values. This measures how well the model
            reconstructs what it can't see.

    Returns:
        ReconMetrics with MAE, MSE, RMSE.
    """
    if mask is not None and only_unobserved:
        # Only evaluate on missing values
        selector = (mask == 0)
        if selector.sum() == 0:
            return ReconMetrics(mae=0.0, mse=0.0, rmse=0.0)
        pred = pred[selector]
        true = true[selector]

    diff = pred - true
    mae = torch.abs(diff).mean().item()
    mse = (diff ** 2).mean().item()
    rmse = np.sqrt(mse)

    return ReconMetrics(mae=mae, mse=mse, rmse=rmse)


# ---------------------------------------------------------------------------
# Anomaly detection metrics
# ---------------------------------------------------------------------------

def compute_anomaly_metrics(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
    scores: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> AnomalyMetrics:
    """Compute anomaly detection metrics.

    Args:
        pred_labels: (K,) predicted binary labels (0 or 1).
        true_labels: (K,) ground truth binary labels.
        scores: (K,) continuous anomaly scores (for AUROC).
            If None, AUROC is computed from pred_labels.
        mask: (K,) observation mask. If provided, only evaluate on observed values.

    Returns:
        AnomalyMetrics with precision, recall, F1, AUROC.
    """
    if mask is not None:
        selector = mask > 0
        pred_labels = pred_labels[selector]
        true_labels = true_labels[selector]
        if scores is not None:
            scores = scores[selector]

    y_pred = pred_labels.numpy().astype(int)
    y_true = true_labels.numpy().astype(int)

    # Handle edge cases (no positives in ground truth)
    if y_true.sum() == 0:
        return AnomalyMetrics(
            precision=1.0 if y_pred.sum() == 0 else 0.0,
            recall=1.0,
            f1=1.0 if y_pred.sum() == 0 else 0.0,
            auroc=1.0,
        )

    if y_true.sum() == len(y_true):
        # All positive — edge case
        return AnomalyMetrics(
            precision=1.0 if y_pred.sum() == len(y_pred) else y_true.sum() / max(y_pred.sum(), 1),
            recall=y_pred.sum() / y_true.sum(),
            f1=f1_score(y_true, y_pred, zero_division=0.0),
            auroc=0.5,
        )

    precision = precision_score(y_true, y_pred, zero_division=0.0)
    recall = recall_score(y_true, y_pred, zero_division=0.0)
    f1 = f1_score(y_true, y_pred, zero_division=0.0)

    # AUROC uses continuous scores if available
    if scores is not None:
        s = scores.numpy()
        try:
            auroc = roc_auc_score(y_true, s)
        except ValueError:
            auroc = 0.5
    else:
        try:
            auroc = roc_auc_score(y_true, y_pred.astype(float))
        except ValueError:
            auroc = 0.5

    return AnomalyMetrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        auroc=float(auroc),
    )


# ---------------------------------------------------------------------------
# Aggregate metrics over a dataset
# ---------------------------------------------------------------------------

def aggregate_metrics(metrics_list: list[ReconMetrics]) -> ReconMetrics:
    """Average a list of ReconMetrics."""
    if not metrics_list:
        return ReconMetrics(mae=0.0, mse=0.0, rmse=0.0)
    mae = np.mean([m.mae for m in metrics_list])
    mse = np.mean([m.mse for m in metrics_list])
    rmse = np.sqrt(mse)
    return ReconMetrics(mae=float(mae), mse=float(mse), rmse=float(rmse))
