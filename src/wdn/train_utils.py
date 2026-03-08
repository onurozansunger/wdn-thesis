from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
from torch import nn


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return torch.mean((pred - target) ** 2)
    mask = mask.float()
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return torch.sum(((pred - target) ** 2) * mask) / (mask.sum() + 1e-8)


def recon_loss(
    p_pred: torch.Tensor,
    q_pred: torch.Tensor,
    p_true: torch.Tensor,
    q_true: torch.Tensor,
    p_mask: torch.Tensor,
    q_mask: torch.Tensor,
    loss_on_all: bool,
) -> torch.Tensor:
    if loss_on_all:
        p_loss = masked_mse(p_pred, p_true)
        q_loss = masked_mse(q_pred, q_true)
    else:
        p_loss = masked_mse(p_pred, p_true, ~p_mask)
        q_loss = masked_mse(q_pred, q_true, ~q_mask)
    return p_loss + q_loss


def anomaly_loss(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    p_labels: torch.Tensor,
    q_labels: torch.Tensor,
) -> torch.Tensor:
    bce = nn.BCEWithLogitsLoss()
    p_loss = bce(p_logits, p_labels)
    q_loss = bce(q_logits, q_labels)
    return p_loss + q_loss


def batch_to_device(batch: Iterable[Dict[str, torch.Tensor]], device: torch.device):
    for sample in batch:
        for key, value in sample.items():
            if torch.is_tensor(value):
                sample[key] = value.to(device)
    return batch
