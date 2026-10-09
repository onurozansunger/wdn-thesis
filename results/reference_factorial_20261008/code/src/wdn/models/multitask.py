"""MultiTaskGNN: Joint state reconstruction and anomaly detection.

Architecture:
    1. Shared GNN backbone: learns node + edge embeddings
    2. Reconstruction head: pressure/flow prediction (same as ReconGNN)
    3. Anomaly head: binary classification per sensor

The anomaly detector uses reconstruction residuals + learned embeddings.
This is more powerful than post-hoc residual thresholding because:
    - The model learns what "normal" reconstruction error looks like
    - Shared representations let reconstruction inform detection
    - End-to-end training jointly optimizes both objectives
"""

from __future__ import annotations

import torch
import torch.nn as nn

from wdn.models.gnn import GNNBackbone, MLP


class MultiTaskGNN(nn.Module):
    """Joint reconstruction + anomaly detection model.

    Args:
        node_in_dim: Input dimension for node features.
        edge_in_dim: Input dimension for edge features.
        hidden_dim: Hidden layer dimension.
        num_layers: Number of GNN layers.
        dropout: Dropout rate.
        gnn_type: GNN architecture type.
        heads: Attention heads (for GAT/Transformer).
    """

    def __init__(
        self,
        node_in_dim: int = 7,
        edge_in_dim: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        gnn_type: str = "GraphSAGE",
        heads: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Shared encoder
        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)

        # Shared GNN backbone
        self.gnn = GNNBackbone(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            gnn_type=gnn_type,
            heads=heads,
            edge_dim=edge_in_dim,
        )

        # --- Reconstruction heads (same as ReconGNN) ---
        self.pressure_head = MLP(hidden_dim, hidden_dim, 1, dropout)
        self.flow_head = MLP(hidden_dim * 2 + edge_in_dim, hidden_dim, 1, dropout)

        # --- Anomaly detection heads ---
        # Node anomaly: embedding + reconstruction residual features
        # Input: [node_embedding, pressure_obs, pressure_pred, |obs - pred|, mask]
        self.pressure_anomaly_head = MLP(
            hidden_dim + 4, hidden_dim // 2, 1, dropout,
        )

        # Edge anomaly: similar for flow
        # Input: [src_emb, dst_emb, flow_obs, flow_pred, |obs - pred|, mask]
        self.flow_anomaly_head = MLP(
            hidden_dim * 2 + 4, hidden_dim // 2, 1, dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        is_original_edge: torch.Tensor,
        batch: torch.Tensor | None = None,
        pressure_obs: torch.Tensor | None = None,
        flow_obs: torch.Tensor | None = None,
        pressure_mask: torch.Tensor | None = None,
        flow_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: (N, node_in_dim) node features.
            edge_index: (2, E) edges.
            edge_attr: (E, edge_in_dim) edge features.
            is_original_edge: (E,) bool mask for original edges.
            batch: (N,) batch vector.
            pressure_obs: (N,) normalized observed pressures.
            flow_obs: (NE_orig,) normalized observed flows.
            pressure_mask: (N,) observation mask.
            flow_mask: (NE_orig,) observation mask.

        Returns:
            Dict with reconstruction and anomaly predictions.
        """
        # 1. Encode
        h = torch.relu(self.node_encoder(x))

        # 2. GNN
        h = self.gnn(h, edge_index, edge_attr, batch=batch)

        # 3. Reconstruction
        pressure_pred = self.pressure_head(h).squeeze(-1)

        orig_src = edge_index[0][is_original_edge]
        orig_dst = edge_index[1][is_original_edge]
        src_emb = h[orig_src]
        dst_emb = h[orig_dst]
        edge_feat = edge_attr[is_original_edge]
        edge_input = torch.cat([src_emb, dst_emb, edge_feat], dim=-1)
        flow_pred = self.flow_head(edge_input).squeeze(-1)

        result = {
            "pressure_pred": pressure_pred,
            "flow_pred": flow_pred,
            "node_embeddings": h,
        }

        # 4. Anomaly detection (only when obs/mask are provided)
        if pressure_obs is not None and pressure_mask is not None:
            p_residual = torch.abs(pressure_obs - pressure_pred).detach()
            p_anomaly_input = torch.stack([
                pressure_obs,
                pressure_pred.detach(),
                p_residual,
                pressure_mask,
            ], dim=-1)  # (N, 4)
            p_anomaly_input = torch.cat([h, p_anomaly_input], dim=-1)
            p_anomaly_logits = self.pressure_anomaly_head(p_anomaly_input).squeeze(-1)
            result["pressure_anomaly_logits"] = p_anomaly_logits

        if flow_obs is not None and flow_mask is not None:
            q_residual = torch.abs(flow_obs - flow_pred).detach()
            q_anomaly_input = torch.stack([
                flow_obs,
                flow_pred.detach(),
                q_residual,
                flow_mask,
            ], dim=-1)  # (NE, 4)
            q_anomaly_input = torch.cat([src_emb, dst_emb, q_anomaly_input], dim=-1)
            q_anomaly_logits = self.flow_anomaly_head(q_anomaly_input).squeeze(-1)
            result["flow_anomaly_logits"] = q_anomaly_logits

        return result

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        is_original_edge: torch.Tensor,
        batch: torch.Tensor | None = None,
        pressure_obs: torch.Tensor | None = None,
        flow_obs: torch.Tensor | None = None,
        pressure_mask: torch.Tensor | None = None,
        flow_mask: torch.Tensor | None = None,
        n_samples: int = 30,
    ) -> dict[str, torch.Tensor]:
        """MC Dropout uncertainty estimation."""
        self.train()

        p_samples, q_samples = [], []
        p_anom_samples, q_anom_samples = [], []

        with torch.no_grad():
            for _ in range(n_samples):
                out = self.forward(
                    x, edge_index, edge_attr, is_original_edge, batch,
                    pressure_obs, flow_obs, pressure_mask, flow_mask,
                )
                p_samples.append(out["pressure_pred"])
                q_samples.append(out["flow_pred"])
                if "pressure_anomaly_logits" in out:
                    p_anom_samples.append(torch.sigmoid(out["pressure_anomaly_logits"]))
                if "flow_anomaly_logits" in out:
                    q_anom_samples.append(torch.sigmoid(out["flow_anomaly_logits"]))

        p_stack = torch.stack(p_samples, dim=0)
        q_stack = torch.stack(q_samples, dim=0)

        result = {
            "pressure_mean": p_stack.mean(dim=0),
            "pressure_std": p_stack.std(dim=0),
            "flow_mean": q_stack.mean(dim=0),
            "flow_std": q_stack.std(dim=0),
        }

        if p_anom_samples:
            pa_stack = torch.stack(p_anom_samples, dim=0)
            result["pressure_anomaly_prob"] = pa_stack.mean(dim=0)
            result["pressure_anomaly_uncertainty"] = pa_stack.std(dim=0)

        if q_anom_samples:
            qa_stack = torch.stack(q_anom_samples, dim=0)
            result["flow_anomaly_prob"] = qa_stack.mean(dim=0)
            result["flow_anomaly_uncertainty"] = qa_stack.std(dim=0)

        return result


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def multitask_loss(
    pressure_pred: torch.Tensor,
    pressure_true: torch.Tensor,
    flow_pred: torch.Tensor,
    flow_true: torch.Tensor,
    pressure_mask: torch.Tensor,
    flow_mask: torch.Tensor,
    pressure_anomaly_logits: torch.Tensor | None = None,
    flow_anomaly_logits: torch.Tensor | None = None,
    pressure_anomaly_true: torch.Tensor | None = None,
    flow_anomaly_true: torch.Tensor | None = None,
    lambda_anomaly: float = 1.0,
    loss_on_all: bool = True,
) -> dict[str, torch.Tensor]:
    """Combined reconstruction + anomaly detection loss.

    Args:
        pressure_pred/true: Pressure predictions and targets.
        flow_pred/true: Flow predictions and targets.
        pressure_mask/flow_mask: Observation masks.
        pressure_anomaly_logits/flow_anomaly_logits: Anomaly logits.
        pressure_anomaly_true/flow_anomaly_true: Ground truth anomaly labels.
        lambda_anomaly: Weight for anomaly loss.
        loss_on_all: Compute reconstruction on all or only unobserved.

    Returns:
        Dict with loss components and total loss.
    """
    # Reconstruction loss
    if loss_on_all:
        recon_loss = (
            nn.functional.mse_loss(pressure_pred, pressure_true)
            + nn.functional.mse_loss(flow_pred, flow_true)
        )
    else:
        unobs_p = (pressure_mask == 0)
        unobs_q = (flow_mask == 0)
        p_loss = nn.functional.mse_loss(pressure_pred[unobs_p], pressure_true[unobs_p]) if unobs_p.sum() > 0 else torch.tensor(0.0, device=pressure_pred.device)
        q_loss = nn.functional.mse_loss(flow_pred[unobs_q], flow_true[unobs_q]) if unobs_q.sum() > 0 else torch.tensor(0.0, device=flow_pred.device)
        recon_loss = p_loss + q_loss

    result = {"recon_loss": recon_loss}

    # Anomaly detection loss (BCE on observed sensors only)
    anomaly_loss = torch.tensor(0.0, device=pressure_pred.device)

    if (pressure_anomaly_logits is not None
            and pressure_anomaly_true is not None):
        # Only compute on observed sensors (can't detect anomalies in missing data)
        obs_p = pressure_mask > 0
        if obs_p.sum() > 0:
            anomaly_loss = anomaly_loss + nn.functional.binary_cross_entropy_with_logits(
                pressure_anomaly_logits[obs_p],
                pressure_anomaly_true[obs_p],
            )

    if (flow_anomaly_logits is not None
            and flow_anomaly_true is not None):
        obs_q = flow_mask > 0
        if obs_q.sum() > 0:
            anomaly_loss = anomaly_loss + nn.functional.binary_cross_entropy_with_logits(
                flow_anomaly_logits[obs_q],
                flow_anomaly_true[obs_q],
            )

    result["anomaly_loss"] = anomaly_loss
    result["total_loss"] = recon_loss + lambda_anomaly * anomaly_loss

    return result
