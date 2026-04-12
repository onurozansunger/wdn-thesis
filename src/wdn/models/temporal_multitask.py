"""TemporalMultiTaskGNN: Spatio-temporal model for state reconstruction and anomaly detection.

Architecture:
    1. Shared encoder: node features -> hidden_dim
    2. Spatial GNN + Temporal GRU: processes T consecutive snapshots
    3. Reconstruction heads: pressure/flow prediction from final hidden state
    4. Anomaly heads: binary classification using embeddings + residuals

The temporal component enables detection of time-dependent attacks
(e.g., replay attacks) that single-snapshot models cannot capture.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from wdn.models.gnn import GNNBackbone, TemporalGNN, MLP


class TemporalMultiTaskGNN(nn.Module):
    """Spatio-temporal joint reconstruction + anomaly detection model.

    Uses GNN for spatial message passing at each timestep, then GRU
    to capture temporal dynamics across a window of T snapshots.

    Args:
        node_in_dim: Input dimension for node features.
        edge_in_dim: Input dimension for edge features.
        hidden_dim: Hidden layer dimension.
        num_layers: Number of GNN layers.
        num_temporal_layers: Number of GRU layers.
        window_size: Number of consecutive timesteps (T).
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
        num_temporal_layers: int = 1,
        window_size: int = 6,
        dropout: float = 0.1,
        gnn_type: str = "GraphSAGE",
        heads: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.window_size = window_size

        # Shared encoder for each timestep
        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)

        # Spatial backbone
        spatial_backbone = GNNBackbone(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            gnn_type=gnn_type,
            heads=heads,
            edge_dim=edge_in_dim,
        )

        # Spatio-temporal backbone
        self.temporal_gnn = TemporalGNN(
            spatial_backbone=spatial_backbone,
            hidden_dim=hidden_dim,
            num_temporal_layers=num_temporal_layers,
            dropout=dropout,
        )

        # --- Reconstruction heads ---
        self.pressure_head = MLP(hidden_dim, hidden_dim, 1, dropout)
        self.flow_head = MLP(hidden_dim * 2 + edge_in_dim, hidden_dim, 1, dropout)

        # --- Anomaly detection heads ---
        # Input: [node_embedding, pressure_obs, pressure_pred, |obs - pred|, mask]
        self.pressure_anomaly_head = MLP(
            hidden_dim + 4, hidden_dim // 2, 1, dropout,
        )
        # Input: [src_emb, dst_emb, flow_obs, flow_pred, |obs - pred|, mask]
        self.flow_anomaly_head = MLP(
            hidden_dim * 2 + 4, hidden_dim // 2, 1, dropout,
        )

    def forward(
        self,
        x_seq: list[torch.Tensor],
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        is_original_edge: torch.Tensor,
        pressure_obs: torch.Tensor | None = None,
        flow_obs: torch.Tensor | None = None,
        pressure_mask: torch.Tensor | None = None,
        flow_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass on a sequence of snapshots.

        Args:
            x_seq: List of T tensors, each (N, node_in_dim).
            edge_index: (2, E) shared graph structure.
            edge_attr: (E, edge_in_dim) shared edge features.
            is_original_edge: (E,) bool mask for original edges.
            pressure_obs: (N,) observed pressures for the LAST timestep.
            flow_obs: (NE,) observed flows for the LAST timestep.
            pressure_mask: (N,) observation mask for the LAST timestep.
            flow_mask: (NE,) observation mask for the LAST timestep.

        Returns:
            Dict with reconstruction and anomaly predictions.
        """
        # 1. Encode each timestep
        encoded_seq = [torch.relu(self.node_encoder(x_t)) for x_t in x_seq]

        # 2. Spatio-temporal encoding (GNN + GRU)
        h = self.temporal_gnn(encoded_seq, edge_index, edge_attr)  # (N, hidden_dim)

        # 3. Reconstruction (predict for the last timestep)
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

        # 4. Anomaly detection
        if pressure_obs is not None and pressure_mask is not None:
            p_residual = torch.abs(pressure_obs - pressure_pred).detach()
            p_anomaly_input = torch.stack([
                pressure_obs,
                pressure_pred.detach(),
                p_residual,
                pressure_mask,
            ], dim=-1)
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
            ], dim=-1)
            q_anomaly_input = torch.cat([src_emb, dst_emb, q_anomaly_input], dim=-1)
            q_anomaly_logits = self.flow_anomaly_head(q_anomaly_input).squeeze(-1)
            result["flow_anomaly_logits"] = q_anomaly_logits

        return result
