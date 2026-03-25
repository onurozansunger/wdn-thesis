"""ReconGNN: Graph Neural Network for WDN state reconstruction.

Architecture:
    1. Linear encoder: raw node features -> hidden_dim
    2. GNN backbone (GAT/GATv2/Transformer/GPS/GraphSAGE/GCN): message passing
    3. Node head (MLP): node embeddings -> pressure prediction
    4. Edge head (MLP): [src_emb, dst_emb, edge_features] -> flow prediction

Uncertainty quantification:
    MC Dropout: enable dropout at inference time, run multiple forward passes,
    and compute mean/variance of predictions as point estimates + confidence.

Physics-informed loss:
    Mass conservation at each node: sum of incoming flows = sum of outgoing flows
    Enforced as: ||B^T * q_pred||^2 ~ 0 (B = incidence matrix)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

from wdn.models.gnn import GNNBackbone, MLP


class ReconGNN(nn.Module):
    """State reconstruction model for water distribution networks.

    Predicts pressure at every node and flow at every edge.

    Args:
        node_in_dim: Input dimension for node features.
        edge_in_dim: Input dimension for edge features.
        hidden_dim: Hidden layer dimension.
        num_layers: Number of GNN layers.
        dropout: Dropout rate.
        gnn_type: "GAT", "GATv2", "Transformer", "GPS", "GraphSAGE", "GCN".
        heads: Attention heads (GAT/Transformer).
    """

    def __init__(
        self,
        node_in_dim: int = 7,
        edge_in_dim: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        gnn_type: str = "GAT",
        heads: int = 4,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Node encoder: project raw features to hidden dim
        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)

        # GNN backbone
        self.gnn = GNNBackbone(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            gnn_type=gnn_type,
            heads=heads,
            edge_dim=edge_in_dim,
        )

        # Pressure prediction head (node-level)
        self.pressure_head = MLP(hidden_dim, hidden_dim, 1, dropout)

        # Flow prediction head (edge-level)
        # Input: [src_embedding, dst_embedding, edge_features]
        self.flow_head = MLP(hidden_dim * 2 + edge_in_dim, hidden_dim, 1, dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        is_original_edge: torch.Tensor,
        batch: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: (N_total, node_in_dim) node features (batched).
            edge_index: (2, E_total) bidirectional edge connectivity (batched).
            edge_attr: (E_total, edge_in_dim) edge features (batched).
            is_original_edge: (E_total,) bool mask for original directed edges.
            batch: (N_total,) batch assignment vector (for GPS).
            return_attention: Return attention weights.

        Returns:
            Dict with pressure_pred, flow_pred, node_embeddings, [attn_weights].
        """
        # 1. Encode node features
        h = torch.relu(self.node_encoder(x))

        # 2. GNN message passing
        if return_attention:
            h, attn_weights = self.gnn(
                h, edge_index, edge_attr, batch=batch, return_attention=True,
            )
        else:
            h = self.gnn(h, edge_index, edge_attr, batch=batch)
            attn_weights = None

        # 3. Pressure prediction (node-level)
        pressure_pred = self.pressure_head(h).squeeze(-1)

        # 4. Flow prediction (edge-level, original directed edges only)
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
        if attn_weights is not None:
            result["attn_weights"] = attn_weights

        return result

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        is_original_edge: torch.Tensor,
        batch: torch.Tensor | None = None,
        n_samples: int = 30,
    ) -> dict[str, torch.Tensor]:
        """MC Dropout uncertainty estimation.

        Runs multiple forward passes with dropout enabled to get
        a distribution of predictions. Returns mean and std as
        point estimate + confidence interval.

        Args:
            x, edge_index, edge_attr, is_original_edge, batch: Standard inputs.
            n_samples: Number of MC samples (more = better uncertainty estimate).

        Returns:
            Dict with pressure/flow mean, std, and all samples.
        """
        self.train()  # enable dropout

        p_samples = []
        q_samples = []

        with torch.no_grad():
            for _ in range(n_samples):
                out = self.forward(x, edge_index, edge_attr, is_original_edge, batch)
                p_samples.append(out["pressure_pred"])
                q_samples.append(out["flow_pred"])

        p_stack = torch.stack(p_samples, dim=0)  # (n_samples, N)
        q_stack = torch.stack(q_samples, dim=0)  # (n_samples, NE)

        return {
            "pressure_mean": p_stack.mean(dim=0),
            "pressure_std": p_stack.std(dim=0),
            "flow_mean": q_stack.mean(dim=0),
            "flow_std": q_stack.std(dim=0),
            "pressure_samples": p_stack,
            "flow_samples": q_stack,
        }


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def reconstruction_loss(
    pressure_pred: torch.Tensor,
    pressure_true: torch.Tensor,
    flow_pred: torch.Tensor,
    flow_true: torch.Tensor,
    pressure_mask: torch.Tensor | None = None,
    flow_mask: torch.Tensor | None = None,
    loss_on_all: bool = True,
) -> torch.Tensor:
    """MSE reconstruction loss for pressure and flow.

    Args:
        pressure_pred/true: (N,) pressure values.
        flow_pred/true: (NE,) flow values.
        pressure_mask/flow_mask: (N,)/(NE,) binary masks.
        loss_on_all: If True, compute loss on all values.
            If False, only on unobserved values (harder task).

    Returns:
        Scalar loss tensor.
    """
    if loss_on_all or pressure_mask is None:
        p_loss = nn.functional.mse_loss(pressure_pred, pressure_true)
    else:
        unobs_p = (pressure_mask == 0)
        if unobs_p.sum() > 0:
            p_loss = nn.functional.mse_loss(pressure_pred[unobs_p], pressure_true[unobs_p])
        else:
            p_loss = torch.tensor(0.0, device=pressure_pred.device)

    if loss_on_all or flow_mask is None:
        q_loss = nn.functional.mse_loss(flow_pred, flow_true)
    else:
        unobs_q = (flow_mask == 0)
        if unobs_q.sum() > 0:
            q_loss = nn.functional.mse_loss(flow_pred[unobs_q], flow_true[unobs_q])
        else:
            q_loss = torch.tensor(0.0, device=flow_pred.device)

    return p_loss + q_loss


def physics_loss(
    flow_pred: torch.Tensor,
    incidence_matrix: torch.Tensor,
    batch_size: int | None = None,
    num_edges_per_graph: int | None = None,
) -> torch.Tensor:
    """Mass conservation loss: B * q should be ~ 0 at junction nodes.

    Handles batched flow predictions by reshaping into per-graph chunks.

    Args:
        flow_pred: (NE_total,) predicted flow values (possibly batched).
        incidence_matrix: (N, NE) signed incidence matrix (single graph).
        batch_size: Number of graphs in batch.
        num_edges_per_graph: NE per graph. Required if batch_size > 1.

    Returns:
        Scalar physics violation loss.
    """
    NE = incidence_matrix.shape[1]

    if batch_size is not None and batch_size > 1 and num_edges_per_graph is not None:
        assert flow_pred.shape[0] == batch_size * num_edges_per_graph, (
            f"Expected {batch_size * num_edges_per_graph} flows, got {flow_pred.shape[0]}"
        )
        flow_batched = flow_pred.reshape(batch_size, num_edges_per_graph)
        net_flow = incidence_matrix @ flow_batched.T  # (N, B)
        return (net_flow ** 2).mean()
    else:
        net_flow = incidence_matrix @ flow_pred  # (N,)
        return (net_flow ** 2).mean()
