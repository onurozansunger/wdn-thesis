"""GNN backbone using PyTorch Geometric.

Supports multiple architectures:
    - GAT: Graph Attention Network (attention-based, explainable)
    - GATv2: Improved GAT with dynamic attention
    - GraphSAGE: Sample & aggregate
    - GCN: Graph Convolutional Network
    - Transformer: Graph Transformer with global attention
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import (
    GATConv, GATv2Conv, SAGEConv, GCNConv,
    TransformerConv, GPSConv, global_mean_pool,
)


class GNNBackbone(nn.Module):
    """Multi-layer GNN encoder that produces node embeddings.

    Args:
        in_dim: Input feature dimension per node.
        hidden_dim: Hidden layer dimension.
        num_layers: Number of GNN layers.
        dropout: Dropout rate.
        gnn_type: "GAT", "GATv2", "GraphSAGE", "GCN", "Transformer", "GPS".
        heads: Number of attention heads (GAT/Transformer).
        edge_dim: Edge feature dimension.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        gnn_type: str = "GAT",
        heads: int = 4,
        edge_dim: int | None = None,
    ):
        super().__init__()
        self.gnn_type = gnn_type
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_channels = in_dim if i == 0 else hidden_dim

            if gnn_type == "GAT":
                if i < num_layers - 1:
                    conv = GATConv(
                        in_channels, hidden_dim // heads,
                        heads=heads, dropout=dropout,
                        edge_dim=edge_dim, concat=True,
                    )
                else:
                    conv = GATConv(
                        in_channels, hidden_dim,
                        heads=1, dropout=dropout,
                        edge_dim=edge_dim, concat=False,
                    )

            elif gnn_type == "GATv2":
                if i < num_layers - 1:
                    conv = GATv2Conv(
                        in_channels, hidden_dim // heads,
                        heads=heads, dropout=dropout,
                        edge_dim=edge_dim, concat=True,
                    )
                else:
                    conv = GATv2Conv(
                        in_channels, hidden_dim,
                        heads=1, dropout=dropout,
                        edge_dim=edge_dim, concat=False,
                    )

            elif gnn_type == "Transformer":
                if i < num_layers - 1:
                    conv = TransformerConv(
                        in_channels, hidden_dim // heads,
                        heads=heads, dropout=dropout,
                        edge_dim=edge_dim, concat=True,
                    )
                else:
                    conv = TransformerConv(
                        in_channels, hidden_dim,
                        heads=1, dropout=dropout,
                        edge_dim=edge_dim, concat=False,
                    )

            elif gnn_type == "GPS":
                # GPS wraps a local MPNN + global attention
                local_conv = GATConv(
                    hidden_dim, hidden_dim // heads,
                    heads=heads, dropout=dropout,
                    edge_dim=edge_dim, concat=True,
                )
                conv = GPSConv(
                    channels=hidden_dim,
                    conv=local_conv,
                    heads=heads,
                    dropout=dropout,
                    attn_type="multihead",
                )
                # GPS requires input to be hidden_dim already
                if i == 0 and in_channels != hidden_dim:
                    self.input_proj = nn.Linear(in_channels, hidden_dim)

            elif gnn_type == "GraphSAGE":
                conv = SAGEConv(in_channels, hidden_dim)

            elif gnn_type == "GCN":
                conv = GCNConv(in_channels, hidden_dim)

            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")

            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.drop = nn.Dropout(dropout)
        self.out_dim = hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list]:
        """Forward pass.

        Args:
            x: (N, in_dim) node features.
            edge_index: (2, E) edge connectivity.
            edge_attr: (E, edge_dim) edge features.
            batch: (N,) batch assignment for GPS global attention.
            return_attention: Return attention weights (GAT/Transformer).

        Returns:
            h: (N, hidden_dim) node embeddings.
            attn_weights: List of attention tensors (if return_attention).
        """
        attn_weights = []

        h = x
        # GPS needs input projected to hidden_dim
        if self.gnn_type == "GPS" and hasattr(self, "input_proj"):
            h = self.input_proj(h)

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            if self.gnn_type in ("GAT", "GATv2", "Transformer"):
                if return_attention:
                    h_new, (edge_idx, alpha) = conv(
                        h, edge_index, edge_attr=edge_attr,
                        return_attention_weights=True,
                    )
                    attn_weights.append(alpha)
                else:
                    h_new = conv(h, edge_index, edge_attr=edge_attr)

            elif self.gnn_type == "GPS":
                h_new = conv(h, edge_index, batch=batch, edge_attr=edge_attr)

            else:
                h_new = conv(h, edge_index)

            h_new = norm(h_new)

            # Residual connection (after first layer when dims match)
            if i > 0:
                h_new = h_new + h

            h = torch.relu(h_new)
            h = self.drop(h)

        if return_attention:
            return h, attn_weights
        return h


class MLP(nn.Module):
    """Multi-layer perceptron for prediction heads."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalGNN(nn.Module):
    """Spatio-temporal GNN: spatial GNN + temporal GRU.

    Processes a sequence of graph snapshots:
        1. Spatial encoding: GNNBackbone on each timestep
        2. Temporal encoding: GRU over the sequence of node embeddings
        3. Output: final hidden state as node embeddings

    Args:
        spatial_backbone: GNNBackbone for spatial feature extraction.
        hidden_dim: Dimension of GRU hidden state.
        num_temporal_layers: Number of GRU layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        spatial_backbone: GNNBackbone,
        hidden_dim: int = 64,
        num_temporal_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.spatial = spatial_backbone
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(
            input_size=spatial_backbone.out_dim,
            hidden_size=hidden_dim,
            num_layers=num_temporal_layers,
            batch_first=True,
            dropout=dropout if num_temporal_layers > 1 else 0.0,
        )

        # Project GRU output back to hidden_dim if needed
        if spatial_backbone.out_dim != hidden_dim:
            self.out_proj = nn.Linear(hidden_dim, spatial_backbone.out_dim)
        else:
            self.out_proj = None

        self.out_dim = spatial_backbone.out_dim

    def forward(
        self,
        x_seq: list[torch.Tensor],
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass on a sequence of snapshots.

        Args:
            x_seq: List of T tensors, each (N, node_in_dim).
            edge_index: (2, E) shared graph structure.
            edge_attr: (E, edge_dim) shared edge features.

        Returns:
            h: (N, out_dim) node embeddings from the last timestep.
        """
        T = len(x_seq)
        N = x_seq[0].shape[0]

        # Spatial encoding: GNN on each timestep
        spatial_out = []
        for t in range(T):
            h_t = self.spatial(x_seq[t], edge_index, edge_attr)
            spatial_out.append(h_t)  # (N, hidden_dim)

        # Stack into (N, T, hidden_dim) for GRU
        h_seq = torch.stack(spatial_out, dim=1)  # (N, T, hidden_dim)

        # Temporal encoding
        gru_out, _ = self.gru(h_seq)  # (N, T, hidden_dim)
        h_final = gru_out[:, -1, :]   # (N, hidden_dim) — last timestep

        if self.out_proj is not None:
            h_final = self.out_proj(h_final)

        return h_final
