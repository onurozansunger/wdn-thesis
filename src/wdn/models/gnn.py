"""GNN backbone using PyTorch Geometric.

Supports GAT (Graph Attention Network), GraphSAGE, and GCN.
GAT is the default — gives us attention weights for explainability.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, SAGEConv, GCNConv


class GNNBackbone(nn.Module):
    """Multi-layer GNN encoder that produces node embeddings.

    Supports multiple GNN types:
        - GAT: Graph Attention Network (default, gives explainability)
        - GraphSAGE: Sample & aggregate
        - GCN: Graph Convolutional Network

    Args:
        in_dim: Input feature dimension per node.
        hidden_dim: Hidden layer dimension.
        num_layers: Number of GNN layers.
        dropout: Dropout rate.
        gnn_type: One of "GAT", "GraphSAGE", "GCN".
        heads: Number of attention heads (GAT only).
        edge_dim: Edge feature dimension (GAT supports edge features).
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
            if i == 0:
                in_channels = in_dim
            else:
                in_channels = hidden_dim

            if gnn_type == "GAT":
                # GAT: multi-head attention, concat heads except last layer
                if i < num_layers - 1:
                    # Intermediate layers: concat heads
                    conv = GATConv(
                        in_channels,
                        hidden_dim // heads,
                        heads=heads,
                        dropout=dropout,
                        edge_dim=edge_dim,
                        concat=True,
                    )
                else:
                    # Last layer: average heads for stable output dim
                    conv = GATConv(
                        in_channels,
                        hidden_dim,
                        heads=1,
                        dropout=dropout,
                        edge_dim=edge_dim,
                        concat=False,
                    )
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
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list]:
        """Forward pass.

        Args:
            x: (N, in_dim) node features.
            edge_index: (2, E) edge connectivity.
            edge_attr: (E, edge_dim) edge features (used by GAT).
            return_attention: If True, return attention weights (GAT only).

        Returns:
            h: (N, hidden_dim) node embeddings.
            attn_weights: List of attention weight tensors (if return_attention=True).
        """
        attn_weights = []

        h = x
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            if self.gnn_type == "GAT":
                if return_attention:
                    h_new, (edge_idx, alpha) = conv(
                        h, edge_index, edge_attr=edge_attr,
                        return_attention_weights=True,
                    )
                    attn_weights.append(alpha)
                else:
                    h_new = conv(h, edge_index, edge_attr=edge_attr)
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
    """Simple multi-layer perceptron for prediction heads."""

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
