from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


def make_bidirectional(edge_index: torch.Tensor) -> torch.Tensor:
    """Return edge_index with both directions."""
    src, dst = edge_index
    rev = torch.stack([dst, src], dim=0)
    return torch.cat([edge_index, rev], dim=1)


class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        agg = torch.zeros_like(x)
        agg.index_add_(0, src, x[dst])
        deg = torch.zeros(x.size(0), device=x.device)
        deg.index_add_(0, src, torch.ones_like(src, dtype=x.dtype))
        deg = deg.clamp(min=1).unsqueeze(1)
        agg = agg / deg

        out = self.lin_self(x) + self.lin_neigh(agg)
        out = F.relu(out)
        out = self.dropout(out)
        return out


class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        layers = []
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        layers.append(GraphSAGELayer(in_dim, hidden_dim, dropout))
        for _ in range(num_layers - 1):
            layers.append(GraphSAGELayer(hidden_dim, hidden_dim, dropout))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index)
        return x


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
