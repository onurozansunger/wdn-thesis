from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from wdn.models.gnn import GraphSAGE, MLP, make_bidirectional


class ReconGNN(nn.Module):
    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)
        self.gnn = GraphSAGE(hidden_dim, hidden_dim, num_layers, dropout)

        self.node_head = MLP(hidden_dim, hidden_dim, 1, dropout)
        self.edge_head = MLP(hidden_dim * 2 + edge_in_dim, hidden_dim, 1, dropout)

    def forward(
        self,
        edge_index: torch.Tensor,
        node_static: torch.Tensor,
        edge_static: torch.Tensor,
        p_obs: torch.Tensor,
        q_obs: torch.Tensor,
        p_mask: torch.Tensor,
        q_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        node_feats = torch.cat(
            [node_static, p_obs.unsqueeze(-1), p_mask.unsqueeze(-1).float()], dim=-1
        )
        node_feats = self.node_encoder(node_feats)

        bi_edge_index = make_bidirectional(edge_index)
        node_emb = self.gnn(node_feats, bi_edge_index)

        p_hat = self.node_head(node_emb).squeeze(-1)

        src = edge_index[0]
        dst = edge_index[1]
        edge_feats = torch.cat(
            [node_emb[src], node_emb[dst], edge_static, q_obs.unsqueeze(-1), q_mask.unsqueeze(-1).float()],
            dim=-1,
        )
        q_hat = self.edge_head(edge_feats).squeeze(-1)

        return p_hat, q_hat
