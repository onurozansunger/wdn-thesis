"""Mixture-of-Experts Attacker.

Symmetric to the defender's TemporalMixtureOfExpertsGNN: a small router
picks one of ``K`` specialist attackers per snapshot, and the experts'
outputs are mixed by the router probabilities. The intended outcome is
for each expert to specialise on a different attack mode (e.g. one
expert favours single-sensor large perturbations, another favours
distributed small perturbations) without any explicit attack-class
supervision — the diversity loss + load-balancing entropy do the job.

The output API is identical to ``AttackerGNN`` (a dict with delta_p,
delta_q, mask_p_logits, mask_q_logits) so the existing self-play loop
and stealth-budget projection work unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from wdn.models.attacker import AttackerGNN
from wdn.models.gnn import GNNBackbone


class AttackerRouter(nn.Module):
    """Graph-pooled classifier that picks an expert per snapshot."""

    def __init__(
        self, node_in_dim: int, edge_in_dim: int,
        hidden_dim: int = 32, num_experts: int = 4,
        gnn_type: str = "GraphSAGE",
    ):
        super().__init__()
        self.encoder = nn.Linear(node_in_dim, hidden_dim)
        self.gnn = GNNBackbone(
            in_dim=hidden_dim, hidden_dim=hidden_dim,
            num_layers=2, dropout=0.1, gnn_type=gnn_type,
            heads=4, edge_dim=edge_in_dim,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor,
        edge_attr: torch.Tensor, num_nodes_per_graph: int,
    ) -> torch.Tensor:
        h = torch.relu(self.encoder(x))
        h = self.gnn(h, edge_index, edge_attr)         # (B*N, H)
        N = num_nodes_per_graph
        B = h.shape[0] // N
        graph_emb = h.view(B, N, -1).mean(dim=1)       # (B, H)
        return self.classifier(graph_emb)              # (B, K)


class MixtureOfAttackersGNN(nn.Module):
    """Population of ``num_experts`` AttackerGNNs plus a router.

    The forward pass returns the same dict an ``AttackerGNN`` returns,
    so it is a drop-in replacement in the self-play training loop.
    Internally we run every expert and mix their outputs by the router
    probabilities. The router logits are also exposed so the training
    loop can add a load-balancing penalty.
    """

    def __init__(
        self,
        node_in_dim: int = 7,
        edge_in_dim: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 2,
        gnn_type: str = "GraphSAGE",
        dropout: float = 0.1,
        num_experts: int = 4,
        router_hidden_dim: int = 32,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.router = AttackerRouter(
            node_in_dim=node_in_dim, edge_in_dim=edge_in_dim,
            hidden_dim=router_hidden_dim, num_experts=num_experts,
            gnn_type=gnn_type,
        )
        self.experts = nn.ModuleList([
            AttackerGNN(
                node_in_dim=node_in_dim, edge_in_dim=edge_in_dim,
                hidden_dim=hidden_dim, num_layers=num_layers,
                gnn_type=gnn_type, dropout=dropout,
            )
            for _ in range(num_experts)
        ])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        is_original_edge: torch.Tensor,
        num_nodes_per_graph: int | None = None,
    ) -> dict[str, torch.Tensor]:
        # Router needs the per-graph node count; if not provided assume
        # batch_size = 1 (used at smoke-test time).
        N = num_nodes_per_graph if num_nodes_per_graph else x.shape[0]
        B = x.shape[0] // N

        router_logits = self.router(x, edge_index, edge_attr, N)   # (B, K)
        router_probs = F.softmax(router_logits, dim=-1)            # (B, K)

        # Run every expert.
        outs = [
            e(x, edge_index, edge_attr, is_original_edge)
            for e in self.experts
        ]

        # Mix node-level outputs: shape (B*N,) for each expert,
        # multiplied by the router weight of the graph the node belongs
        # to. ``repeat_interleave(N)`` broadcasts router_probs to
        # node-level.
        delta_p_stack = torch.stack([o["delta_p"] for o in outs], dim=-1)  # (B*N, K)
        mask_p_stack = torch.stack([o["mask_p_logits"] for o in outs], dim=-1)
        node_w = router_probs.repeat_interleave(N, dim=0)                  # (B*N, K)

        delta_p = (delta_p_stack * node_w).sum(dim=-1)
        mask_p_logits = (mask_p_stack * node_w).sum(dim=-1)

        # Mix edge-level outputs.
        NE = outs[0]["delta_q"].shape[0] // max(B, 1)
        delta_q_stack = torch.stack([o["delta_q"] for o in outs], dim=-1)
        mask_q_stack = torch.stack([o["mask_q_logits"] for o in outs], dim=-1)
        edge_w = router_probs.repeat_interleave(NE, dim=0)
        delta_q = (delta_q_stack * edge_w).sum(dim=-1)
        mask_q_logits = (mask_q_stack * edge_w).sum(dim=-1)

        return {
            "delta_p": delta_p,
            "delta_q": delta_q,
            "mask_p_logits": mask_p_logits,
            "mask_q_logits": mask_q_logits,
            "router_logits": router_logits,
            "router_probs": router_probs,
            # Per-expert raw outputs are useful for the diversity loss and
            # for vocabulary mining downstream.
            "expert_delta_p": delta_p_stack,
            "expert_delta_q": delta_q_stack,
        }


def diversity_loss(expert_deltas: torch.Tensor) -> torch.Tensor:
    """Encourage experts to disagree.

    ``expert_deltas`` has shape (M, K) where M is the number of
    perturbed entries (nodes or edges) and K is the number of experts.
    We compute the mean pairwise cosine SIMILARITY between expert
    columns and return it as the loss — a smaller mean similarity =
    more diverse experts. Returns 0 if K < 2.
    """
    K = expert_deltas.shape[-1]
    if K < 2:
        return expert_deltas.new_zeros(())
    # Normalise per-expert vectors.
    norms = expert_deltas.norm(dim=0, keepdim=True) + 1e-6
    normed = expert_deltas / norms                              # (M, K)
    sim = normed.T @ normed                                     # (K, K)
    # Drop the diagonal and average the off-diagonal similarities.
    mask = 1.0 - torch.eye(K, device=expert_deltas.device)
    return (sim * mask).sum() / (K * (K - 1))


def balance_loss(router_logits: torch.Tensor) -> torch.Tensor:
    """Negative entropy of batch-mean router probabilities.

    Same shape as the defender MoE balance term: penalises a router
    that always picks the same expert.
    """
    probs = F.softmax(router_logits, dim=-1).mean(dim=0)
    return (probs * (probs + 1e-10).log()).sum()
