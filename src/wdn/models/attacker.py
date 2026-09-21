"""Adversarial Attacker GNN for self-play training.

The attacker observes a clean snapshot of the network and produces a
sparse, bounded perturbation of the sensor readings. The defender then
sees the corrupted snapshot and tries to flag the attacked sensors;
both networks are trained jointly in a Stackelberg self-play loop.

The output of a single forward pass for one snapshot is:

    delta_p   : (N,)  signed perturbation on every junction's pressure
    delta_q   : (NE,) signed perturbation on every pipe's flow
    mask_p    : (N,)  attack-mask logits for pressure sensors
    mask_q    : (NE,) attack-mask logits for flow sensors

Stealth is not enforced inside the network. The training loop calls
``apply_stealth_budget`` to clip the magnitudes and pick the top-k
sensors to attack, so the perturbation actually written into the
defender's input always satisfies the budget exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from wdn.models.gnn import GNNBackbone, MLP


@dataclass
class StealthBudget:
    """Constraints on what the attacker is allowed to do.

    ``epsilon_p`` and ``epsilon_q`` cap the L-infinity magnitude of the
    perturbation on pressures (in metres) and flows (in CMS). ``k_p``
    and ``k_q`` cap how many sensors the attacker may touch in one
    snapshot — typically a fraction of the network so that an outright
    "shut down all sensors" attack is impossible.
    """

    epsilon_p: float = 3.0
    epsilon_q: float = 0.05
    k_p: int = 4
    k_q: int = 4


class AttackerGNN(nn.Module):
    """Graph-aware attacker. Same backbone family as the defender so
    the two networks can be compared on equal footing.

    Args:
        node_in_dim: Input feature dim per junction (pressure_true,
            elevation, base_demand, type one-hot, ...). Matches the
            defender's node features.
        edge_in_dim: Input feature dim per pipe.
        hidden_dim: Hidden width.
        num_layers: GraphSAGE layers.
        gnn_type: Backbone family. GraphSAGE wins the spatial ablation
            so it is the default; kept configurable for ablations.
        dropout: Dropout on the backbone.
    """

    def __init__(
        self,
        node_in_dim: int = 7,
        edge_in_dim: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 2,
        gnn_type: str = "GraphSAGE",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)
        self.backbone = GNNBackbone(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            gnn_type=gnn_type,
            heads=4,
            edge_dim=edge_in_dim,
        )

        # Two heads on every node: one outputs a real-valued delta on
        # the pressure reading, one outputs the logit of "do I attack
        # this sensor?". A tanh squashes delta to [-1, 1] so the
        # stealth-budget projection only has to scale by epsilon.
        self.delta_p_head = MLP(hidden_dim, hidden_dim, 1, dropout)
        self.mask_p_head = MLP(hidden_dim, hidden_dim, 1, dropout)

        # Same for flow on each original edge: pool the two endpoint
        # embeddings + the edge feature, then two heads.
        self.delta_q_head = MLP(hidden_dim * 2 + edge_in_dim, hidden_dim, 1, dropout)
        self.mask_q_head = MLP(hidden_dim * 2 + edge_in_dim, hidden_dim, 1, dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        is_original_edge: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run the attacker on a single graph.

        Returns raw (pre-budget) outputs. The stealth-budget projection
        is applied separately by the training loop so the defender
        always receives a feasible attack.
        """
        h = torch.relu(self.node_encoder(x))
        h = self.backbone(h, edge_index, edge_attr)            # (N, H)

        delta_p = torch.tanh(self.delta_p_head(h).squeeze(-1))  # (N,) in [-1, 1]
        mask_p_logits = self.mask_p_head(h).squeeze(-1)         # (N,)

        # Edge features: concatenate src embedding, dst embedding, edge attribute.
        src = edge_index[0][is_original_edge]
        dst = edge_index[1][is_original_edge]
        edge_input = torch.cat(
            [h[src], h[dst], edge_attr[is_original_edge]], dim=-1
        )
        delta_q = torch.tanh(self.delta_q_head(edge_input).squeeze(-1))
        mask_q_logits = self.mask_q_head(edge_input).squeeze(-1)

        return {
            "delta_p": delta_p,
            "delta_q": delta_q,
            "mask_p_logits": mask_p_logits,
            "mask_q_logits": mask_q_logits,
        }


def apply_stealth_budget(
    out: dict[str, torch.Tensor],
    budget: StealthBudget,
    *,
    hard: bool = False,
) -> dict[str, torch.Tensor]:
    """Project the raw attacker output onto the stealth budget.

    The mask is taken to be the top-k sensors by ``mask_logits``; the
    perturbation magnitude is scaled by ``epsilon`` and gated by the
    mask. With ``hard=False`` (default) we use a soft mask (sigmoid
    times a top-k indicator) so gradients still flow into ``mask_logits``
    during training. With ``hard=True`` the mask is binary, used at
    evaluation time so the constraint is exactly satisfied.

    Returns a dict with the projected ``delta_p``, ``delta_q``, ``mask_p``
    and ``mask_q`` tensors and the indices of the chosen sensors.
    """
    mask_p_logits = out["mask_p_logits"]
    mask_q_logits = out["mask_q_logits"]

    # ---- Top-k selection on every snapshot ----
    k_p = min(budget.k_p, mask_p_logits.numel())
    k_q = min(budget.k_q, mask_q_logits.numel())
    top_p_idx = torch.topk(mask_p_logits, k_p).indices
    top_q_idx = torch.topk(mask_q_logits, k_q).indices

    indicator_p = mask_p_logits.new_zeros(mask_p_logits.shape)
    indicator_p[top_p_idx] = 1.0
    indicator_q = mask_q_logits.new_zeros(mask_q_logits.shape)
    indicator_q[top_q_idx] = 1.0

    if hard:
        mask_p = indicator_p
        mask_q = indicator_q
    else:
        # Sigmoid keeps the gradient flowing back into mask_logits;
        # multiplying by the indicator restricts the attack to exactly
        # k sensors per snapshot.
        mask_p = torch.sigmoid(mask_p_logits) * indicator_p
        mask_q = torch.sigmoid(mask_q_logits) * indicator_q

    delta_p = out["delta_p"] * budget.epsilon_p * mask_p
    delta_q = out["delta_q"] * budget.epsilon_q * mask_q

    return {
        "delta_p": delta_p,
        "delta_q": delta_q,
        "mask_p": mask_p,
        "mask_q": mask_q,
        "top_p_idx": top_p_idx,
        "top_q_idx": top_q_idx,
    }
