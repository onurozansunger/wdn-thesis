"""Mixture-of-Experts GNN for attack-specific state reconstruction and detection.

Motivation:
    A single multi-task GNN handles all attack types with one shared set of
    weights. Different attacks leave very different fingerprints in the data
    (random falsification looks nothing like a gradual stealthy drift), so a
    single model is forced to compromise. Replay in particular is the hardest
    case because individual replayed values look perfectly legitimate.

Architecture:
    1. Router: a small GNN that classifies the dominant attack type of the
       incoming snapshot from graph-level pooled features. Produces a
       distribution over K attack classes.
    2. Experts: K copies of the MultiTaskGNN, one per attack class. Each
       expert is free to specialize on its own attack fingerprint.
    3. Aggregator: the final reconstruction / anomaly outputs are a
       soft-weighted combination of the expert outputs, where the weights
       per sample come from the router distribution. At inference this
       reduces to picking the argmax expert if `hard_routing=True`.

Training:
    - Router is supervised with cross-entropy against the true attack label
      that `corruption.py` now stores on each CorruptedSnapshot.
    - Experts are trained jointly end-to-end through the weighted mixture.
    - A small load-balancing entropy bonus discourages router collapse.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from wdn.models.gnn import GNNBackbone, MLP
from wdn.models.multitask import MultiTaskGNN


class AttackRouter(nn.Module):
    """Small GNN that classifies the dominant attack type of a graph.

    Outputs unnormalized logits over `num_classes` attack types. The router
    is intentionally lightweight compared to the experts: it only needs to
    decide *which* expert should handle the sample, not reconstruct the
    state itself.
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int = 32,
        num_classes: int = 6,
        num_layers: int = 2,
        dropout: float = 0.1,
        gnn_type: str = "GraphSAGE",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)
        self.gnn = GNNBackbone(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            gnn_type=gnn_type,
            heads=4,
            edge_dim=edge_in_dim,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = torch.relu(self.node_encoder(x))
        h = self.gnn(h, edge_index, edge_attr, batch=batch)
        if batch is None:
            # Single graph -> mean pool over all nodes
            g = h.mean(dim=0, keepdim=True)
        else:
            g = global_mean_pool(h, batch)
        return self.classifier(g)  # (num_graphs, num_classes)


class MixtureOfExpertsGNN(nn.Module):
    """Mixture-of-Experts wrapper around MultiTaskGNN.

    The router routes each graph in the batch to one of K experts. During
    training we use a soft mixture (weighted sum of expert outputs) so the
    gradients flow through every expert. At inference we support hard
    routing which only runs the argmax expert for speed and interpretability.

    Args:
        node_in_dim: Node feature dim (shared by router and experts).
        edge_in_dim: Edge feature dim.
        hidden_dim: Expert hidden dim.
        num_experts: Number of experts (one per attack class).
        router_hidden_dim: Router hidden dim (typically much smaller).
        num_layers: Expert GNN depth.
        dropout: Dropout.
        gnn_type: GNN backbone type for experts.
        heads: Attention heads (GAT/Transformer experts).
        hard_routing: If True, forward uses top-1 hard dispatch.
    """

    def __init__(
        self,
        node_in_dim: int = 7,
        edge_in_dim: int = 8,
        hidden_dim: int = 64,
        num_experts: int = 6,
        router_hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
        gnn_type: str = "GraphSAGE",
        heads: int = 4,
        hard_routing: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.hard_routing = hard_routing

        self.router = AttackRouter(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=router_hidden_dim,
            num_classes=num_experts,
            num_layers=num_layers,
            dropout=dropout,
            gnn_type=gnn_type,
        )

        self.experts = nn.ModuleList([
            MultiTaskGNN(
                node_in_dim=node_in_dim,
                edge_in_dim=edge_in_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                gnn_type=gnn_type,
                heads=heads,
            )
            for _ in range(num_experts)
        ])

    def _node_graph_id(
        self,
        num_nodes: int,
        batch: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        """Return a (num_nodes,) long tensor with the graph id for each node."""
        if batch is None:
            return torch.zeros(num_nodes, dtype=torch.long, device=device)
        return batch

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
        """Forward pass with soft (default) or hard routing.

        Returns a dict with the same keys MultiTaskGNN produces, plus:
            - router_logits: (B, K) raw router output
            - router_probs:  (B, K) softmax weights actually used to mix
            - expert_weights_node: (N, K) per-node weight used for mixing
        """
        device = x.device
        node_graph_id = self._node_graph_id(x.shape[0], batch, device)

        # ---- Router ----
        router_logits = self.router(x, edge_index, edge_attr, batch=batch)
        router_probs = F.softmax(router_logits, dim=-1)  # (B, K)

        # Soft weights per sample. If hard_routing, collapse to one-hot on
        # the argmax class (still differentiable via straight-through in a
        # pinch, but we only use hard routing at eval).
        if self.hard_routing and not self.training:
            top = router_probs.argmax(dim=-1)                     # (B,)
            mix = F.one_hot(top, num_classes=self.num_experts).float()
        else:
            mix = router_probs                                    # (B, K)

        # ---- Experts ----
        # We always run every expert for simplicity — K is small (<=6) and
        # each graph is tiny. The per-sample mixing happens after.
        expert_outs = []
        for expert in self.experts:
            out = expert(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                is_original_edge=is_original_edge,
                batch=batch,
                pressure_obs=pressure_obs,
                flow_obs=flow_obs,
                pressure_mask=pressure_mask,
                flow_mask=flow_mask,
            )
            expert_outs.append(out)

        # Stack per-node and per-original-edge expert outputs.
        p_stack = torch.stack([o["pressure_pred"] for o in expert_outs], dim=-1)    # (N, K)
        q_stack = torch.stack([o["flow_pred"] for o in expert_outs], dim=-1)        # (NE, K)

        # Broadcast per-graph router weights to nodes and original edges.
        node_weights = mix[node_graph_id]                                            # (N, K)

        orig_src = edge_index[0][is_original_edge]
        edge_graph_id = node_graph_id[orig_src]
        edge_weights = mix[edge_graph_id]                                            # (NE, K)

        pressure_pred = (p_stack * node_weights).sum(dim=-1)
        flow_pred = (q_stack * edge_weights).sum(dim=-1)

        result: dict[str, torch.Tensor] = {
            "pressure_pred": pressure_pred,
            "flow_pred": flow_pred,
            "router_logits": router_logits,
            "router_probs": router_probs,
            "expert_weights_node": node_weights,
        }

        # Mix anomaly logits the same way.
        if "pressure_anomaly_logits" in expert_outs[0]:
            pa_stack = torch.stack(
                [o["pressure_anomaly_logits"] for o in expert_outs], dim=-1
            )                                                                        # (N, K)
            result["pressure_anomaly_logits"] = (pa_stack * node_weights).sum(dim=-1)

        if "flow_anomaly_logits" in expert_outs[0]:
            qa_stack = torch.stack(
                [o["flow_anomaly_logits"] for o in expert_outs], dim=-1
            )                                                                        # (NE, K)
            result["flow_anomaly_logits"] = (qa_stack * edge_weights).sum(dim=-1)

        return result


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def moe_loss(
    out: dict[str, torch.Tensor],
    batch,
    base_loss_fn,
    lambda_router: float = 0.5,
    lambda_balance: float = 0.01,
    lambda_anomaly: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Combined MoE loss.

    Args:
        out: Model forward output (must contain router_logits).
        batch: PyG batch with y_pressure, y_flow, masks, anomaly labels,
            and attack_type (graph-level class id).
        base_loss_fn: Callable returning recon + anomaly dict, same
            signature as `multitask_loss`.
        lambda_router: Weight on router cross-entropy.
        lambda_balance: Weight on entropy bonus that discourages collapse.
        lambda_anomaly: Passed through to base_loss_fn.
    """
    base = base_loss_fn(
        out["pressure_pred"], batch.y_pressure,
        out["flow_pred"], batch.y_flow,
        batch.pressure_mask, batch.flow_mask,
        out.get("pressure_anomaly_logits"),
        out.get("flow_anomaly_logits"),
        batch.pressure_anomaly,
        batch.flow_anomaly,
        lambda_anomaly=lambda_anomaly,
        loss_on_all=True,
    )

    # Router CE: supervise the router with the ground-truth attack class.
    router_logits = out["router_logits"]                                    # (B, K)
    targets = batch.attack_type.view(-1).long()                             # (B,)
    router_ce = F.cross_entropy(router_logits, targets)

    # Load-balancing: penalize low entropy of the *batch-mean* router
    # distribution (averaged over samples). If one expert takes everything,
    # batch-mean becomes one-hot and entropy drops to 0; we want to keep
    # at least some diversity so all experts receive gradient signal.
    mean_probs = F.softmax(router_logits, dim=-1).mean(dim=0)
    # Negative entropy so *minimizing* this pushes toward uniform.
    balance = (mean_probs * (mean_probs + 1e-10).log()).sum()

    total = (
        base["total_loss"]
        + lambda_router * router_ce
        + lambda_balance * balance
    )

    return {
        "recon_loss": base["recon_loss"],
        "anomaly_loss": base["anomaly_loss"],
        "router_ce": router_ce,
        "balance": balance,
        "total_loss": total,
    }
