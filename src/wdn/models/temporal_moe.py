"""Temporal Mixture-of-Experts GNN.

Same idea as `MixtureOfExpertsGNN`, but every expert is a
`TemporalMultiTaskGNN` and the router also sees the temporal window.

Why temporal matters:
    A spatial-only expert cannot detect replay attacks — a replayed
    reading looks perfectly plausible on its own. Only by comparing
    against recent history can the model flag the value as stale.
    Giving each expert (and the router) a 6-step window restores the
    signal needed for replay detection, and the specialization layer
    on top lets the replay expert focus entirely on that fingerprint.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from wdn.models.gnn import GNNBackbone
from wdn.models.temporal_multitask import TemporalMultiTaskGNN


class TemporalAttackRouter(nn.Module):
    """Small spatio-temporal classifier for attack type.

    For each timestep the router runs a shallow GNN to produce node
    embeddings, averages them across the window, then pools per graph
    and classifies. The router is intentionally smaller than the
    experts — it only needs to pick which expert to use.
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
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        x_seq: list[torch.Tensor],
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch_size: int,
        num_nodes_per_graph: int,
    ) -> torch.Tensor:
        """Returns (batch_size, num_classes) logits.

        Assumes all graphs in the batch share the same topology so
        node ordering is deterministic: node i belongs to graph
        ``i // num_nodes_per_graph``.
        """
        # Encode + GNN at each timestep independently.
        embs = []
        for x_t in x_seq:
            h = torch.relu(self.node_encoder(x_t))
            h = self.gnn(h, edge_index, edge_attr)
            embs.append(h)

        stack = torch.stack(embs, dim=0)                   # (T, B*N, H)

        # Split the temporal stack into "recent" (last timestep) and
        # "history" (mean of the preceding timesteps). Concatenating the
        # two gives the classifier an explicit delta signal, which is
        # exactly what replay detection needs.
        last = stack[-1]                                   # (B*N, H)
        if stack.shape[0] > 1:
            history = stack[:-1].mean(dim=0)               # (B*N, H)
        else:
            history = last
        combined = torch.cat([last, history], dim=-1)      # (B*N, 2H)

        # Graph-level mean pool (same topology per graph, so reshape).
        B = batch_size
        N = num_nodes_per_graph
        combined = combined.view(B, N, -1).mean(dim=1)     # (B, 2H)

        return self.classifier(combined)                   # (B, num_classes)


class TemporalMixtureOfExpertsGNN(nn.Module):
    """Mixture-of-Experts wrapper around TemporalMultiTaskGNN."""

    def __init__(
        self,
        node_in_dim: int = 7,
        edge_in_dim: int = 8,
        hidden_dim: int = 48,
        num_experts: int = 6,
        router_hidden_dim: int = 32,
        num_layers: int = 2,
        num_temporal_layers: int = 1,
        window_size: int = 6,
        dropout: float = 0.1,
        gnn_type: str = "GraphSAGE",
        heads: int = 4,
        hard_routing: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.hard_routing = hard_routing

        self.router = TemporalAttackRouter(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=router_hidden_dim,
            num_classes=num_experts,
            num_layers=num_layers,
            dropout=dropout,
            gnn_type=gnn_type,
        )

        self.experts = nn.ModuleList([
            TemporalMultiTaskGNN(
                node_in_dim=node_in_dim,
                edge_in_dim=edge_in_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_temporal_layers=num_temporal_layers,
                window_size=window_size,
                dropout=dropout,
                gnn_type=gnn_type,
                heads=heads,
            )
            for _ in range(num_experts)
        ])

    def forward(
        self,
        x_seq: list[torch.Tensor],
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        is_original_edge: torch.Tensor,
        batch_size: int,
        num_nodes_per_graph: int,
        pressure_obs: torch.Tensor | None = None,
        flow_obs: torch.Tensor | None = None,
        pressure_mask: torch.Tensor | None = None,
        flow_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B = batch_size
        N = num_nodes_per_graph

        # ---- Router ----
        router_logits = self.router(
            x_seq, edge_index, edge_attr,
            batch_size=B, num_nodes_per_graph=N,
        )
        router_probs = F.softmax(router_logits, dim=-1)               # (B, K)

        if self.hard_routing and not self.training:
            top = router_probs.argmax(dim=-1)
            mix = F.one_hot(top, num_classes=self.num_experts).float()
        else:
            mix = router_probs

        # ---- Experts ----
        expert_outs = []
        for expert in self.experts:
            out = expert(
                x_seq=x_seq,
                edge_index=edge_index,
                edge_attr=edge_attr,
                is_original_edge=is_original_edge,
                pressure_obs=pressure_obs,
                flow_obs=flow_obs,
                pressure_mask=pressure_mask,
                flow_mask=flow_mask,
            )
            expert_outs.append(out)

        p_stack = torch.stack([o["pressure_pred"] for o in expert_outs], dim=-1)    # (B*N, K)
        q_stack = torch.stack([o["flow_pred"] for o in expert_outs], dim=-1)        # (B*NE, K)

        # Broadcast per-graph mix weights to nodes and original edges.
        node_weights = mix.repeat_interleave(N, dim=0)                 # (B*N, K)

        # Number of original edges per graph (same topology across batch).
        total_orig = p_stack.new_zeros(()).long() + q_stack.shape[0]
        NE = int(total_orig.item()) // B if B > 0 else 0
        edge_weights = mix.repeat_interleave(NE, dim=0)                # (B*NE, K)

        pressure_pred = (p_stack * node_weights).sum(dim=-1)
        flow_pred = (q_stack * edge_weights).sum(dim=-1)

        result: dict[str, torch.Tensor] = {
            "pressure_pred": pressure_pred,
            "flow_pred": flow_pred,
            "router_logits": router_logits,
            "router_probs": router_probs,
        }

        if "pressure_anomaly_logits" in expert_outs[0]:
            pa_stack = torch.stack(
                [o["pressure_anomaly_logits"] for o in expert_outs], dim=-1
            )
            result["pressure_anomaly_logits"] = (pa_stack * node_weights).sum(dim=-1)

        if "flow_anomaly_logits" in expert_outs[0]:
            qa_stack = torch.stack(
                [o["flow_anomaly_logits"] for o in expert_outs], dim=-1
            )
            result["flow_anomaly_logits"] = (qa_stack * edge_weights).sum(dim=-1)

        return result


# ---------------------------------------------------------------------------
# Loss (same shape as moe_loss but works on temporal batches, which are dicts
# rather than PyG Data objects).
# ---------------------------------------------------------------------------

def temporal_moe_loss(
    out: dict[str, torch.Tensor],
    batch: dict,
    lambda_router: float = 0.5,
    lambda_balance: float = 0.01,
    lambda_anomaly: float = 1.0,
    anomaly_pos_weight: float = 5.0,
) -> dict[str, torch.Tensor]:
    """Reconstruction + anomaly + router CE + balance entropy.

    `anomaly_pos_weight` upweights the positive class in the BCE loss.
    Only ~15% of observed sensors are attacked at any time, so plain BCE
    pushes the model to predict "clean" almost always — high precision,
    near-zero recall (especially for replay where the residual is small).
    Setting pos_weight ≈ neg/pos ratio recovers recall.
    """
    pressure_pred = out["pressure_pred"]
    flow_pred = out["flow_pred"]

    # --- Reconstruction ---
    recon_loss = (
        F.mse_loss(pressure_pred, batch["y_pressure"])
        + F.mse_loss(flow_pred, batch["y_flow"])
    )

    # --- Anomaly detection ---
    pos_w = pressure_pred.new_tensor(anomaly_pos_weight)
    anomaly_loss = pressure_pred.new_zeros(())
    obs_p = batch["pressure_mask"] > 0
    if "pressure_anomaly_logits" in out and obs_p.sum() > 0:
        anomaly_loss = anomaly_loss + F.binary_cross_entropy_with_logits(
            out["pressure_anomaly_logits"][obs_p],
            batch["pressure_anomaly"][obs_p],
            pos_weight=pos_w,
        )
    obs_q = batch["flow_mask"] > 0
    if "flow_anomaly_logits" in out and obs_q.sum() > 0:
        anomaly_loss = anomaly_loss + F.binary_cross_entropy_with_logits(
            out["flow_anomaly_logits"][obs_q],
            batch["flow_anomaly"][obs_q],
            pos_weight=pos_w,
        )

    # --- Router CE ---
    router_logits = out["router_logits"]
    targets = batch["attack_type"].long()
    router_ce = F.cross_entropy(router_logits, targets)

    # --- Balance (negative-entropy of batch-mean probs) ---
    mean_probs = F.softmax(router_logits, dim=-1).mean(dim=0)
    balance = (mean_probs * (mean_probs + 1e-10).log()).sum()

    total = (
        recon_loss
        + lambda_anomaly * anomaly_loss
        + lambda_router * router_ce
        + lambda_balance * balance
    )

    return {
        "recon_loss": recon_loss,
        "anomaly_loss": anomaly_loss,
        "router_ce": router_ce,
        "balance": balance,
        "total_loss": total,
    }
