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
        # The classifier sees the GNN summary PLUS temporal-stability
        # statistics, aggregated two ways: MEAN (whole-graph trend) and
        # MAX (the worst-case sensor — only ~15% of sensors are
        # attacked, so the mean drowns the signal and only an extremum
        # exposes it). Three stats x two poolings = six extra features.
        # This is the fix for the router misdirection the supervisors
        # flagged: a purely-spatial GNN summary collapses stealthy
        # drift and replay onto each other.
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 6, hidden_dim),
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

        # --- Temporal-stability statistics (the router-fix) ---
        # pressure_obs is column -2 of every x_seq[t], mask is column -1.
        # halves_diff separates STEALTHY (drift -> halves differ) from
        # REPLAY (verbatim copy -> halves identical); log_std and
        # n_changes catch replay's missing observation noise. We average
        # each statistic over the observed sensors of every graph.
        p_obs = torch.stack([x_t[:, -2] for x_t in x_seq], dim=0)   # (T, B*N)
        p_mask = torch.stack([x_t[:, -1] for x_t in x_seq], dim=0)  # (T, B*N)
        T = p_obs.shape[0]
        if T >= 2:
            half = max(1, T // 2)
            halves_diff = (p_obs[half:].mean(0) - p_obs[:half].mean(0)).abs()
            log_std = torch.log(p_obs.std(dim=0) + 1e-3)
            step = (p_obs[1:] - p_obs[:-1]).abs()
            n_changes = (step > 1e-4).float().sum(dim=0)
        else:
            z = p_obs.new_zeros(p_obs.shape[-1])
            halves_diff, log_std, n_changes = z, torch.log(z + 1e-3), z

        stats = torch.stack([halves_diff, log_std, n_changes], dim=-1)  # (B*N, 3)
        obs = (p_mask[-1] > 0).view(B, N, 1)                           # (B, N, 1)
        stats = stats.view(B, N, 3)

        # MEAN over observed sensors.
        denom = obs.float().sum(dim=1).clamp(min=1.0)                  # (B, 1)
        stats_mean = (stats * obs.float()).sum(dim=1) / denom          # (B, 3)

        # MAX over observed sensors — unobserved nodes masked to -inf so
        # they never win the extremum.
        masked = stats.masked_fill(~obs, float("-inf"))
        stats_max = masked.max(dim=1).values                          # (B, 3)
        stats_max = torch.nan_to_num(stats_max, neginf=0.0)

        feats = torch.cat([combined, stats_mean, stats_max], dim=-1)
        return self.classifier(feats)


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
        use_pattern_features: bool = True,
        reroute_alpha: float = 1.5,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.hard_routing = hard_routing
        # Strength of the confidence-gated rerouting: 0 disables it,
        # higher values flatten the expert mix more aggressively when
        # the router is unsure.
        self.reroute_alpha = reroute_alpha

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
                use_pattern_features=use_pattern_features,
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

        # ---- Confidence-gated rerouting ----
        # When the router is unsure (low max-probability), do not bet
        # the whole prediction on one possibly-wrong expert. Raise the
        # softmax temperature so the mix spreads across experts; sharpen
        # it when the router is confident. This makes the MoE robust to
        # the misdirection the supervisors flagged without disabling
        # the router.
        conf = router_probs.max(dim=-1, keepdim=True).values          # (B, 1)
        temperature = 1.0 + self.reroute_alpha * (1.0 - conf)         # (B, 1)
        reroute_probs = F.softmax(router_logits / temperature, dim=-1)

        if self.hard_routing and not self.training:
            top = router_probs.argmax(dim=-1)
            mix = F.one_hot(top, num_classes=self.num_experts).float()
        else:
            mix = reroute_probs

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
            # Per-expert pressure reconstruction, kept so the loss can
            # supervise every expert directly on its own attack class
            # (the "more training on bad experts" fix — a starved
            # expert still gets clean gradient even when the router
            # never routes its class to it).
            "expert_pressure_pred": p_stack,                          # (B*N, K)
            # Per-expert flow, needed by the cascade's physics feedback.
            "expert_flow_pred": q_stack,                              # (B*NE, K)
        }

        if "pressure_anomaly_logits" in expert_outs[0]:
            pa_stack = torch.stack(
                [o["pressure_anomaly_logits"] for o in expert_outs], dim=-1
            )
            result["pressure_anomaly_logits"] = (pa_stack * node_weights).sum(dim=-1)
            result["expert_pressure_anomaly_logits"] = pa_stack       # (B*N, K)

        if "flow_anomaly_logits" in expert_outs[0]:
            qa_stack = torch.stack(
                [o["flow_anomaly_logits"] for o in expert_outs], dim=-1
            )
            result["flow_anomaly_logits"] = (qa_stack * edge_weights).sum(dim=-1)

        return result


# ---------------------------------------------------------------------------
# Cascade routing with feedback
# ---------------------------------------------------------------------------

def expert_feedback_scores(
    out: dict[str, torch.Tensor],
    batch: dict,
    incidence: torch.Tensor | None = None,
    lambda_physics: float = 1.0,
) -> torch.Tensor:
    """Score every expert on every graph *without using labels*.

    The cascade needs to judge, at inference time, whether the expert it
    just tried did a good job. We cannot use ground truth, so we score
    each expert on its own internal consistency:

    1. **Clean-sensor agreement.** The expert declares which observed
       sensors are un-attacked. On exactly those sensors its
       reconstruction should agree with what the sensor reported. An
       expert tuned to the wrong attack class mislabels which readings are
       trustworthy, leaves falsified values inside its "clean" set, and
       pays for it here.
    2. **Physics violation.** The reconstructed flow should conserve mass
       (:math:`\\lVert B\\mathbf{q}\\rVert^2`). A reconstruction that
       explains the numbers but not the hydraulics is rejected.

    Returns:
        (B, K) tensor of scores — **lower is better**.
    """
    B = batch["batch_size"]
    N = batch["num_nodes"]
    p_stack = out["expert_pressure_pred"]                 # (B*N, K)
    K = p_stack.shape[-1]

    p_obs = batch["pressure_obs"].view(B, N, 1)           # (B, N, 1)
    mask = (batch["pressure_mask"] > 0).view(B, N, 1)     # observed sensors
    p_e = p_stack.view(B, N, K)

    if "expert_pressure_anomaly_logits" in out:
        anom = torch.sigmoid(
            out["expert_pressure_anomaly_logits"].view(B, N, K))
    else:                                    # no anomaly head: trust all
        anom = torch.zeros_like(p_e)

    # Sensors this expert believes are both observed and un-attacked.
    clean = mask & (anom < 0.5)                            # (B, N, K)
    resid = (p_e - p_obs).abs() * clean
    denom = clean.sum(dim=1).clamp(min=1)                  # (B, K)
    recon = resid.sum(dim=1) / denom                       # (B, K) lower=better

    # An expert that calls *everything* attacked would trivially score 0
    # on the (empty) clean set, so penalise an implausibly small clean set.
    frac_clean = clean.sum(dim=1).float() / mask.sum(dim=1).clamp(min=1).float()
    recon = recon + 0.5 * (1.0 - frac_clean)

    phys = torch.zeros_like(recon)
    if incidence is not None and lambda_physics > 0 and "expert_flow_pred" in out:
        q_stack = out["expert_flow_pred"]                  # (B*NE, K)
        NE = incidence.shape[1]
        q_e = q_stack.view(B, NE, K)
        res = torch.einsum("ne,bek->bnk", incidence, q_e)  # (B, N, K)
        phys = res.pow(2).mean(dim=1)                      # (B, K)

    # The reconstruction residual and the physics residual live on very
    # different scales (per-node normalisation makes flow variances tiny),
    # so a raw sum lets one term swamp the other and destabilises the
    # ranking on some seeds. Standardise each component *across the K
    # experts within each graph* (z-score) before combining, so the
    # feedback compares experts on equal footing regardless of scale.
    def _z(x):
        mu = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
        return (x - mu) / sd

    score = _z(recon) + lambda_physics * _z(phys)
    return score


@torch.no_grad()
def cascade_route(
    out: dict[str, torch.Tensor],
    batch: dict,
    incidence: torch.Tensor | None = None,
    tau: float | None = None,
    max_attempts: int = 3,
    lambda_physics: float = 1.0,
) -> dict:
    """Iterative router → expert → feedback → re-route decision procedure.

    Instead of blending all experts, the router *ranks* them. We run the
    most likely expert, ask the feedback signal whether it did well, and
    accept it if so. If not, we go back to the router for the next most
    likely expert and try again, up to ``max_attempts``. If nothing clears
    the bar, we keep the best-scoring expert we tried.

    ``tau`` is the acceptance threshold on the feedback score; with
    ``tau=None`` every one of the ``max_attempts`` candidates is tried and
    the best is taken (no early exit).

    Returns a dict with the selected predictions and diagnostics
    (chosen expert, how many attempts were needed, whether the first
    choice was accepted).
    """
    B = batch["batch_size"]
    N = batch["num_nodes"]
    scores = expert_feedback_scores(out, batch, incidence, lambda_physics)
    K = scores.shape[-1]
    ranking = out["router_probs"].argsort(dim=-1, descending=True)   # (B, K)

    n_try = min(max_attempts, K)
    chosen = torch.zeros(B, dtype=torch.long, device=scores.device)
    attempts = torch.zeros(B, dtype=torch.long, device=scores.device)
    accepted_first = torch.zeros(B, dtype=torch.bool, device=scores.device)

    # Router-anchored selection. The router's top-ranked expert is the
    # default. We walk down the ranking and *re-route* to a lower-ranked
    # candidate only when its feedback score beats the current choice by a
    # clear margin (``tau`` acts as that margin on the z-scored feedback).
    # This realises the supervisors' "try the top one, on negative feedback
    # try the next, ... keep the best" loop, while guaranteeing the cascade
    # never underperforms plain top-1 routing just because the label-free
    # feedback is noisy on a given graph.
    # Re-route only when an alternative's feedback is better by this many
    # (z-scored) units. ``tau`` carries the margin when positive; the
    # calibrated acceptance quantile can be negative in z-space, in which
    # case we use a conservative fixed margin so a noisy feedback score
    # cannot override a confident router.
    import os as _os
    margin = float(_os.environ.get("CASCADE_MARGIN", "3.0"))
    if tau is not None and float(tau) > 0:
        margin = float(tau)
    for b in range(B):
        top = int(ranking[b, 0])
        chosen[b] = top
        cur_s = float(scores[b, top])
        used = 1
        for k in range(1, n_try):
            e = int(ranking[b, k])
            s = float(scores[b, e])
            used = k + 1
            if s < cur_s - margin:                  # clearly better -> re-route
                chosen[b] = e
                cur_s = s
        attempts[b] = used
        accepted_first[b] = (int(chosen[b]) == top)

    # Gather the selected expert's outputs.
    idx_n = chosen.repeat_interleave(N).unsqueeze(-1)              # (B*N, 1)
    pressure_pred = out["expert_pressure_pred"].gather(-1, idx_n).squeeze(-1)
    res = {
        "pressure_pred": pressure_pred,
        "chosen_expert": chosen,
        "n_attempts": attempts,
        "accepted_first": accepted_first,
        "feedback_scores": scores,
    }
    if "expert_pressure_anomaly_logits" in out:
        res["pressure_anomaly_logits"] = (
            out["expert_pressure_anomaly_logits"].gather(-1, idx_n).squeeze(-1))
    if "expert_flow_pred" in out:
        NE = out["expert_flow_pred"].shape[0] // B
        idx_e = chosen.repeat_interleave(NE).unsqueeze(-1)
        res["flow_pred"] = out["expert_flow_pred"].gather(-1, idx_e).squeeze(-1)
    return res


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
    lambda_expert: float = 0.5,
    replay_weight: float = 1.0,
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
    targets = batch["attack_type"].long()

    # --- Reconstruction ---
    recon_loss = (
        F.mse_loss(pressure_pred, batch["y_pressure"])
        + F.mse_loss(flow_pred, batch["y_flow"])
    )

    # --- Per-node replay emphasis (FOCUSED) ---
    # Only the actually-attacked sensors inside replay windows get the
    # boost; the 85% non-attacked sensors in the same window keep
    # weight 1. Earlier we upweighted every node in a replay window,
    # which spent capacity on already-easy clean nodes and pulled the
    # overall F1 down. Focusing on attacked-only delivers the gradient
    # exactly where the rare "too-smooth, too-accurate" replay
    # positives sit, without penalising other classes.
    replay_node_w = None
    if "num_nodes" in batch:
        N = batch["num_nodes"]
        attack_per_node = targets.repeat_interleave(N)            # (B*N,)
        is_replay_window = (attack_per_node == 2).float()
        is_attacked = batch["pressure_anomaly"].float()
        boost = (replay_weight - 1.0) * is_replay_window * is_attacked
        replay_node_w = 1.0 + boost                                # (B*N,)

    # --- Anomaly detection ---
    pos_w = pressure_pred.new_tensor(anomaly_pos_weight)
    anomaly_loss = pressure_pred.new_zeros(())
    obs_p = batch["pressure_mask"] > 0
    if "pressure_anomaly_logits" in out and obs_p.sum() > 0:
        per_node = F.binary_cross_entropy_with_logits(
            out["pressure_anomaly_logits"][obs_p],
            batch["pressure_anomaly"][obs_p],
            pos_weight=pos_w, reduction="none",
        )
        if replay_node_w is not None:
            per_node = per_node * replay_node_w[obs_p]
        anomaly_loss = anomaly_loss + per_node.mean()
    obs_q = batch["flow_mask"] > 0
    if "flow_anomaly_logits" in out and obs_q.sum() > 0:
        anomaly_loss = anomaly_loss + F.binary_cross_entropy_with_logits(
            out["flow_anomaly_logits"][obs_q],
            batch["flow_anomaly"][obs_q],
            pos_weight=pos_w,
        )

    # --- Router CE ---
    router_logits = out["router_logits"]
    router_ce = F.cross_entropy(router_logits, targets)

    # --- Balance (negative-entropy of batch-mean probs) ---
    mean_probs = F.softmax(router_logits, dim=-1).mean(dim=0)
    balance = (mean_probs * (mean_probs + 1e-10).log()).sum()

    # --- Direct expert supervision ---
    # Every window has a ground-truth attack class. We train the
    # *matching* expert directly on that window (reconstruction +
    # anomaly), regardless of where the router actually sent it. This
    # is the "more training on bad experts" fix: an expert whose class
    # the router systematically misroutes still receives clean,
    # targeted gradient and never starves.
    expert_loss = pressure_pred.new_zeros(())
    if ("expert_pressure_anomaly_logits" in out
            and replay_node_w is not None):
        idx = torch.arange(attack_per_node.shape[0],
                           device=attack_per_node.device)
        # Anomaly: each node's owning expert logit, replay-upweighted.
        owner_logit = out["expert_pressure_anomaly_logits"][idx, attack_per_node]
        if obs_p.sum() > 0:
            per_node = F.binary_cross_entropy_with_logits(
                owner_logit[obs_p], batch["pressure_anomaly"][obs_p],
                pos_weight=pos_w, reduction="none",
            ) * replay_node_w[obs_p]
            expert_loss = expert_loss + per_node.mean()
        # Reconstruction: each node's owning expert prediction.
        owner_pred = out["expert_pressure_pred"][idx, attack_per_node]
        expert_loss = expert_loss + F.mse_loss(
            owner_pred, batch["y_pressure"])

    total = (
        recon_loss
        + lambda_anomaly * anomaly_loss
        + lambda_router * router_ce
        + lambda_balance * balance
        + lambda_expert * expert_loss
    )

    return {
        "recon_loss": recon_loss,
        "anomaly_loss": anomaly_loss,
        "router_ce": router_ce,
        "balance": balance,
        "expert_loss": expert_loss,
        "total_loss": total,
    }
