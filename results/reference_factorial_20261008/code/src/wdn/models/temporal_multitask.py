"""TemporalMultiTaskGNN: Spatio-temporal model for state reconstruction and anomaly detection.

Architecture:
    1. Shared encoder: node features -> hidden_dim
    2. Spatial GNN + Temporal GRU: processes T consecutive snapshots
    3. Reconstruction heads: pressure/flow prediction from final hidden state
    4. Anomaly heads: binary classification using embeddings + residuals

The temporal component enables detection of time-dependent attacks
(e.g., replay attacks) that single-snapshot models cannot capture.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from wdn.models.gnn import GNNBackbone, TemporalGNN, MLP


def masked_temporal_statistics(
    values: torch.Tensor,
    masks: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute per-sensor temporal statistics without treating missing as zero.

    Args:
        values: ``(T, N)`` observation tensor. Values at missing positions may
            contain any placeholder.
        masks: ``(T, N)`` tensor where positive entries are observed.

    Returns:
        Nine ``(N,)`` tensors used by the anomaly head. Statistics requiring
        two observations are zero when insufficient history is available.
    """
    valid = masks > 0
    weight = valid.to(values.dtype)
    count = weight.sum(dim=0)
    safe_count = count.clamp(min=1.0)
    mean = (values * weight).sum(dim=0) / safe_count

    centered = values - mean.unsqueeze(0)
    series_var = (centered.square() * weight).sum(dim=0) / safe_count
    has_two = count >= 2
    series_var = torch.where(has_two, series_var, torch.zeros_like(series_var))
    window_std = torch.sqrt(series_var.clamp(min=0.0))
    log_std = torch.where(
        has_two,
        torch.log(window_std + 1e-3),
        torch.zeros_like(window_std),
    )

    pos_inf = torch.full_like(values, float("inf"))
    neg_inf = torch.full_like(values, float("-inf"))
    min_value = torch.where(valid, values, pos_inf).min(dim=0).values
    max_value = torch.where(valid, values, neg_inf).max(dim=0).values
    window_range = torch.where(
        count > 0,
        max_value - min_value,
        torch.zeros_like(max_value),
    )

    T = values.shape[0]
    if T >= 2:
        pair_valid = valid[1:] & valid[:-1]
        pair_weight = pair_valid.to(values.dtype)
        pair_count = pair_weight.sum(dim=0)
        signed_diffs = values[1:] - values[:-1]

        last_pair = valid[-1] & valid[-2]
        temporal_delta = torch.where(
            last_pair,
            signed_diffs[-1].abs(),
            torch.zeros_like(values[-1]),
        )
        n_changes = (
            (signed_diffs.abs() > 1e-4).to(values.dtype) * pair_weight
        ).sum(dim=0)

        safe_pairs = pair_count.clamp(min=1.0)
        diff_mean = (signed_diffs * pair_weight).sum(dim=0) / safe_pairs
        diff_var = (
            (signed_diffs - diff_mean.unsqueeze(0)).square() * pair_weight
        ).sum(dim=0) / safe_pairs
        diff_var = torch.where(
            pair_count >= 2, diff_var, torch.zeros_like(diff_var)
        )
        adj_diff_std = torch.sqrt(diff_var.clamp(min=0.0))

        left = centered[:-1]
        right = centered[1:]
        numerator = (left * right * pair_weight).sum(dim=0)
        left_energy = (left.square() * pair_weight).sum(dim=0)
        right_energy = (right.square() * pair_weight).sum(dim=0)
        denominator = torch.sqrt(left_energy * right_energy).clamp(min=1e-6)
        autocorr_lag1 = torch.where(
            pair_count >= 2,
            numerator / denominator,
            torch.zeros_like(numerator),
        )
        noise_ratio = torch.where(
            has_two & (series_var > 1e-6),
            diff_var / series_var.clamp(min=1e-6),
            torch.zeros_like(series_var),
        )
    else:
        zeros = values.new_zeros(values.shape[1])
        temporal_delta = zeros
        n_changes = zeros
        adj_diff_std = zeros
        autocorr_lag1 = zeros
        noise_ratio = zeros

    half = max(1, T // 2)
    first_values, first_weight = values[:half], weight[:half]
    second_values, second_weight = values[half:], weight[half:]
    first_count = first_weight.sum(dim=0)
    second_count = second_weight.sum(dim=0)
    first_mean = (first_values * first_weight).sum(dim=0) / first_count.clamp(min=1.0)
    second_mean = (
        (second_values * second_weight).sum(dim=0) / second_count.clamp(min=1.0)
        if second_values.shape[0] > 0
        else torch.zeros_like(first_mean)
    )
    halves_diff = torch.where(
        (first_count > 0) & (second_count > 0),
        (second_mean - first_mean).abs(),
        torch.zeros_like(first_mean),
    )

    return {
        "temporal_delta": temporal_delta,
        "window_std": window_std,
        "window_range": window_range,
        "log_std": log_std,
        "halves_diff": halves_diff,
        "n_changes": n_changes,
        "autocorr_lag1": autocorr_lag1,
        "adj_diff_std": adj_diff_std,
        "noise_ratio": noise_ratio,
    }


class TemporalMultiTaskGNN(nn.Module):
    """Spatio-temporal joint reconstruction + anomaly detection model.

    Uses GNN for spatial message passing at each timestep, then GRU
    to capture temporal dynamics across a window of T snapshots.

    Args:
        node_in_dim: Input dimension for node features.
        edge_in_dim: Input dimension for edge features.
        hidden_dim: Hidden layer dimension.
        num_layers: Number of GNN layers.
        num_temporal_layers: Number of GRU layers.
        window_size: Number of consecutive timesteps (T).
        dropout: Dropout rate.
        gnn_type: GNN architecture type.
        heads: Attention heads (for GAT/Transformer).
    """

    def __init__(
        self,
        node_in_dim: int = 7,
        edge_in_dim: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_temporal_layers: int = 1,
        window_size: int = 6,
        dropout: float = 0.1,
        gnn_type: str = "GraphSAGE",
        heads: int = 4,
        use_pattern_features: bool = True,
        use_topology: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        # When False the spatial backbone receives an empty edge set, so
        # GraphSAGE reduces to its root transform and each sensor is
        # modelled independently over time. This is the topology-free
        # baseline: same capacity and same temporal machinery, no message
        # passing, which isolates what the graph itself contributes.
        self.use_topology = use_topology
        # When False, the anomaly head only sees the 6 stability features
        # (delta/std/range/log_std/halves_diff/n_changes) — this matches the
        # pre-pattern baseline used to measure the lift of the new replay
        # signatures (autocorr_lag1, adj_diff_std, noise_ratio).
        self.use_pattern_features = use_pattern_features

        # Shared encoder for each timestep
        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)

        # Spatial backbone
        spatial_backbone = GNNBackbone(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            gnn_type=gnn_type,
            heads=heads,
            edge_dim=edge_in_dim,
        )

        # Spatio-temporal backbone
        self.temporal_gnn = TemporalGNN(
            spatial_backbone=spatial_backbone,
            hidden_dim=hidden_dim,
            num_temporal_layers=num_temporal_layers,
            dropout=dropout,
        )

        # --- Reconstruction heads ---
        self.pressure_head = MLP(hidden_dim, hidden_dim, 1, dropout)
        self.flow_head = MLP(hidden_dim * 2 + edge_in_dim, hidden_dim, 1, dropout)

        # --- Anomaly detection heads ---
        # Pressure head input (4 base + 9 temporal):
        #   [node_embedding, pressure_obs, pressure_pred, |obs - pred|, mask,
        #    temporal_delta, window_std, window_range,
        #    log_window_std, halves_diff, n_changes,
        #    autocorr_lag1, adj_diff_std, noise_ratio]
        # The first 6 temporal features are stability signals; the trailing
        # 3 are explicit replay-pattern signatures. Replayed readings echo
        # past *true* values without observation noise, so the series is
        # smooth: high lag-1 autocorrelation, small diff-std, and low
        # diff/var noise ratio. The three cues are complementary across
        # attack speeds and noise regimes.
        n_temporal = 9 if use_pattern_features else 6
        self.pressure_anomaly_head = MLP(
            hidden_dim + 4 + n_temporal, hidden_dim // 2, 1, dropout,
        )
        # Flow head: same structure, edge-level (no temporal features for
        # flow since the dataset only stores last-timestep flow obs).
        self.flow_anomaly_head = MLP(
            hidden_dim * 2 + 4, hidden_dim // 2, 1, dropout,
        )

    def forward(
        self,
        x_seq: list[torch.Tensor],
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        is_original_edge: torch.Tensor,
        pressure_obs: torch.Tensor | None = None,
        flow_obs: torch.Tensor | None = None,
        pressure_mask: torch.Tensor | None = None,
        flow_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass on a sequence of snapshots.

        Args:
            x_seq: List of T tensors, each (N, node_in_dim).
            edge_index: (2, E) shared graph structure.
            edge_attr: (E, edge_in_dim) shared edge features.
            is_original_edge: (E,) bool mask for original edges.
            pressure_obs: (N,) observed pressures for the LAST timestep.
            flow_obs: (NE,) observed flows for the LAST timestep.
            pressure_mask: (N,) observation mask for the LAST timestep.
            flow_mask: (NE,) observation mask for the LAST timestep.

        Returns:
            Dict with reconstruction and anomaly predictions.
        """
        # 1. Encode each timestep
        encoded_seq = [torch.relu(self.node_encoder(x_t)) for x_t in x_seq]

        # 2. Spatio-temporal encoding (GNN + GRU). The flow head below still
        # uses the real edges; only message passing is disabled.
        gnn_edges, gnn_eattr = edge_index, edge_attr
        if not self.use_topology:
            gnn_edges = edge_index.new_zeros((2, 0))
            gnn_eattr = edge_attr[:0] if edge_attr is not None else None
        h = self.temporal_gnn(encoded_seq, gnn_edges, gnn_eattr)  # (N, hidden)

        # 3. Reconstruction (predict for the last timestep)
        pressure_pred = self.pressure_head(h).squeeze(-1)

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

        # 4. Anomaly detection
        if pressure_obs is not None and pressure_mask is not None:
            p_residual = torch.abs(pressure_obs - pressure_pred).detach()

            # Temporal-stability fingerprints from the observation sequence.
            # The dataset packs (pressure_obs, pressure_mask) as the last
            # two columns of every x_seq[t]. Replay attack: a sensor sends
            # the same recorded value over and over, so window_std and
            # temporal_delta both collapse to ~0 — a signal that no
            # purely-spatial residual can pick up.
            p_obs_seq = torch.stack([x_t[:, -2] for x_t in x_seq], dim=0)   # (T, N)
            p_mask_seq = torch.stack([x_t[:, -1] for x_t in x_seq], dim=0)  # (T, N)

            stats = masked_temporal_statistics(p_obs_seq, p_mask_seq)
            p_temporal_delta = stats["temporal_delta"]
            p_window_std = stats["window_std"]
            p_window_range = stats["window_range"]
            p_log_std = stats["log_std"]
            p_halves_diff = stats["halves_diff"]
            p_n_changes = stats["n_changes"]
            p_autocorr_lag1 = stats["autocorr_lag1"]
            p_adj_diff_std = stats["adj_diff_std"]
            p_noise_ratio = stats["noise_ratio"]

            cols = [
                pressure_obs,
                pressure_pred.detach(),
                p_residual,
                pressure_mask,
                p_temporal_delta.detach(),
                p_window_std.detach(),
                p_window_range.detach(),
                p_log_std.detach(),
                p_halves_diff.detach(),
                p_n_changes.detach(),
            ]
            if self.use_pattern_features:
                cols.extend([
                    p_autocorr_lag1.detach(),
                    p_adj_diff_std.detach(),
                    p_noise_ratio.detach(),
                ])
            p_anomaly_input = torch.stack(cols, dim=-1)
            p_anomaly_input = torch.cat([h, p_anomaly_input], dim=-1)
            p_anomaly_logits = self.pressure_anomaly_head(p_anomaly_input).squeeze(-1)
            result["pressure_anomaly_logits"] = p_anomaly_logits

        if flow_obs is not None and flow_mask is not None:
            q_residual = torch.abs(flow_obs - flow_pred).detach()
            q_anomaly_input = torch.stack([
                flow_obs,
                flow_pred.detach(),
                q_residual,
                flow_mask,
            ], dim=-1)
            q_anomaly_input = torch.cat([src_emb, dst_emb, q_anomaly_input], dim=-1)
            q_anomaly_logits = self.flow_anomaly_head(q_anomaly_input).squeeze(-1)
            result["flow_anomaly_logits"] = q_anomaly_logits

        return result
