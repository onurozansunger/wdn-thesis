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
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.window_size = window_size
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

        # 2. Spatio-temporal encoding (GNN + GRU)
        h = self.temporal_gnn(encoded_seq, edge_index, edge_attr)  # (N, hidden_dim)

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

            T = p_obs_seq.shape[0]
            if T >= 2:
                # delta vs previous timestep, only meaningful when both
                # endpoints were observed (otherwise mask -> 0).
                both_obs = p_mask_seq[-1] * p_mask_seq[-2]
                p_temporal_delta = (p_obs_seq[-1] - p_obs_seq[-2]).abs() * both_obs
                # spread of the value across the whole window.
                p_window_std = p_obs_seq.std(dim=0)
                p_window_range = (p_obs_seq.max(dim=0).values
                                  - p_obs_seq.min(dim=0).values)
                # log_std: explodes negative when std~0, saturates positive
                # for normal noise. This is the strongest replay signal
                # because the corruption pipeline writes replayed values
                # *without* the Gaussian noise it adds to clean readings.
                p_log_std = torch.log(p_window_std + 1e-3)
                # First half vs second half mean — a stealthy drift
                # signature (random/replay leave both halves identical).
                half = max(1, T // 2)
                first_half_mean = p_obs_seq[:half].mean(dim=0)
                second_half_mean = p_obs_seq[half:].mean(dim=0)
                p_halves_diff = (second_half_mean - first_half_mean).abs()
                # Number of "real" step changes inside the window. Replay
                # gives ~0 because all entries are identical; clean noisy
                # readings give T-1 because each step differs by noise.
                step_diffs = (p_obs_seq[1:] - p_obs_seq[:-1]).abs()
                p_n_changes = (step_diffs > 1e-4).float().sum(dim=0)

                # --- Pattern-detection signatures targeting replay ---
                # The corruption pipeline replays *previous true* values
                # without the Gaussian observation noise, so a replayed
                # series is smooth (it follows the slow hydraulic dynamic)
                # while a clean series carries independent noise on top.
                # The four features below all measure that "missing noise".
                centered = p_obs_seq - p_obs_seq.mean(dim=0, keepdim=True)
                # 1. Lag-1 autocorrelation. Clean noisy readings -> ~0
                #    because consecutive samples are independent. Replay ->
                #    close to +1 because the underlying signal is smooth.
                num = (centered[1:] * centered[:-1]).sum(dim=0)
                den = (centered * centered).sum(dim=0) + 1e-6
                p_autocorr_lag1 = num / den
                # 2. Std of consecutive differences. Clean readings have
                #    diff std ~= sqrt(2) * sigma_noise; replay readings
                #    have diff std governed only by the small hydraulic
                #    step, which is typically much smaller.
                step_diffs_signed = p_obs_seq[1:] - p_obs_seq[:-1]
                p_adj_diff_std = step_diffs_signed.std(dim=0)
                # 3. Noise-to-signal ratio: var(differences) / var(series).
                #    For an i.i.d. noise process this is ~2; for a smooth
                #    series (replay) it collapses toward 0.
                series_var = p_obs_seq.var(dim=0) + 1e-6
                diff_var = step_diffs_signed.var(dim=0)
                p_noise_ratio = diff_var / series_var
            else:
                zeros = pressure_obs.new_zeros(pressure_obs.shape)
                p_temporal_delta = zeros
                p_window_std = zeros
                p_window_range = zeros
                p_log_std = torch.log(zeros + 1e-3)
                p_halves_diff = zeros
                p_n_changes = zeros
                p_autocorr_lag1 = zeros
                p_adj_diff_std = zeros
                p_noise_ratio = zeros

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
