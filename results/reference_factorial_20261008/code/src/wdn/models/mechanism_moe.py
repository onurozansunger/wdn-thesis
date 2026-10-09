"""Observation-only, mechanism-specific extensions of the temporal MoE."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from wdn.models.temporal_moe import TemporalMixtureOfExpertsGNN
from wdn.models.temporal_multitask import TemporalMultiTaskGNN, masked_temporal_statistics


FAMILY_NAMES = ("clean", "random", "replay", "stealthy", "noise", "targeted")
FEATURE_NAMES = ("coverage", "adjacent_coverage", "slope", "last_change",
                 "window_std", "window_range", "autocorr_lag1", "adj_diff_std",
                 "noise_ratio", "spatial_residual", "lag_advantage", "lag_coverage",
                 "self_lag_distance", "self_lag_coverage")
PROFILES = {
    "general": FEATURE_NAMES,
    "abrupt": ("coverage", "adjacent_coverage", "last_change", "window_range", "spatial_residual"),
    "replay": ("coverage", "adjacent_coverage", "autocorr_lag1", "spatial_residual", "lag_advantage", "lag_coverage", "self_lag_distance", "self_lag_coverage"),
    "drift": ("coverage", "slope", "last_change", "window_range", "spatial_residual"),
    "noise": ("coverage", "adjacent_coverage", "window_std", "adj_diff_std", "noise_ratio"),
}

ADVANCED_NAMES = ("last_observed_gap", "last_change_rate", "change_rate_std",
                  "residual_current", "residual_mean", "residual_std", "residual_slope",
                  "residual_shift", "residual_coverage", "sequence_lag_advantage",
                  "sequence_lag_coverage")
PROFILES_V2 = {
    "general": FEATURE_NAMES + ADVANCED_NAMES,
    "abrupt": PROFILES["abrupt"] + ("last_observed_gap", "last_change_rate", "residual_current", "residual_shift", "residual_coverage"),
    "replay": PROFILES["replay"] + ("last_observed_gap", "residual_mean", "residual_coverage", "sequence_lag_advantage", "sequence_lag_coverage"),
    "drift": PROFILES["drift"] + ("last_observed_gap", "last_change_rate", "residual_mean", "residual_slope", "residual_shift", "residual_coverage"),
    "noise": PROFILES["noise"] + ("last_observed_gap", "change_rate_std", "residual_std", "residual_coverage"),
}


def mechanism_features(x_seq, edge_index):
    """Masked temporal and neighbourhood cues; no labels or clean targets.

    Lag advantage compares current sensor data to current versus historical
    neighbour means. It does not assume a replay loses measurement noise.
    Missing history produces neutral values and explicit coverage indicators.
    """
    values = torch.stack([x[:, -2] for x in x_seq])
    masks = torch.stack([x[:, -1] > 0 for x in x_seq])
    weight = masks.to(values.dtype)
    safe_values = torch.where(masks, values, 0.)
    count = weight.sum(0).clamp(min=1)
    stats = masked_temporal_statistics(safe_values, weight)
    times = torch.linspace(-1., 1., len(x_seq), device=values.device)[:, None]
    time_mean = (times * weight).sum(0) / count
    centered_t = times - time_mean
    value_mean = safe_values.sum(0) / count
    slope = (centered_t * (safe_values-value_mean) * weight).sum(0)
    slope = slope / (centered_t.square()*weight).sum(0).clamp(min=1e-6)
    src, dst = edge_index
    neighbour_sum = values.new_zeros(values.shape)
    neighbour_count = values.new_zeros(values.shape)
    neighbour_sum.index_add_(1, dst, safe_values[:, src])
    neighbour_count.index_add_(1, dst, weight[:, src])
    neighbour_mean = neighbour_sum / neighbour_count.clamp(min=1.)
    current_valid = masks[-1] & (neighbour_count[-1] > 0)
    current_residual = (safe_values[-1] - neighbour_mean[-1]).abs()
    spatial_residual = torch.where(current_valid, current_residual, 0.)
    lag_errors, lag_valid, self_errors, self_valid = [], [], [], []
    for lag in range(1, min(7, len(x_seq))):
        valid = current_valid & (neighbour_count[-1-lag] > 0)
        lag_errors.append(torch.where(valid,
            (safe_values[-1]-neighbour_mean[-1-lag]).abs(), float("inf")))
        lag_valid.append(valid)
        observed_pair = masks[-1] & masks[-1-lag]
        self_valid.append(observed_pair)
        self_errors.append(torch.where(observed_pair,
            (safe_values[-1]-safe_values[-1-lag]).abs(), float("inf")))
    if lag_errors:
        supported = torch.stack(lag_valid).any(0)
        minimum = torch.stack(lag_errors).min(0).values
        advantage = torch.where(supported, spatial_residual-minimum, 0.)
        lag_coverage = torch.stack(lag_valid).float().mean(0)
        own_supported = torch.stack(self_valid).any(0)
        own_distance = torch.where(own_supported, torch.stack(self_errors).min(0).values, 0.)
        own_coverage = torch.stack(self_valid).float().mean(0)
    else:
        advantage = torch.zeros_like(spatial_residual)
        lag_coverage = torch.zeros_like(spatial_residual)
        own_distance = torch.zeros_like(spatial_residual)
        own_coverage = torch.zeros_like(spatial_residual)
    return {
        "coverage": weight.mean(0),
        "adjacent_coverage": (masks[1:] & masks[:-1]).float().mean(0)
            if len(x_seq) > 1 else torch.zeros_like(spatial_residual),
        "slope": slope,
        "last_change": stats["temporal_delta"],
        "window_std": stats["window_std"], "window_range": stats["window_range"],
        "autocorr_lag1": stats["autocorr_lag1"], "adj_diff_std": stats["adj_diff_std"],
        "noise_ratio": stats["noise_ratio"], "spatial_residual": spatial_residual,
        "lag_advantage": advantage, "lag_coverage": lag_coverage,
        "self_lag_distance": own_distance, "self_lag_coverage": own_coverage,
    }


def specialist_features(x_seq, edge_index):
    """V2: gap-aware rates and anomalies relative to observed neighbours.

    Neighbour residuals reduce shared diurnal variation without using hidden
    clean values. Lag comparisons use the SAME valid endpoints on both sides,
    require two pairs, and compare sequences instead of one lucky value match.
    All features are causal with respect to the window's last endpoint.
    """
    result = mechanism_features(x_seq, edge_index)
    values = torch.stack([x[:, -2] for x in x_seq])
    valid = torch.stack([x[:, -1] > 0 for x in x_seq])
    values = torch.where(valid, values, 0.)
    T, N = values.shape
    previous_value = values.new_zeros(N)
    previous_time = torch.full((N,), -1, device=values.device, dtype=torch.long)
    rates, pairs = [], []
    last_gap = values.new_zeros(N)
    for t in range(T):
        pair = valid[t] & (previous_time >= 0)
        gap = (t - previous_time).clamp(min=1)
        rate = torch.where(pair, (values[t]-previous_value)/gap, 0.)
        rates.append(rate); pairs.append(pair)
        if t == T-1:
            last_gap = torch.where(pair, gap.to(values.dtype), 0.)
        previous_value = torch.where(valid[t], values[t], previous_value)
        previous_time = torch.where(valid[t], t, previous_time)
    rates, pair_weights = torch.stack(rates), torch.stack(pairs).to(values.dtype)
    count = pair_weights.sum(0).clamp(min=1)
    rate_mean = rates.sum(0)/count
    rate_var = ((rates-rate_mean).square()*pair_weights).sum(0)/count
    rate_std = torch.where(pair_weights.sum(0) >= 2, rate_var.sqrt(), 0.)

    src, dst = edge_index
    sums, counts = values.new_zeros((T, N)), values.new_zeros((T, N))
    sums.index_add_(1, dst, values[:, src])
    counts.index_add_(1, dst, valid[:, src].to(values.dtype))
    neighbours = sums/counts.clamp(min=1)
    residual_valid = valid & (counts > 0)
    rw = residual_valid.to(values.dtype)
    residual = torch.where(residual_valid, values-neighbours, 0.)
    rcount = rw.sum(0).clamp(min=1)
    rmean = residual.sum(0)/rcount
    rvar = ((residual-rmean).square()*rw).sum(0)/rcount
    times = torch.arange(T, device=values.device, dtype=values.dtype)[:, None]
    centered = times-(times*rw).sum(0)/rcount
    rslope = (centered*(residual-rmean)*rw).sum(0)/(centered.square()*rw).sum(0).clamp(min=1e-6)
    half = max(1, T//2)
    first_count, last_count = rw[:half].sum(0), rw[half:].sum(0)
    shift = residual[half:].sum(0)/last_count.clamp(min=1)-residual[:half].sum(0)/first_count.clamp(min=1)
    shift = torch.where((first_count > 0) & (last_count > 0), shift, 0.)
    advantages, supports = [], []
    for lag in range(1, min(7, T)):
        pair = residual_valid[lag:] & (counts[:-lag] > 0)
        weight = pair.to(values.dtype)
        samples = weight.sum(0)
        current_error = (values[lag:]-neighbours[lag:]).abs()
        delayed_error = (values[lag:]-neighbours[:-lag]).abs()
        improvement = ((current_error-delayed_error)*weight).sum(0)/samples.clamp(min=1)
        advantages.append(torch.where(samples >= 2, improvement, float("-inf")))
        supports.append(torch.where(samples >= 2, samples/(T-lag), 0.))
    if advantages:
        stacked = torch.stack(advantages)
        advantage, best = stacked.max(0)
        supported = torch.isfinite(advantage)
        advantage = torch.where(supported, advantage, 0.)
        support = torch.stack(supports).gather(0, best[None]).squeeze(0)
    else:
        advantage, support = values.new_zeros(N), values.new_zeros(N)
    result.update({
        "last_observed_gap": last_gap, "last_change_rate": rates[-1],
        "change_rate_std": rate_std, "residual_current": residual[-1],
        "residual_mean": rmean, "residual_std": torch.where(rw.sum(0) >= 2, rvar.sqrt(), 0.),
        "residual_slope": rslope, "residual_shift": shift, "residual_coverage": rw.mean(0),
        "sequence_lag_advantage": advantage, "sequence_lag_coverage": support,
    })
    return result


@dataclass(frozen=True)
class ExpertSpec:
    name: str
    mechanism: str
    window: int
    hidden_dim: int


class MechanismExpert(TemporalMultiTaskGNN):
    def __init__(self, spec: ExpertSpec, feature_version=1, **kwargs):
        super().__init__(hidden_dim=spec.hidden_dim, window_size=spec.window, **kwargs)
        self.spec = spec
        self.feature_version = feature_version
        self.feature_names = (PROFILES_V2 if feature_version == 2 else PROFILES)[spec.mechanism]
        if feature_version == 2:
            self.embedding_norm = nn.LayerNorm(spec.hidden_dim)
            self.feature_norm = nn.BatchNorm1d(len(self.feature_names))
        self.specialist_head = nn.Sequential(
            nn.Linear(spec.hidden_dim + len(self.feature_names), spec.hidden_dim),
            nn.ReLU(), nn.Linear(spec.hidden_dim, 1),
        )

    def forward(self, x_seq, edge_index, features=None, **kwargs):
        history = x_seq[-self.spec.window:]
        out = super().forward(x_seq=history, edge_index=edge_index, **kwargs)
        if features is None:
            features = (specialist_features if self.feature_version == 2 else mechanism_features)(history, edge_index)
        extra = torch.stack([features[name] for name in self.feature_names], -1)
        embedding = out["node_embeddings"]
        if self.feature_version == 2:
            extra = self.feature_norm(extra.sign()*extra.abs().log1p())
            embedding = self.embedding_norm(embedding)
        out["pressure_anomaly_logits"] = out["pressure_anomaly_logits"] + self.specialist_head(
            torch.cat([embedding, extra], -1)).squeeze(-1)
        return out


class MechanismRouter(nn.Module):
    def __init__(self, hidden_dim, num_experts, temperature=1., uniform=False, feature_version=1):
        super().__init__()
        if temperature <= 0:
            raise ValueError("Router temperature must be positive")
        self.temperature, self.uniform, self.num_experts = temperature, uniform, num_experts
        self.feature_version = feature_version
        self.feature_names = FEATURE_NAMES + ADVANCED_NAMES if feature_version == 2 else FEATURE_NAMES
        self.classifier = nn.Sequential(nn.Linear(2*len(self.feature_names), hidden_dim),
                                        nn.ReLU(), nn.Linear(hidden_dim, num_experts))

    def forward(self, x_seq, edge_index, edge_attr, batch_size, num_nodes_per_graph, features=None):
        if self.uniform:
            return x_seq[0].new_zeros((batch_size, self.num_experts))
        if features is None:
            features = (specialist_features if self.feature_version == 2 else mechanism_features)(x_seq, edge_index)
        nodes = torch.stack([features[name] for name in self.feature_names], -1)
        if self.feature_version == 2:
            nodes = nodes.sign()*nodes.abs().log1p()
        nodes = nodes.reshape(batch_size, num_nodes_per_graph, -1)
        summary = torch.cat([nodes.mean(1), nodes.amax(1)], -1)
        return self.classifier(summary) / self.temperature


class MechanismMoE(TemporalMixtureOfExpertsGNN):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim=32, router_hidden_dim=32,
                 short_window=4, medium_window=8, long_window=16, replay_hidden_dim=None,
                 drift_hidden_dim=None, dropout=.1, num_layers=2,
                 router_temperature=1., uniform=False, feature_version=1):
        if not 1 <= short_window <= medium_window <= long_window:
            raise ValueError("Require 1 <= short <= medium <= long window")
        if feature_version not in (1, 2):
            raise ValueError("feature_version must be 1 or 2")
        super().__init__(node_in_dim=node_in_dim, edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim, router_hidden_dim=router_hidden_dim,
            num_experts=6, num_layers=num_layers, window_size=long_window,
            dropout=dropout, reroute_alpha=0.)
        self.feature_version = feature_version
        self.specs = [
            ExpertSpec("clean", "general", long_window, hidden_dim),
            ExpertSpec("random", "abrupt", short_window, hidden_dim),
            ExpertSpec("replay", "replay", long_window, replay_hidden_dim or hidden_dim),
            ExpertSpec("stealthy", "drift", long_window, drift_hidden_dim or hidden_dim),
            ExpertSpec("noise", "noise", medium_window, hidden_dim),
            ExpertSpec("targeted", "abrupt", short_window, hidden_dim),
        ]
        self.experts = nn.ModuleList([MechanismExpert(spec,
            feature_version=feature_version,
            node_in_dim=node_in_dim, edge_in_dim=edge_in_dim,
            num_layers=num_layers, dropout=dropout) for spec in self.specs])
        self.router = MechanismRouter(router_hidden_dim, 6, router_temperature, uniform, feature_version)

    def forward(self, x_seq, edge_index, edge_attr, is_original_edge, batch_size,
                num_nodes_per_graph, pressure_obs=None, flow_obs=None,
                pressure_mask=None, flow_mask=None):
        if self.feature_version == 1:
            return super().forward(x_seq, edge_index, edge_attr, is_original_edge,
                batch_size, num_nodes_per_graph, pressure_obs, flow_obs, pressure_mask, flow_mask)
        # Reuse deterministic feature extraction between experts with the same
        # window, without sharing their learned parameters or gradients.
        windows = {min(spec.window, len(x_seq)) for spec in self.specs} | {len(x_seq)}
        cache = {window: specialist_features(x_seq[-window:], edge_index) for window in windows}
        router_logits = self.router(x_seq, edge_index, edge_attr, batch_size,
                                     num_nodes_per_graph, features=cache[len(x_seq)])
        probs = router_logits.softmax(-1)
        outputs = [expert(x_seq, edge_index, edge_attr=edge_attr,
            is_original_edge=is_original_edge, pressure_obs=pressure_obs, flow_obs=flow_obs,
            pressure_mask=pressure_mask, flow_mask=flow_mask,
            features=cache[min(expert.spec.window, len(x_seq))]) for expert in self.experts]
        result = {"router_logits": router_logits, "router_probs": probs}
        for key in ("pressure_pred", "flow_pred", "pressure_anomaly_logits", "flow_anomaly_logits"):
            if key not in outputs[0]:
                continue
            stack = torch.stack([out[key] for out in outputs], -1)
            weights = probs.repeat_interleave(stack.shape[0]//batch_size, 0)
            result[key] = (stack*weights).sum(-1)
            result["expert_"+key] = stack
        return result
