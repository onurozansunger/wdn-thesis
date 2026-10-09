"""Small past-only mechanism experts for gradual drift and injected noise."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from wdn.models.family_specific import FAMILY_IDS, _balanced_weights


PROFILES = {
    "drift": ("reference_support", "last_gap", "residual_rate", "dynamic_innovation",
        "dynamic_abs_innovation", "dynamic_cusum_positive", "dynamic_cusum_negative",
        "dynamic_level_shift_8", "dynamic_level_shift_16", "dynamic_innovation_mean_8",
        "dynamic_innovation_mean_16", "seq_innovation", "drift_cusum_positive_0.25",
        "drift_cusum_positive_0.5", "drift_cusum_positive_1.0",
        "drift_cusum_negative_0.25", "drift_cusum_negative_0.5",
        "drift_cusum_negative_1.0", "drift_ramp_strength_3", "drift_mean_strength_3",
        "drift_consistency_5", "drift_ramp_strength_5", "drift_mean_strength_5",
        "drift_consistency_8", "drift_ramp_strength_8", "drift_mean_strength_8",
        "drift_consistency_12", "drift_ramp_strength_12", "drift_mean_strength_12"),
    "noise": ("reference_support", "last_gap", "dynamic_innovation",
        "dynamic_abs_innovation", "dynamic_sigma", "dynamic_past_std_8",
        "dynamic_past_std_16", "dynamic_innovation_rms_8", "dynamic_innovation_rms_16",
        "seq_abs_innovation", "seq_normal_sigma", "noise_state_2.0", "noise_state_4.0",
        "noise_state_9.0", "noise_energy_3", "noise_energy_5", "noise_energy_8",
        "noise_energy_12", "noise_energy_16", "seq_support_3", "seq_support_5",
        "seq_support_8", "seq_support_12", "seq_support_16"),
}


@dataclass(frozen=True)
class CausalSequenceConfig:
    window: int = 12
    hidden_size: int = 20
    embedding_size: int = 24
    epochs: int = 10
    batch_size: int = 512
    learning_rate: float = 0.0015
    weight_decay: float = 0.004
    dropout: float = 0.15
    early_multiplier: float = 2.0
    seed: int = 1907


def causal_row_ids(arrays, window, target_rows=None):
    """Return fixed-hour past windows; -1 denotes a missing observation."""
    if not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a positive integer")
    scenario, node, timestep = (np.asarray(arrays[key])
                                for key in ("scenario", "node", "timestep"))
    if not (len(scenario) == len(node) == len(timestep)):
        raise ValueError("Endpoint keys must align")
    if not all(np.issubdtype(value.dtype, np.integer) for value in (scenario, node, timestep)):
        raise ValueError("Scenario/node/timestep identifiers must be integers")
    if np.any(scenario < 0) or np.any(node < 0):
        raise ValueError("Scenario and node identifiers must be nonnegative")
    rows = (np.arange(len(scenario), dtype=np.int64) if target_rows is None
            else np.asarray(target_rows, dtype=np.int64))
    if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= len(scenario)):
        raise ValueError("target_rows must index endpoint rows")
    # Integer keys plus chunked binary search avoid a Python tuple dictionary,
    # which dominates memory on expanded TRAIN campaigns.
    node_span = int(node.max(initial=0)) + 1
    time_min, time_max = int(timestep.min(initial=0)), int(timestep.max(initial=0))
    time_span = time_max - time_min + 2 * window + 1
    key = ((scenario.astype(np.int64) * node_span + node.astype(np.int64)) * time_span
           + timestep.astype(np.int64) - time_min + window)
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    if np.any(sorted_key[1:] == sorted_key[:-1]):
        raise ValueError("Duplicate scenario/node/timestep endpoint")
    result = np.full((len(rows), window), -1, dtype=np.int64)
    offsets = np.arange(1-window, 1)
    for start in range(0, len(rows), 100_000):
        stop = min(start + 100_000, len(rows))
        query = key[rows[start:stop], None] + offsets
        position = np.searchsorted(sorted_key, query)
        safe = np.minimum(position, max(len(sorted_key) - 1, 0))
        matched = (position < len(sorted_key)) & (sorted_key[safe] == query)
        block = np.full(query.shape, -1, dtype=np.int64)
        block[matched] = order[safe[matched]]
        result[start:stop] = block
    if np.any(result[:, -1] != rows):
        raise ValueError("Every causal sequence must end at its scored row")
    return result


def materialise_sequences(X, row_ids, columns, mean, scale):
    X, row_ids = np.asarray(X), np.asarray(row_ids)
    observed = row_ids >= 0
    safe = np.maximum(row_ids, 0)
    values = np.clip((X[safe][:, :, columns]-mean)/scale, -8., 8.).astype(np.float32)
    values[~observed] = 0.
    return np.concatenate((values, observed[..., None].astype(np.float32)), axis=2)


class _CausalGRUNet(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()
        self.embedding = nn.Sequential(nn.Linear(input_size, config.embedding_size), nn.GELU(),
            nn.LayerNorm(config.embedding_size), nn.Dropout(config.dropout))
        self.gru = nn.GRU(config.embedding_size, config.hidden_size, batch_first=True)
        self.readout = nn.Sequential(nn.Linear(config.hidden_size+config.embedding_size,
            config.hidden_size), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1))

    def forward(self, values):
        embedded = self.embedding(values)
        _, hidden = self.gru(embedded)
        return self.readout(torch.cat((hidden[-1], embedded[:, -1]), dim=1)).squeeze(1)


class CausalSequenceExpert:
    """One unidirectional family expert with deterministic CPU fitting."""
    def __init__(self, names, family, config=CausalSequenceConfig(), seed_offset=0):
        if family not in FAMILY_IDS:
            raise ValueError("family must be drift or noise")
        self.names, self.family, self.family_id = tuple(names), family, FAMILY_IDS[family]
        self.config, self.seed_offset = config, seed_offset
        missing = set(PROFILES[family])-set(names)
        if missing:
            raise ValueError(f"Missing causal sequence features: {sorted(missing)}")
        self.columns = np.asarray([names.index(name) for name in PROFILES[family]], dtype=int)

    def fit(self, arrays):
        X = np.asarray(arrays["X"], dtype=float)
        selected, weights = _balanced_weights(arrays, self.family_id,
            self.config.early_multiplier, self.config.seed+self.seed_offset)
        self.mean_ = X[:, self.columns].mean(axis=0).astype(np.float32)
        self.scale_ = np.maximum(X[:, self.columns].std(axis=0), 1e-3).astype(np.float32)
        rows = causal_row_ids(arrays, self.config.window, selected)
        values = materialise_sequences(X, rows, self.columns, self.mean_, self.scale_)
        labels = ((np.asarray(arrays["families"])[selected] == self.family_id)
                  & (np.asarray(arrays["labels"])[selected] > 0)).astype(np.float32)
        seed = self.config.seed+self.seed_offset
        torch.manual_seed(seed); np.random.seed(seed)
        torch.use_deterministic_algorithms(True); torch.set_num_threads(4)
        self.model_ = _CausalGRUNet(values.shape[2], self.config)
        optimiser = torch.optim.AdamW(self.model_.parameters(), lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        dataset = TensorDataset(torch.from_numpy(values), torch.from_numpy(labels),
            torch.from_numpy(weights.astype(np.float32)))
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(seed), num_workers=0)
        self.model_.train()
        for _ in range(self.config.epochs):
            for batch, target, weight in loader:
                optimiser.zero_grad(set_to_none=True)
                loss = (loss_fn(self.model_(batch), target)*weight).mean()
                loss.backward(); nn.utils.clip_grad_norm_(self.model_.parameters(), 3.)
                optimiser.step()
        self.model_.eval()
        return self

    @torch.inference_mode()
    def predict(self, arrays, batch_size=4096, row_ids=None):
        if not hasattr(self, "model_"):
            raise RuntimeError("CausalSequenceExpert must be fitted first")
        X = np.asarray(arrays["X"], dtype=float)
        rows = (causal_row_ids(arrays, self.config.window) if row_ids is None
                else np.asarray(row_ids, dtype=np.int64))
        if rows.shape != (len(X), self.config.window) or np.any(rows[:, -1] != np.arange(len(X))):
            raise ValueError("row_ids do not match prediction arrays/config window")
        result = []
        for start in range(0, len(rows), batch_size):
            values = materialise_sequences(X, rows[start:start+batch_size], self.columns,
                self.mean_, self.scale_)
            result.append(torch.sigmoid(self.model_(torch.from_numpy(values))).numpy())
        return np.concatenate(result).astype(np.float64)

    def metadata(self):
        return {"architecture": "current-feature skip plus unidirectional causal GRU",
            "family": self.family, "uses_future": False, "uses_true_event_reset": False,
            "features": [self.names[index] for index in self.columns],
            "config": asdict(self.config)}


def causal_score_memory(scores, scenario, timestep, node, half_life):
    """Past-only decaying maximum without event-boundary resets."""
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    scores = np.asarray(scores, dtype=float)
    output = np.empty_like(scores)
    for sid in np.unique(scenario):
        rows = np.flatnonzero(np.asarray(scenario) == sid)
        rows = rows[np.lexsort((np.asarray(node)[rows], np.asarray(timestep)[rows]))]
        state = {}
        for row in rows:
            sensor, time = int(node[row]), int(timestep[row])
            if sensor in state:
                previous_time, previous = state[sensor]
                previous *= 2**(-(time-previous_time)/half_life)
                value = max(scores[row], previous)
            else:
                value = scores[row]
            output[row] = value; state[sensor] = (time, value)
    return output
