"""Small bidirectional trajectory classifier for offline sensor localisation.

The model deliberately consumes observations on both sides of the scored time.  It is
therefore suitable only for retrospective analysis after an inspection window closes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class TrajectoryConfig:
    radius: int = 12
    hidden_size: int = 24
    convolution_channels: int = 32
    epochs: int = 24
    batch_size: int = 256
    learning_rate: float = 0.002
    weight_decay: float = 0.002
    dropout: float = 0.25
    seed: int = 1701


class BidirectionalTrajectoryNet(nn.Module):
    """Convolutional front end followed by a bidirectional GRU."""

    def __init__(self, input_channels: int, config: TrajectoryConfig):
        super().__init__()
        width = config.convolution_channels
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Conv1d(width, width, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
        )
        self.recurrent = nn.GRU(width, config.hidden_size, batch_first=True,
                                bidirectional=True)
        representation = 4 * config.hidden_size
        self.readout = nn.Sequential(
            nn.LayerNorm(representation),
            nn.Dropout(config.dropout),
            nn.Linear(representation, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(values.transpose(1, 2)).transpose(1, 2)
        sequence, _ = self.recurrent(encoded)
        center = sequence[:, sequence.shape[1] // 2]
        pooled = sequence.mean(dim=1)
        return self.readout(torch.cat((center, pooled), dim=1)).squeeze(1)


def _scenario_balanced_weights(labels: np.ndarray, scenarios: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=bool)
    scenarios = np.asarray(scenarios)
    weights = np.zeros(len(labels), dtype=np.float32)
    unique = np.unique(scenarios)
    for scenario in unique:
        local = scenarios == scenario
        for target in (False, True):
            selected = local & (labels == target)
            if not selected.any():
                raise ValueError("Each fitted scenario needs positive and negative localisation rows")
            weights[selected] = 0.5 / (len(unique) * selected.sum())
    weights *= len(weights) / weights.sum()
    return weights


def fit_trajectory_model(train_x: np.ndarray, train_y: np.ndarray,
                         train_scenarios: np.ndarray, config: TrajectoryConfig,
                         *, seed_offset: int = 0) -> BidirectionalTrajectoryNet:
    """Fit one deterministic CPU model with scenario/class balanced loss."""
    train_x = np.asarray(train_x, dtype=np.float32)
    train_y = np.asarray(train_y, dtype=np.float32)
    if train_x.ndim != 3 or train_y.shape != (len(train_x),):
        raise ValueError("Expected [sample,time,channel] trajectories and one label per sample")
    if not np.isfinite(train_x).all() or not np.isfinite(train_y).all():
        raise ValueError("Trajectory training arrays must be finite")
    seed = config.seed + seed_offset
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(4, max(torch.get_num_threads(), 1)))
    model = BidirectionalTrajectoryNet(train_x.shape[2], config)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    weights = _scenario_balanced_weights(train_y > 0, train_scenarios)
    dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y),
                            torch.from_numpy(weights))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                        generator=generator, num_workers=0)
    model.train()
    for _ in range(config.epochs):
        for values, labels, sample_weights in loader:
            optimiser.zero_grad(set_to_none=True)
            loss = (criterion(model(values), labels) * sample_weights).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
    model.eval()
    return model


@torch.inference_mode()
def predict_trajectory_model(model: BidirectionalTrajectoryNet, values: np.ndarray,
                             batch_size: int = 512) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("Expected finite [sample,time,channel] trajectories")
    loader = DataLoader(TensorDataset(torch.from_numpy(values)), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    return np.concatenate([torch.sigmoid(model(batch[0])).cpu().numpy()
                           for batch in loader]).astype(np.float64)


def model_metadata(config: TrajectoryConfig) -> dict:
    return {"architecture": "two-layer temporal convolution plus bidirectional GRU",
            "retrospective": True, "uses_future_observations": True,
            "config": asdict(config)}
