from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np


@dataclass
class Normalizer:
    p_mean: float
    p_std: float
    q_mean: float
    q_std: float

    def transform_p(self, values: np.ndarray) -> np.ndarray:
        return (values - self.p_mean) / (self.p_std + 1e-8)

    def transform_q(self, values: np.ndarray) -> np.ndarray:
        return (values - self.q_mean) / (self.q_std + 1e-8)

    def inverse_p(self, values: np.ndarray) -> np.ndarray:
        return values * (self.p_std + 1e-8) + self.p_mean

    def inverse_q(self, values: np.ndarray) -> np.ndarray:
        return values * (self.q_std + 1e-8) + self.q_mean

    def to_dict(self) -> Dict[str, float]:
        return {
            "p_mean": float(self.p_mean),
            "p_std": float(self.p_std),
            "q_mean": float(self.q_mean),
            "q_std": float(self.q_std),
        }

    @staticmethod
    def from_dict(data: Dict[str, float]) -> "Normalizer":
        return Normalizer(
            p_mean=float(data["p_mean"]),
            p_std=float(data["p_std"]),
            q_mean=float(data["q_mean"]),
            q_std=float(data["q_std"]),
        )


def compute_normalizer(p_true: np.ndarray, q_true: np.ndarray) -> Normalizer:
    return Normalizer(
        p_mean=float(np.mean(p_true)),
        p_std=float(np.std(p_true)),
        q_mean=float(np.mean(q_true)),
        q_std=float(np.std(q_true)),
    )


def save_normalizer(normalizer: Normalizer, path: Path) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(normalizer.to_dict(), f, sort_keys=False)


def load_normalizer(path: Path) -> Normalizer:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Normalizer.from_dict(data)
