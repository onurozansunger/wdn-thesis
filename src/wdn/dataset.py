from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from wdn.config import SplitConfig
from wdn.normalization import Normalizer


@dataclass
class DatasetSplits:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def load_npz(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def split_indices(num_samples: int, split: SplitConfig, seed: int) -> DatasetSplits:
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    if split.shuffle:
        rng.shuffle(indices)
    n_train = int(num_samples * split.train)
    n_val = int(num_samples * split.val)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return DatasetSplits(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


class WDNDataset(Dataset):
    def __init__(
        self,
        data: Dict[str, np.ndarray],
        indices: np.ndarray,
        normalizer: Optional[Normalizer] = None,
    ) -> None:
        self.data = data
        self.indices = indices
        self.normalizer = normalizer

        self.edge_index = torch.as_tensor(data["edge_index"], dtype=torch.long)
        self.node_static = torch.as_tensor(data["node_static"], dtype=torch.float32)
        self.edge_static = torch.as_tensor(data["edge_static"], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i = self.indices[idx]
        p_true = self.data["P_true"][i]
        q_true = self.data["Q_true"][i]
        p_obs = self.data["P_obs"][i]
        q_obs = self.data["Q_obs"][i]
        p_mask = self.data["P_mask"][i]
        q_mask = self.data["Q_mask"][i]
        p_anom = self.data["P_anom"][i]
        q_anom = self.data["Q_anom"][i]

        p_obs_filled = np.nan_to_num(p_obs, nan=0.0)
        q_obs_filled = np.nan_to_num(q_obs, nan=0.0)

        if self.normalizer is not None:
            p_obs_filled = self.normalizer.transform_p(p_obs_filled)
            q_obs_filled = self.normalizer.transform_q(q_obs_filled)
            p_true = self.normalizer.transform_p(p_true)
            q_true = self.normalizer.transform_q(q_true)
            p_obs_filled[~p_mask] = 0.0
            q_obs_filled[~q_mask] = 0.0

        sample = {
            "edge_index": self.edge_index,
            "node_static": self.node_static,
            "edge_static": self.edge_static,
            "P_obs": torch.as_tensor(p_obs_filled, dtype=torch.float32),
            "Q_obs": torch.as_tensor(q_obs_filled, dtype=torch.float32),
            "P_mask": torch.as_tensor(p_mask, dtype=torch.bool),
            "Q_mask": torch.as_tensor(q_mask, dtype=torch.bool),
            "P_true": torch.as_tensor(p_true, dtype=torch.float32),
            "Q_true": torch.as_tensor(q_true, dtype=torch.float32),
            "P_anom": torch.as_tensor(p_anom, dtype=torch.float32),
            "Q_anom": torch.as_tensor(q_anom, dtype=torch.float32),
        }
        return sample


def create_dataloaders(
    data: Dict[str, np.ndarray],
    splits: DatasetSplits,
    batch_size: int,
    num_workers: int,
    normalizer: Optional[Normalizer] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = WDNDataset(data, splits.train_idx, normalizer)
    val_ds = WDNDataset(data, splits.val_idx, normalizer)
    test_ds = WDNDataset(data, splits.test_idx, normalizer)

    def _collate(batch):
        return batch

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate
    )
    return train_loader, val_loader, test_loader
