"""PyTorch Dataset for WDN snapshots.

Combines clean Snapshots with CorruptedSnapshots into model-ready
PyTorch Geometric Data objects.
"""

from __future__ import annotations

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data

from wdn.data_generation import Snapshot, WDNGraph
from wdn.corruption import CorruptedSnapshot


class WDNDataset(Dataset):
    """Dataset of corrupted WDN snapshots.

    Each item is a PyTorch Geometric `Data` object containing:
        - x: (N, F) node features [static | pressure_obs | pressure_mask]
        - edge_index: (2, 2*NE) bidirectional connectivity
        - edge_attr: (2*NE, F_edge) edge features [static | flow_obs | flow_mask]
        - edge_map: (2*NE,) mapping from bidirectional to original edges
        - y_pressure: (N,) ground truth pressure
        - y_flow: (NE,) ground truth flow
        - pressure_mask: (N,) observation mask
        - flow_mask: (NE,) observation mask
        - pressure_anomaly: (N,) anomaly labels
        - flow_anomaly: (NE,) anomaly labels
    """

    def __init__(
        self,
        snapshots: list[Snapshot],
        corrupted: list[CorruptedSnapshot],
        normalizer: Normalizer | None = None,
    ):
        assert len(snapshots) == len(corrupted)
        self.snapshots = snapshots
        self.corrupted = corrupted
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.snapshots)

    def __getitem__(self, idx: int) -> Data:
        snap = self.snapshots[idx]
        corr = self.corrupted[idx]

        # Normalize ground truth and observations if normalizer is available
        p_true = snap.pressure_true
        q_true = snap.flow_true
        p_obs = corr.pressure_obs.clone()
        q_obs = corr.flow_obs.clone()

        if self.normalizer is not None:
            p_true = self.normalizer.normalize_pressure(p_true)
            q_true = self.normalizer.normalize_flow(q_true)
            # Only normalize observed values (masked ones are already 0)
            p_obs = self.normalizer.normalize_pressure(p_obs) * corr.pressure_mask
            q_obs = self.normalizer.normalize_flow(q_obs) * corr.flow_mask

        # --- Node features: [static(5) | pressure_obs(1) | pressure_mask(1)] ---
        node_features = torch.cat([
            snap.node_static,                    # (N, 5)
            p_obs.unsqueeze(-1),                 # (N, 1)
            corr.pressure_mask.unsqueeze(-1),    # (N, 1)
        ], dim=-1)  # (N, 7)

        # --- Edge features for bidirectional edges ---
        # For original edges: use the actual flow obs and mask
        # For reversed edges: use the same values (flow is undirected for the model)
        NE = snap.flow_true.shape[0]
        n_bi = snap.edge_index.shape[1]  # 2 * NE

        # Expand edge static features to bidirectional
        edge_static_bi = snap.edge_static[snap.edge_map]  # (2*NE, F_edge)

        # Expand flow obs and mask to bidirectional
        q_obs_bi = q_obs[snap.edge_map]                      # (2*NE,)
        q_mask_bi = corr.flow_mask[snap.edge_map]             # (2*NE,)

        edge_features = torch.cat([
            edge_static_bi,                      # (2*NE, 6)
            q_obs_bi.unsqueeze(-1),              # (2*NE, 1)
            q_mask_bi.unsqueeze(-1),             # (2*NE, 1)
        ], dim=-1)  # (2*NE, 8)

        data = Data(
            x=node_features,
            edge_index=snap.edge_index,
            edge_attr=edge_features,
            edge_map=snap.edge_map,
            # Targets
            y_pressure=p_true,
            y_flow=q_true,
            # Masks
            pressure_mask=corr.pressure_mask,
            flow_mask=corr.flow_mask,
            pressure_obs=p_obs,
            flow_obs=q_obs,
            # Anomaly labels
            pressure_anomaly=corr.pressure_anomaly,
            flow_anomaly=corr.flow_anomaly,
            # Metadata
            num_original_edges=torch.tensor(NE, dtype=torch.long),
        )

        return data


# ---------------------------------------------------------------------------
# Normalizer: Z-score normalization fitted on training set
# ---------------------------------------------------------------------------

class Normalizer:
    """Z-score normalizer for pressure and flow values.

    Fit on training data, then apply to all splits.
    """

    def __init__(self):
        self.p_mean: float = 0.0
        self.p_std: float = 1.0
        self.q_mean: float = 0.0
        self.q_std: float = 1.0

    def fit(self, snapshots: list[Snapshot]):
        """Compute mean/std from a list of snapshots (use training set only!)."""
        all_p = torch.cat([s.pressure_true for s in snapshots])
        all_q = torch.cat([s.flow_true for s in snapshots])

        self.p_mean = all_p.mean().item()
        self.p_std = all_p.std().item() + 1e-8
        self.q_mean = all_q.mean().item()
        self.q_std = all_q.std().item() + 1e-8

        print(f"Normalizer fitted: P(mean={self.p_mean:.2f}, std={self.p_std:.2f}), "
              f"Q(mean={self.q_mean:.4f}, std={self.q_std:.4f})")

    def normalize_pressure(self, p: torch.Tensor) -> torch.Tensor:
        return (p - self.p_mean) / self.p_std

    def normalize_flow(self, q: torch.Tensor) -> torch.Tensor:
        return (q - self.q_mean) / self.q_std

    def denormalize_pressure(self, p: torch.Tensor) -> torch.Tensor:
        return p * self.p_std + self.p_mean

    def denormalize_flow(self, q: torch.Tensor) -> torch.Tensor:
        return q * self.q_std + self.q_mean

    def state_dict(self) -> dict:
        return {"p_mean": self.p_mean, "p_std": self.p_std,
                "q_mean": self.q_mean, "q_std": self.q_std}

    def load_state_dict(self, d: dict):
        self.p_mean = d["p_mean"]
        self.p_std = d["p_std"]
        self.q_mean = d["q_mean"]
        self.q_std = d["q_std"]


# ---------------------------------------------------------------------------
# Helpers: split & create DataLoaders
# ---------------------------------------------------------------------------

def train_val_test_split(
    snapshots: list[Snapshot],
    corrupted: list[CorruptedSnapshot],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list, list, list, list, list, list]:
    """Split snapshots into train/val/test sets.

    Returns:
        (train_snaps, train_corr, val_snaps, val_corr, test_snaps, test_corr)
    """
    n = len(snapshots)
    indices = np.random.default_rng(seed).permutation(n)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    def _select(lst, idx):
        return [lst[i] for i in idx]

    return (
        _select(snapshots, train_idx), _select(corrupted, train_idx),
        _select(snapshots, val_idx), _select(corrupted, val_idx),
        _select(snapshots, test_idx), _select(corrupted, test_idx),
    )


def create_dataloaders(
    train_snaps, train_corr,
    val_snaps, val_corr,
    test_snaps, test_corr,
    batch_size: int = 8,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, Normalizer]:
    """Create DataLoaders with normalization fitted on training set.

    Returns:
        (train_loader, val_loader, test_loader, normalizer)
    """
    from torch_geometric.loader import DataLoader as PyGDataLoader

    # Fit normalizer on training data only
    normalizer = Normalizer()
    normalizer.fit(train_snaps)

    train_ds = WDNDataset(train_snaps, train_corr, normalizer)
    val_ds = WDNDataset(val_snaps, val_corr, normalizer)
    test_ds = WDNDataset(test_snaps, test_corr, normalizer)

    train_loader = PyGDataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = PyGDataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = PyGDataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"DataLoaders: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    return train_loader, val_loader, test_loader, normalizer
