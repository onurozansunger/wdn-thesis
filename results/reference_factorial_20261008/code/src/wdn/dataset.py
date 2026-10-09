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

        # Boolean mask: which bidirectional edges are "original" (not reversed)
        # First NE are original, next NE are reversed copies
        is_original = torch.zeros(n_bi, dtype=torch.bool)
        is_original[:NE] = True

        # Graph-level attack type label (for router supervision).
        # Defaults to 0 ("clean") if the loaded CorruptedSnapshot was
        # produced before attack_type_id was tracked.
        attack_id = getattr(corr, "attack_type_id", 0)
        attack_type = torch.tensor([attack_id], dtype=torch.long)

        data = Data(
            x=node_features,
            edge_index=snap.edge_index,
            edge_attr=edge_features,
            edge_map=snap.edge_map,
            is_original_edge=is_original,
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
            # Graph-level attack type label (tensor of shape (1,) so
            # PyG concatenates one per graph during batching)
            attack_type=attack_type,
        )

        return data


# ---------------------------------------------------------------------------
# Normalizer: Z-score normalization fitted on training set
# ---------------------------------------------------------------------------

def _floor_std(std: torch.Tensor, frac: float = 0.1) -> torch.Tensor:
    """Clamp per-sensor standard deviations from below.

    A sensor that never moves carries no temporal information; scaling by
    its (near-zero) std would turn its observation noise into the largest
    signal in the input. Floor at ``frac`` of the median sensor's spread.
    """
    med = std.median()
    floor = torch.clamp(frac * med, min=1e-6)
    return torch.clamp(std, min=float(floor))


class Normalizer:
    """Z-score normalizer for pressure and flow values.

    Fit on training data, then apply to all splits.

    Two modes:

    ``global`` (default)
        One scalar mean/std over every sensor and timestep. Simple, but it
        measures each sensor against the spread of the *whole network*.

    ``per_node``
        A separate mean/std per sensor, computed over time. On networks
        where every sensor sits in a narrow band far from the network
        mean — Modena pressure varies by ~0.01 m within a window while the
        network-wide std is ~5 m — global scaling squashes the within-sensor
        variation to almost nothing. Per-node scaling restores it, which is
        what a replayed (stale) reading has to be detected against.
    """

    def __init__(self, mode: str = "global"):
        assert mode in ("global", "per_node"), f"unknown norm mode {mode!r}"
        self.mode = mode
        # Scalars in "global" mode; (N,) / (NE,) tensors in "per_node" mode.
        self.p_mean: float | torch.Tensor = 0.0
        self.p_std: float | torch.Tensor = 1.0
        self.q_mean: float | torch.Tensor = 0.0
        self.q_std: float | torch.Tensor = 1.0

    def fit(self, snapshots: list[Snapshot]):
        """Compute mean/std from a list of snapshots (use training set only!)."""
        if self.mode == "per_node":
            # (T, S) statistics along time, one per sensor.
            P = torch.stack([s.pressure_true for s in snapshots])
            Q = torch.stack([s.flow_true for s in snapshots])
            self.p_mean = P.mean(dim=0)
            self.q_mean = Q.mean(dim=0)
            # Some sensors are effectively constant (a reservoir at fixed
            # head has std = 0). Dividing by their std would amplify pure
            # noise by orders of magnitude, so floor the scale at a small
            # fraction of the typical sensor's variation.
            self.p_std = _floor_std(P.std(dim=0))
            self.q_std = _floor_std(Q.std(dim=0))
            n_floored_p = int((P.std(dim=0) < self.p_std).sum())
            print(
                f"Normalizer fitted (per-node): "
                f"P std [{self.p_std.min():.4f}, {self.p_std.max():.4f}] "
                f"over {self.p_std.numel()} nodes "
                f"({n_floored_p} near-constant sensors floored), "
                f"Q std [{self.q_std.min():.6f}, {self.q_std.max():.6f}]"
            )
            return

        all_p = torch.cat([s.pressure_true for s in snapshots])
        all_q = torch.cat([s.flow_true for s in snapshots])

        self.p_mean = all_p.mean().item()
        self.p_std = all_p.std().item() + 1e-8
        self.q_mean = all_q.mean().item()
        self.q_std = all_q.std().item() + 1e-8

        print(f"Normalizer fitted: P(mean={self.p_mean:.2f}, std={self.p_std:.2f}), "
              f"Q(mean={self.q_mean:.4f}, std={self.q_std:.4f})")

    @staticmethod
    def _match(stat, x: torch.Tensor) -> torch.Tensor | float:
        """Broadcast a per-sensor statistic onto ``x``.

        Per-node stats have length S (one per sensor), but tensors reach us
        either as a single graph ``(S,)`` or as a flattened batch
        ``(B*S,)``. Tile the stat to match, and move it to x's device.
        """
        if not torch.is_tensor(stat):
            return stat
        stat = stat.to(x.device)
        if x.numel() == stat.numel():
            return stat.view_as(x) if x.shape != stat.shape else stat
        if x.dim() == 1 and x.numel() % stat.numel() == 0:
            return stat.repeat(x.numel() // stat.numel())
        return stat

    def normalize_pressure(self, p: torch.Tensor) -> torch.Tensor:
        return (p - self._match(self.p_mean, p)) / self._match(self.p_std, p)

    def normalize_flow(self, q: torch.Tensor) -> torch.Tensor:
        return (q - self._match(self.q_mean, q)) / self._match(self.q_std, q)

    def denormalize_pressure(self, p: torch.Tensor) -> torch.Tensor:
        return p * self._match(self.p_std, p) + self._match(self.p_mean, p)

    def denormalize_flow(self, q: torch.Tensor) -> torch.Tensor:
        return q * self._match(self.q_std, q) + self._match(self.q_mean, q)

    def state_dict(self) -> dict:
        return {"mode": self.mode,
                "p_mean": self.p_mean, "p_std": self.p_std,
                "q_mean": self.q_mean, "q_std": self.q_std}

    def load_state_dict(self, d: dict):
        self.mode = d.get("mode", "global")
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
