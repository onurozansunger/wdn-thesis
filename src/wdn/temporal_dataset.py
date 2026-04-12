"""Temporal dataset for WDN snapshots.

Groups consecutive snapshots from the same scenario into sliding windows
of size T for spatio-temporal modeling (GNN + GRU).
"""

from __future__ import annotations

from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import Dataset

from wdn.data_generation import Snapshot
from wdn.corruption import CorruptedSnapshot
from wdn.dataset import Normalizer


class TemporalWDNDataset(Dataset):
    """Dataset that returns windows of T consecutive snapshots.

    Each item contains:
        - x_seq: list of T node feature tensors (N, F)
        - edge_index, edge_attr: shared graph structure
        - targets for the LAST timestep (pressure, flow, anomaly labels)

    Snapshots must be ordered by (scenario_id, timestep).
    """

    def __init__(
        self,
        snapshots: list[Snapshot],
        corrupted: list[CorruptedSnapshot],
        window_size: int = 6,
        normalizer: Normalizer | None = None,
    ):
        assert len(snapshots) == len(corrupted)
        self.snapshots = snapshots
        self.corrupted = corrupted
        self.window_size = window_size
        self.normalizer = normalizer

        # Group snapshots by scenario_id
        scenario_groups: dict[int, list[int]] = defaultdict(list)
        for i, snap in enumerate(snapshots):
            scenario_groups[snap.scenario_id].append(i)

        # Sort each group by timestep
        for sid in scenario_groups:
            scenario_groups[sid].sort(key=lambda i: snapshots[i].timestep)

        # Build valid windows: each window is a list of T consecutive indices
        self.windows: list[list[int]] = []
        for sid in sorted(scenario_groups.keys()):
            indices = scenario_groups[sid]
            if len(indices) < window_size:
                continue
            for start in range(len(indices) - window_size + 1):
                self.windows.append(indices[start:start + window_size])

        if len(self.windows) == 0:
            raise ValueError(
                f"No valid windows of size {window_size}. "
                f"Scenarios may have fewer than {window_size} timesteps."
            )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        window_indices = self.windows[idx]
        T = len(window_indices)

        # Build node features for each timestep
        x_seq = []
        for t_idx in window_indices:
            snap = self.snapshots[t_idx]
            corr = self.corrupted[t_idx]

            p_obs = corr.pressure_obs.clone()
            if self.normalizer is not None:
                p_obs = self.normalizer.normalize_pressure(p_obs) * corr.pressure_mask

            node_features = torch.cat([
                snap.node_static,                    # (N, 5)
                p_obs.unsqueeze(-1),                 # (N, 1)
                corr.pressure_mask.unsqueeze(-1),    # (N, 1)
            ], dim=-1)  # (N, 7)
            x_seq.append(node_features)

        # Edge features and targets come from the LAST timestep
        last_idx = window_indices[-1]
        last_snap = self.snapshots[last_idx]
        last_corr = self.corrupted[last_idx]

        # Normalize targets
        p_true = last_snap.pressure_true
        q_true = last_snap.flow_true
        q_obs = last_corr.flow_obs.clone()

        if self.normalizer is not None:
            p_true = self.normalizer.normalize_pressure(p_true)
            q_true = self.normalizer.normalize_flow(q_true)
            q_obs = self.normalizer.normalize_flow(q_obs) * last_corr.flow_mask

        p_obs_last = x_seq[-1][:, 5]  # pressure_obs from last timestep features

        # Edge features (bidirectional)
        NE = last_snap.flow_true.shape[0]
        n_bi = last_snap.edge_index.shape[1]

        edge_static_bi = last_snap.edge_static[last_snap.edge_map]
        q_obs_bi = q_obs[last_snap.edge_map]
        q_mask_bi = last_corr.flow_mask[last_snap.edge_map]

        edge_features = torch.cat([
            edge_static_bi,
            q_obs_bi.unsqueeze(-1),
            q_mask_bi.unsqueeze(-1),
        ], dim=-1)

        is_original = torch.zeros(n_bi, dtype=torch.bool)
        is_original[:NE] = True

        return {
            "x_seq": x_seq,                         # list of T × (N, 7)
            "edge_index": last_snap.edge_index,      # (2, 2*NE)
            "edge_attr": edge_features,               # (2*NE, 8)
            "is_original_edge": is_original,           # (2*NE,)
            "y_pressure": p_true,                      # (N,)
            "y_flow": q_true,                          # (NE,)
            "pressure_mask": last_corr.pressure_mask,  # (N,)
            "flow_mask": last_corr.flow_mask,          # (NE,)
            "pressure_obs": p_obs_last,                # (N,)
            "flow_obs": q_obs,                         # (NE,)
            "pressure_anomaly": last_corr.pressure_anomaly,  # (N,)
            "flow_anomaly": last_corr.flow_anomaly,          # (NE,)
        }


def temporal_collate_fn(batch: list[dict]) -> dict:
    """Custom collate for temporal batches.

    Since all snapshots share the same graph structure (same network),
    we can batch by stacking node features across the batch dimension.
    """
    T = len(batch[0]["x_seq"])
    B = len(batch)

    # Stack x_seq: for each timestep, stack across batch
    x_seq = []
    for t in range(T):
        x_t = torch.cat([b["x_seq"][t] for b in batch], dim=0)  # (B*N, 7)
        x_seq.append(x_t)

    N = batch[0]["x_seq"][0].shape[0]  # nodes per graph

    # Build batched edge_index (offset by N for each graph)
    edge_indices = []
    edge_attrs = []
    is_orig = []
    for i, b in enumerate(batch):
        edge_indices.append(b["edge_index"] + i * N)
        edge_attrs.append(b["edge_attr"])
        is_orig.append(b["is_original_edge"])

    return {
        "x_seq": x_seq,
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_attr": torch.cat(edge_attrs, dim=0),
        "is_original_edge": torch.cat(is_orig, dim=0),
        "y_pressure": torch.cat([b["y_pressure"] for b in batch], dim=0),
        "y_flow": torch.cat([b["y_flow"] for b in batch], dim=0),
        "pressure_mask": torch.cat([b["pressure_mask"] for b in batch], dim=0),
        "flow_mask": torch.cat([b["flow_mask"] for b in batch], dim=0),
        "pressure_obs": torch.cat([b["pressure_obs"] for b in batch], dim=0),
        "flow_obs": torch.cat([b["flow_obs"] for b in batch], dim=0),
        "pressure_anomaly": torch.cat([b["pressure_anomaly"] for b in batch], dim=0),
        "flow_anomaly": torch.cat([b["flow_anomaly"] for b in batch], dim=0),
        "batch_size": B,
        "num_nodes": N,
    }


def create_temporal_dataloaders(
    train_snaps, train_corr,
    val_snaps, val_corr,
    test_snaps, test_corr,
    window_size: int = 6,
    batch_size: int = 8,
    num_workers: int = 0,
):
    """Create temporal DataLoaders with normalization.

    Returns:
        (train_loader, val_loader, test_loader, normalizer)
    """
    from torch.utils.data import DataLoader

    normalizer = Normalizer()
    normalizer.fit(train_snaps)

    train_ds = TemporalWDNDataset(train_snaps, train_corr, window_size, normalizer)
    val_ds = TemporalWDNDataset(val_snaps, val_corr, window_size, normalizer)
    test_ds = TemporalWDNDataset(test_snaps, test_corr, window_size, normalizer)

    print(f"Temporal DataLoaders (window={window_size}): "
          f"train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=temporal_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=temporal_collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=temporal_collate_fn,
    )

    return train_loader, val_loader, test_loader, normalizer
