"""Pre-compute demo data: run model on test snapshots and save results.

Run once before launching the dashboard:
    python dashboard/precompute/export_demo_data.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wdn.dataset import WDNDataset, Normalizer, train_val_test_split
from wdn.models.multitask import MultiTaskGNN


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    device = get_device()
    print(f"Device: {device}")

    # Load data
    data_dir = PROJECT_ROOT / "data" / "generated_attacks"
    with open(data_dir / "graph.pkl", "rb") as f:
        graph = pickle.load(f)
    with open(data_dir / "snapshots.pkl", "rb") as f:
        snapshots = pickle.load(f)
    with open(data_dir / "corrupted.pkl", "rb") as f:
        corrupted = pickle.load(f)

    print(f"Loaded {len(snapshots)} snapshots ({graph.num_nodes} nodes, {graph.num_edges} edges)")

    # Split (same as training!)
    train_s, train_c, val_s, val_c, test_s, test_c = train_val_test_split(
        snapshots, corrupted, 0.70, 0.15, seed=42,
    )

    # Normalizer
    normalizer = Normalizer()
    norm_path = PROJECT_ROOT / "runs" / "multitask" / "20260310_201113" / "normalizer.pt"
    norm_state = torch.load(norm_path, map_location="cpu", weights_only=True)
    normalizer.load_state_dict(norm_state)
    print(f"Normalizer: P(mean={normalizer.p_mean:.2f}, std={normalizer.p_std:.2f})")

    # Model
    sample_ds = WDNDataset(test_s[:1], test_c[:1], normalizer)
    sample = sample_ds[0]

    model = MultiTaskGNN(
        node_in_dim=sample.x.shape[1],
        edge_in_dim=sample.edge_attr.shape[1],
        hidden_dim=64, num_layers=2, dropout=0.1,
        gnn_type="GraphSAGE", heads=4,
    ).to(device)

    state = torch.load(
        PROJECT_ROOT / "runs" / "multitask" / "20260310_201113" / "best_model.pt",
        map_location=device, weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()
    print("Model loaded")

    # Pick 20 evenly-spaced test snapshots
    n_demo = min(20, len(test_s))
    indices = np.linspace(0, len(test_s) - 1, n_demo, dtype=int)

    demo_data = []
    ds = WDNDataset(test_s, test_c, normalizer)

    with torch.no_grad():
        for idx in indices:
            data = ds[idx]
            data_dev = data.to(device)

            out = model(
                x=data_dev.x.unsqueeze(0) if data_dev.x.dim() == 1 else data_dev.x,
                edge_index=data_dev.edge_index,
                edge_attr=data_dev.edge_attr,
                is_original_edge=data_dev.is_original_edge,
                pressure_obs=data_dev.pressure_obs,
                flow_obs=data_dev.flow_obs,
                pressure_mask=data_dev.pressure_mask,
                flow_mask=data_dev.flow_mask,
            )

            # Denormalize (ensure CPU)
            p_pred = normalizer.denormalize_pressure(out["pressure_pred"].cpu()).numpy()
            p_true = normalizer.denormalize_pressure(data.y_pressure.cpu()).numpy()
            p_obs_raw = test_s[idx].pressure_true.numpy()  # raw true values
            q_pred = normalizer.denormalize_flow(out["flow_pred"].cpu()).numpy()
            q_true = normalizer.denormalize_flow(data.y_flow.cpu()).numpy()

            p_mask = data.pressure_mask.cpu().numpy()
            q_mask = data.flow_mask.cpu().numpy()

            # Anomaly
            p_anom_true = data.pressure_anomaly.cpu().numpy()
            q_anom_true = data.flow_anomaly.cpu().numpy()
            p_anom_prob = torch.sigmoid(out["pressure_anomaly_logits"]).cpu().numpy() if "pressure_anomaly_logits" in out else np.zeros_like(p_mask)
            q_anom_prob = torch.sigmoid(out["flow_anomaly_logits"]).cpu().numpy() if "flow_anomaly_logits" in out else np.zeros_like(q_mask)

            # Observed values (in original scale, 0 where missing)
            p_obs_display = p_obs_raw * p_mask

            entry = {
                "index": int(idx),
                "pressure_true": p_true.tolist(),
                "pressure_pred": p_pred.tolist(),
                "pressure_obs": p_obs_display.tolist(),
                "pressure_mask": p_mask.tolist(),
                "pressure_error": np.abs(p_pred - p_true).tolist(),
                "flow_true": q_true.tolist(),
                "flow_pred": q_pred.tolist(),
                "flow_mask": q_mask.tolist(),
                "flow_error": np.abs(q_pred - q_true).tolist(),
                "pressure_anomaly_true": p_anom_true.tolist(),
                "pressure_anomaly_prob": p_anom_prob.tolist(),
                "flow_anomaly_true": q_anom_true.tolist(),
                "flow_anomaly_prob": q_anom_prob.tolist(),
            }
            demo_data.append(entry)
            print(f"  Snapshot {idx}: P_MAE={np.abs(p_pred - p_true).mean():.3f}")

    # Save
    out_path = PROJECT_ROOT / "dashboard" / "data" / "demo_snapshots.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"node_names": graph.node_names, "edge_names": graph.edge_names, "snapshots": demo_data}, f, indent=2)

    print(f"\nSaved {len(demo_data)} demo snapshots to {out_path}")


if __name__ == "__main__":
    main()
