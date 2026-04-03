"""Pre-compute Modena demo data for the dashboard.

Run once:
    python dashboard/precompute/export_modena_demo.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wdn.dataset import WDNDataset, Normalizer, train_val_test_split
from wdn.models.multitask import MultiTaskGNN


MODENA_RUN = "20260403_234741"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    device = get_device()
    print(f"Device: {device}")

    data_dir = PROJECT_ROOT / "data" / "modena_attacks"
    run_dir = PROJECT_ROOT / "runs" / "multitask" / MODENA_RUN

    with open(data_dir / "graph.pkl", "rb") as f:
        graph = pickle.load(f)
    with open(data_dir / "snapshots.pkl", "rb") as f:
        snapshots = pickle.load(f)
    with open(data_dir / "corrupted.pkl", "rb") as f:
        corrupted = pickle.load(f)

    print(f"Loaded {len(snapshots)} snapshots ({graph.num_nodes} nodes, {graph.num_edges} edges)")

    train_s, train_c, val_s, val_c, test_s, test_c = train_val_test_split(
        snapshots, corrupted, 0.70, 0.15, seed=42,
    )

    normalizer = Normalizer()
    norm_state = torch.load(run_dir / "normalizer.pt", map_location="cpu", weights_only=True)
    normalizer.load_state_dict(norm_state)

    sample_ds = WDNDataset(test_s[:1], test_c[:1], normalizer)
    sample = sample_ds[0]

    model = MultiTaskGNN(
        node_in_dim=sample.x.shape[1],
        edge_in_dim=sample.edge_attr.shape[1],
        hidden_dim=64, num_layers=2, dropout=0.1,
        gnn_type="GraphSAGE", heads=4,
    ).to(device)

    state = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print("Model loaded")

    # Pick 20 evenly-spaced test snapshots
    n_demo = min(20, len(test_s))
    indices = np.linspace(0, len(test_s) - 1, n_demo, dtype=int)

    ds = WDNDataset(test_s, test_c, normalizer)
    demo_data = []

    with torch.no_grad():
        for idx in indices:
            data = ds[idx].to(device)

            out = model(
                x=data.x, edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                is_original_edge=data.is_original_edge,
                pressure_obs=data.pressure_obs,
                flow_obs=data.flow_obs,
                pressure_mask=data.pressure_mask,
                flow_mask=data.flow_mask,
            )

            p_pred = normalizer.denormalize_pressure(out["pressure_pred"].cpu()).numpy()
            p_true = normalizer.denormalize_pressure(data.y_pressure.cpu()).numpy()
            p_mask = data.pressure_mask.cpu().numpy()
            p_anom_true = data.pressure_anomaly.cpu().numpy()
            p_anom_prob = torch.sigmoid(out["pressure_anomaly_logits"]).cpu().numpy()

            entry = {
                "index": int(idx),
                "pressure_true": p_true.tolist(),
                "pressure_pred": p_pred.tolist(),
                "pressure_mask": p_mask.tolist(),
                "pressure_error": np.abs(p_pred - p_true).tolist(),
                "pressure_anomaly_true": p_anom_true.tolist(),
                "pressure_anomaly_prob": p_anom_prob.tolist(),
            }
            demo_data.append(entry)
            print(f"  Snapshot {idx}: P_MAE={np.abs(p_pred - p_true).mean():.3f}")

    # Graph info for visualization
    graph_info = {
        "node_names": graph.node_names,
        "node_types": graph.node_types.tolist(),
        "node_coordinates": graph.node_coordinates.tolist(),
        "edge_index": graph.edge_index.tolist(),
        "edge_names": graph.edge_names,
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
    }

    out_path = PROJECT_ROOT / "dashboard" / "data" / "modena_demo.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "graph": graph_info,
            "snapshots": demo_data,
            "test_results": json.load(open(run_dir / "test_results.json")),
            "history": json.load(open(run_dir / "history.json")),
        }, f, indent=2)

    print(f"\nSaved Modena demo data to {out_path}")


if __name__ == "__main__":
    main()
