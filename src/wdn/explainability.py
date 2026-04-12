"""GNN Explainability: understanding which nodes and edges drive model decisions.

Uses GNNExplainer to generate node/edge importance masks that show
which parts of the graph are most important for the model's predictions.

Usage:
    python -m wdn.explainability --data_dir data/generated_attacks --model_dir runs/multitask/20260310_201113
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch_geometric.explain import Explainer, GNNExplainer

from wdn.dataset import Normalizer, WDNDataset, train_val_test_split
from wdn.models.multitask import MultiTaskGNN


def get_device() -> torch.device:
    # GNNExplainer works best on CPU for stability
    return torch.device("cpu")


class ExplainableWrapper(torch.nn.Module):
    """Wraps MultiTaskGNN to output a single tensor for the Explainer API."""

    def __init__(self, model, target="pressure"):
        super().__init__()
        self.model = model
        self.target = target

    def forward(self, x, edge_index, edge_attr=None):
        # Build is_original_edge mask
        NE = edge_index.shape[1] // 2
        is_original = torch.zeros(edge_index.shape[1], dtype=torch.bool, device=x.device)
        is_original[:NE] = True

        out = self.model(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            is_original_edge=is_original,
            pressure_obs=x[:, 5],
            flow_obs=None,
            pressure_mask=x[:, 6],
            flow_mask=None,
        )

        if self.target == "pressure":
            return out["pressure_pred"].unsqueeze(-1)
        elif self.target == "anomaly" and "pressure_anomaly_logits" in out:
            return out["pressure_anomaly_logits"].unsqueeze(-1)
        return out["pressure_pred"].unsqueeze(-1)


def explain_snapshots(model, dataset, graph, device, n_snapshots=10, target="pressure"):
    """Run GNNExplainer on multiple snapshots and aggregate importance scores."""
    wrapper = ExplainableWrapper(model, target=target).to(device)
    wrapper.eval()

    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=200, lr=0.01),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="regression",
            task_level="node",
            return_type="raw",
        ),
    )

    node_importance_sum = np.zeros(graph.num_nodes)
    edge_importance_sum = np.zeros(dataset[0].edge_index.shape[1])
    feature_importance_sum = np.zeros(dataset[0].x.shape[1])
    n_explained = 0

    indices = np.random.default_rng(42).choice(len(dataset), min(n_snapshots, len(dataset)), replace=False)

    for idx in indices:
        data = dataset[idx].to(device)

        try:
            explanation = explainer(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                target=None,
            )

            if explanation.node_mask is not None:
                # Node feature importance: average across features per node
                node_feat_imp = explanation.node_mask.detach().cpu().numpy()
                feature_importance_sum += node_feat_imp.mean(axis=0)
                node_importance_sum += node_feat_imp.sum(axis=1)

            if explanation.edge_mask is not None:
                edge_imp = explanation.edge_mask.detach().cpu().numpy()
                edge_importance_sum[:len(edge_imp)] += edge_imp

            n_explained += 1

            if n_explained % 5 == 0:
                print(f"  Explained {n_explained}/{n_snapshots} snapshots")

        except Exception as e:
            print(f"  Warning: explanation failed for snapshot {idx}: {e}")
            continue

    if n_explained == 0:
        raise RuntimeError("No snapshots could be explained")

    # Average
    node_importance = node_importance_sum / n_explained
    edge_importance = edge_importance_sum / n_explained
    feature_importance = feature_importance_sum / n_explained

    # Normalize to [0, 1]
    if node_importance.max() > 0:
        node_importance = node_importance / node_importance.max()
    if edge_importance.max() > 0:
        edge_importance = edge_importance / edge_importance.max()
    if feature_importance.max() > 0:
        feature_importance = feature_importance / feature_importance.max()

    return {
        "node_importance": node_importance.tolist(),
        "edge_importance": edge_importance.tolist(),
        "feature_importance": feature_importance.tolist(),
        "n_explained": n_explained,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/generated_attacks")
    parser.add_argument("--model_dir", type=str, default="runs/multitask/20260310_201113")
    parser.add_argument("--n_snapshots", type=int, default=20)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)

    # Load data
    with open(data_dir / "graph.pkl", "rb") as f:
        graph = pickle.load(f)
    with open(data_dir / "snapshots.pkl", "rb") as f:
        snapshots = pickle.load(f)
    with open(data_dir / "corrupted.pkl", "rb") as f:
        corrupted = pickle.load(f)

    print(f"Network: {graph.num_nodes} nodes, {graph.num_edges} edges")

    # Split and normalize
    train_s, train_c, val_s, val_c, test_s, test_c = train_val_test_split(
        snapshots, corrupted, 0.70, 0.15, seed=42,
    )

    normalizer = Normalizer()
    normalizer.load_state_dict(torch.load(model_dir / "normalizer.pt", weights_only=True))

    test_dataset = WDNDataset(test_s, test_c, normalizer)

    # Load model
    sample = test_dataset[0]
    model = MultiTaskGNN(
        node_in_dim=sample.x.shape[1],
        edge_in_dim=sample.edge_attr.shape[1],
        hidden_dim=64, num_layers=2, dropout=0.1,
        gnn_type="GraphSAGE", heads=4,
    ).to(device)
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device, weights_only=True))
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: MultiTaskGNN with {n_params:,} parameters")

    # Explain reconstruction
    print(f"\nExplaining reconstruction decisions ({args.n_snapshots} snapshots)...")
    recon_results = explain_snapshots(
        model, test_dataset, graph, device,
        n_snapshots=args.n_snapshots, target="pressure",
    )

    # Explain anomaly detection
    print(f"\nExplaining anomaly detection decisions ({args.n_snapshots} snapshots)...")
    anomaly_results = explain_snapshots(
        model, test_dataset, graph, device,
        n_snapshots=args.n_snapshots, target="anomaly",
    )

    # Combine results
    FEATURE_NAMES = ["Elevation", "Base Demand", "Type (Junction)", "Type (Reservoir)", "Type (Tank)",
                     "Pressure Obs", "Pressure Mask"]

    output = {
        "network": "Net1",
        "node_names": graph.node_names,
        "edge_names": graph.edge_names,
        "feature_names": FEATURE_NAMES,
        "reconstruction": recon_results,
        "anomaly_detection": anomaly_results,
    }

    # Node ranking
    node_ranking = sorted(
        zip(graph.node_names, recon_results["node_importance"]),
        key=lambda x: x[1], reverse=True,
    )
    print("\nNode importance ranking (reconstruction):")
    for name, imp in node_ranking:
        print(f"  {name}: {imp:.3f}")

    print("\nFeature importance (reconstruction):")
    for name, imp in zip(FEATURE_NAMES, recon_results["feature_importance"]):
        print(f"  {name}: {imp:.3f}")

    # Save
    out_path = Path("dashboard/data/explainability.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
