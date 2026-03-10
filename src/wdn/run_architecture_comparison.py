"""Compare GNN architectures for state reconstruction.

Tests GAT, GATv2, Transformer, GraphSAGE, and GCN on the same dataset.
Produces a comparison table with reconstruction metrics.

Usage:
    python -m wdn.run_architecture_comparison
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from wdn.dataset import train_val_test_split, create_dataloaders
from wdn.models.recon import ReconGNN, reconstruction_loss, physics_loss
from wdn.metrics import compute_recon_metrics


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_and_evaluate(
    gnn_type: str,
    heads: int,
    graph,
    train_loader,
    val_loader,
    test_loader,
    normalizer,
    device,
    seed: int = 42,
    epochs: int = 100,
    patience: int = 20,
) -> dict:
    """Train a ReconGNN with specified architecture and return test metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    sample = train_loader.dataset[0]
    node_in_dim = sample.x.shape[1]
    edge_in_dim = sample.edge_attr.shape[1]

    model = ReconGNN(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
        gnn_type=gnn_type,
        heads=heads,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
    )
    incidence = torch.tensor(graph.incidence_matrix, dtype=torch.float32).to(device)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            out = model(
                x=batch.x, edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                is_original_edge=batch.is_original_edge,
                batch=batch.batch if hasattr(batch, 'batch') else None,
            )
            r_loss = reconstruction_loss(
                out["pressure_pred"], batch.y_pressure,
                out["flow_pred"], batch.y_flow,
                batch.pressure_mask, batch.flow_mask, loss_on_all=True,
            )
            bs = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
            p_loss = physics_loss(out["flow_pred"], incidence, bs, graph.num_edges)
            loss = r_loss + 0.1 * p_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Validate
        model.eval()
        val_loss_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(
                    x=batch.x, edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    is_original_edge=batch.is_original_edge,
                    batch=batch.batch if hasattr(batch, 'batch') else None,
                )
                r_loss = reconstruction_loss(
                    out["pressure_pred"], batch.y_pressure,
                    out["flow_pred"], batch.y_flow,
                    batch.pressure_mask, batch.flow_mask, loss_on_all=True,
                )
                val_loss_sum += r_loss.item()
                n_val += 1

        val_loss = val_loss_sum / max(n_val, 1)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Test evaluation
    model.load_state_dict(best_state)
    model.eval()

    all_p_pred, all_p_true, all_p_mask = [], [], []
    all_q_pred, all_q_true, all_q_mask = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(
                x=batch.x, edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                is_original_edge=batch.is_original_edge,
                batch=batch.batch if hasattr(batch, 'batch') else None,
            )
            p_pred = normalizer.denormalize_pressure(out["pressure_pred"].cpu())
            q_pred = normalizer.denormalize_flow(out["flow_pred"].cpu())
            p_true = normalizer.denormalize_pressure(batch.y_pressure.cpu())
            q_true = normalizer.denormalize_flow(batch.y_flow.cpu())

            all_p_pred.append(p_pred)
            all_p_true.append(p_true)
            all_p_mask.append(batch.pressure_mask.cpu())
            all_q_pred.append(q_pred)
            all_q_true.append(q_true)
            all_q_mask.append(batch.flow_mask.cpu())

    p_pred = torch.cat(all_p_pred)
    p_true = torch.cat(all_p_true)
    p_mask = torch.cat(all_p_mask)
    q_pred = torch.cat(all_q_pred)
    q_true = torch.cat(all_q_true)
    q_mask = torch.cat(all_q_mask)

    return {
        "gnn_type": gnn_type,
        "heads": heads,
        "n_params": n_params,
        "best_epoch": epoch - patience_counter,
        "p_all": compute_recon_metrics(p_pred, p_true),
        "p_unobs": compute_recon_metrics(p_pred, p_true, p_mask, only_unobserved=True),
        "q_all": compute_recon_metrics(q_pred, q_true),
        "q_unobs": compute_recon_metrics(q_pred, q_true, q_mask, only_unobserved=True),
    }


def main():
    device = get_device()
    print(f"Device: {device}\n")

    # Load data
    data_dir = Path("data/generated")
    with open(data_dir / "graph.pkl", "rb") as f:
        graph = pickle.load(f)
    with open(data_dir / "snapshots.pkl", "rb") as f:
        snapshots = pickle.load(f)
    with open(data_dir / "corrupted.pkl", "rb") as f:
        corrupted = pickle.load(f)

    print(f"Loaded {len(snapshots)} snapshots ({graph.num_nodes} nodes, {graph.num_edges} edges)")

    # Split data (same split for all architectures)
    train_s, train_c, val_s, val_c, test_s, test_c = train_val_test_split(
        snapshots, corrupted, 0.70, 0.15, seed=42,
    )
    train_loader, val_loader, test_loader, normalizer = create_dataloaders(
        train_s, train_c, val_s, val_c, test_s, test_c,
        batch_size=8, num_workers=0,
    )

    # Architectures to test
    architectures = [
        ("GAT", 4),
        ("GATv2", 4),
        ("Transformer", 4),
        ("GraphSAGE", 1),
        ("GCN", 1),
    ]

    results = []

    for gnn_type, heads in architectures:
        print(f"\n{'='*60}")
        print(f"  Training: {gnn_type} (heads={heads})")
        print(f"{'='*60}")

        t0 = time.time()
        metrics = train_and_evaluate(
            gnn_type, heads, graph,
            train_loader, val_loader, test_loader, normalizer,
            device,
        )
        elapsed = time.time() - t0

        metrics["train_time"] = elapsed
        results.append(metrics)

        print(f"  Done in {elapsed:.0f}s (best epoch: {metrics['best_epoch']})")
        print(f"  Params: {metrics['n_params']:,}")
        print(f"  Pressure (unobs): MAE={metrics['p_unobs'].mae:.3f}")
        print(f"  Flow (unobs):     MAE={metrics['q_unobs'].mae:.4f}")

    # Print comparison table
    print(f"\n\n{'='*80}")
    print(f"  ARCHITECTURE COMPARISON (50% missing data)")
    print(f"{'='*80}\n")

    header = f"{'Architecture':<16} | {'Params':>8} | {'P_MAE':>8} {'P_RMSE':>8} | {'Q_MAE':>8} {'Q_RMSE':>8} | {'Time':>6}"
    sep = "-" * len(header)

    print("Unobserved sensors:")
    print(sep)
    print(header)
    print(sep)

    for r in results:
        print(
            f"{r['gnn_type']:<16} | {r['n_params']:>8,} | "
            f"{r['p_unobs'].mae:>8.3f} {r['p_unobs'].rmse:>8.3f} | "
            f"{r['q_unobs'].mae:>8.4f} {r['q_unobs'].rmse:>8.4f} | "
            f"{r['train_time']:>5.0f}s"
        )

    print(sep)

    print("\nAll sensors:")
    print(sep)
    print(header)
    print(sep)

    for r in results:
        print(
            f"{r['gnn_type']:<16} | {r['n_params']:>8,} | "
            f"{r['p_all'].mae:>8.3f} {r['p_all'].rmse:>8.3f} | "
            f"{r['q_all'].mae:>8.4f} {r['q_all'].rmse:>8.4f} | "
            f"{r['train_time']:>5.0f}s"
        )

    print(sep)

    # Find best
    best = min(results, key=lambda r: r["p_unobs"].mae)
    print(f"\nBest architecture: {best['gnn_type']} (P_MAE={best['p_unobs'].mae:.3f})")

    # Save results
    out_path = Path("data/architecture_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {}
    for r in results:
        save_data[r["gnn_type"]] = {
            "heads": r["heads"],
            "n_params": r["n_params"],
            "best_epoch": r["best_epoch"],
            "train_time": r["train_time"],
            "p_all_mae": r["p_all"].mae,
            "p_all_rmse": r["p_all"].rmse,
            "p_unobs_mae": r["p_unobs"].mae,
            "p_unobs_rmse": r["p_unobs"].rmse,
            "q_all_mae": r["q_all"].mae,
            "q_all_rmse": r["q_all"].rmse,
            "q_unobs_mae": r["q_unobs"].mae,
            "q_unobs_rmse": r["q_unobs"].rmse,
        }

    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
