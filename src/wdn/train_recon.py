"""Training script for ReconGNN.

Usage:
    python -m wdn.train_recon --config configs/train_recon.yaml
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from wdn.config import TrainConfig, load_config, save_config
from wdn.dataset import train_val_test_split, create_dataloaders
from wdn.models.recon import ReconGNN, reconstruction_loss, physics_loss
from wdn.metrics import compute_recon_metrics


def get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(
    model: ReconGNN,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    incidence_matrix: torch.Tensor,
    cfg: TrainConfig,
    graph_num_edges: int = 13,
) -> dict[str, float]:
    """Train for one epoch. Returns dict with loss components."""
    model.train()
    total_recon = 0.0
    total_phys = 0.0
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)

        # Forward pass
        out = model(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            is_original_edge=batch.is_original_edge,
        )

        # Reconstruction loss
        r_loss = reconstruction_loss(
            out["pressure_pred"], batch.y_pressure,
            out["flow_pred"], batch.y_flow,
            batch.pressure_mask, batch.flow_mask,
            loss_on_all=cfg.loss_on_all,
        )

        # Physics loss (mass conservation)
        p_loss = torch.tensor(0.0, device=device)
        if cfg.lambda_physics > 0:
            bs = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
            ne = graph_num_edges
            p_loss = physics_loss(out["flow_pred"], incidence_matrix, bs, ne)

        # Total loss
        loss = r_loss + cfg.lambda_physics * p_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_recon += r_loss.item()
        total_phys += p_loss.item()
        total_loss += loss.item()
        n_batches += 1

    return {
        "recon_loss": total_recon / n_batches,
        "physics_loss": total_phys / n_batches,
        "total_loss": total_loss / n_batches,
    }


@torch.no_grad()
def evaluate(
    model: ReconGNN,
    loader,
    device: torch.device,
    incidence_matrix: torch.Tensor,
    cfg: TrainConfig,
    normalizer=None,
    graph_num_edges: int = 13,
) -> dict:
    """Evaluate model on a dataset. Returns loss and metrics."""
    model.eval()
    total_recon = 0.0
    total_phys = 0.0
    n_batches = 0

    all_p_pred, all_p_true, all_p_mask = [], [], []
    all_q_pred, all_q_true, all_q_mask = [], [], []

    for batch in loader:
        batch = batch.to(device)

        out = model(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            is_original_edge=batch.is_original_edge,
        )

        r_loss = reconstruction_loss(
            out["pressure_pred"], batch.y_pressure,
            out["flow_pred"], batch.y_flow,
            batch.pressure_mask, batch.flow_mask,
            loss_on_all=cfg.loss_on_all,
        )

        p_loss = torch.tensor(0.0, device=device)
        if cfg.lambda_physics > 0:
            bs = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
            ne = graph_num_edges
            p_loss = physics_loss(out["flow_pred"], incidence_matrix, bs, ne)

        total_recon += r_loss.item()
        total_phys += p_loss.item()
        n_batches += 1

        # Collect predictions for metrics (denormalize if needed)
        p_pred = out["pressure_pred"].cpu()
        q_pred = out["flow_pred"].cpu()
        p_true = batch.y_pressure.cpu()
        q_true = batch.y_flow.cpu()

        if normalizer is not None:
            p_pred = normalizer.denormalize_pressure(p_pred)
            q_pred = normalizer.denormalize_flow(q_pred)
            p_true = normalizer.denormalize_pressure(p_true)
            q_true = normalizer.denormalize_flow(q_true)

        all_p_pred.append(p_pred)
        all_p_true.append(p_true)
        all_p_mask.append(batch.pressure_mask.cpu())
        all_q_pred.append(q_pred)
        all_q_true.append(q_true)
        all_q_mask.append(batch.flow_mask.cpu())

    # Concatenate and compute metrics
    p_pred = torch.cat(all_p_pred)
    p_true = torch.cat(all_p_true)
    p_mask = torch.cat(all_p_mask)
    q_pred = torch.cat(all_q_pred)
    q_true = torch.cat(all_q_true)
    q_mask = torch.cat(all_q_mask)

    p_metrics_all = compute_recon_metrics(p_pred, p_true)
    p_metrics_unobs = compute_recon_metrics(p_pred, p_true, p_mask, only_unobserved=True)
    q_metrics_all = compute_recon_metrics(q_pred, q_true)
    q_metrics_unobs = compute_recon_metrics(q_pred, q_true, q_mask, only_unobserved=True)

    return {
        "recon_loss": total_recon / max(n_batches, 1),
        "physics_loss": total_phys / max(n_batches, 1),
        "pressure_all": p_metrics_all,
        "pressure_unobs": p_metrics_unobs,
        "flow_all": q_metrics_all,
        "flow_unobs": q_metrics_unobs,
    }


def main():
    parser = argparse.ArgumentParser(description="Train ReconGNN")
    parser.add_argument("--config", type=str, default="configs/train_recon.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config, TrainConfig)
    device = get_device()
    print(f"Device: {device}")

    # Seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load data
    data_dir = Path("data/generated")
    with open(data_dir / "graph.pkl", "rb") as f:
        graph = pickle.load(f)
    with open(data_dir / "snapshots.pkl", "rb") as f:
        snapshots = pickle.load(f)
    with open(data_dir / "corrupted.pkl", "rb") as f:
        corrupted = pickle.load(f)

    print(f"Loaded {len(snapshots)} snapshots ({graph.num_nodes} nodes, {graph.num_edges} edges)")

    # Split and create dataloaders
    train_s, train_c, val_s, val_c, test_s, test_c = train_val_test_split(
        snapshots, corrupted, cfg.train_ratio, cfg.val_ratio, cfg.seed,
    )
    train_loader, val_loader, test_loader, normalizer = create_dataloaders(
        train_s, train_c, val_s, val_c, test_s, test_c,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers,
    )

    # Incidence matrix for physics loss (move to device)
    incidence = torch.tensor(graph.incidence_matrix, dtype=torch.float32).to(device)

    # Build model
    model = ReconGNN(
        node_in_dim=7,      # 5 static + 1 obs + 1 mask
        edge_in_dim=8,      # 6 static + 1 obs + 1 mask
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        gnn_type=cfg.model.gnn_type,
        heads=cfg.model.heads,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg.model.gnn_type} with {n_params:,} parameters")

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
    )

    # Output directory
    run_dir = Path(cfg.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")

    # Training loop
    best_val_loss = float("inf")
    history = []

    print(f"\nTraining for {cfg.epochs} epochs...")
    print(f"  Physics loss weight: {cfg.lambda_physics}")
    print(f"  Loss on all nodes: {cfg.loss_on_all}")
    print()

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, incidence, cfg,
            graph_num_edges=graph.num_edges,
        )

        # Validate
        val_metrics = evaluate(
            model, val_loader, device, incidence, cfg, normalizer,
            graph_num_edges=graph.num_edges,
        )

        # Learning rate scheduling
        scheduler.step(val_metrics["recon_loss"])

        elapsed = time.time() - t0

        # Log
        entry = {
            "epoch": epoch,
            "train_recon_loss": train_metrics["recon_loss"],
            "train_physics_loss": train_metrics["physics_loss"],
            "val_recon_loss": val_metrics["recon_loss"],
            "val_p_mae_unobs": val_metrics["pressure_unobs"].mae,
            "val_q_mae_unobs": val_metrics["flow_unobs"].mae,
        }
        history.append(entry)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{cfg.epochs} ({elapsed:.1f}s) | "
                f"Train: {train_metrics['total_loss']:.4f} | "
                f"Val: {val_metrics['recon_loss']:.4f} | "
                f"P_MAE(unobs): {val_metrics['pressure_unobs'].mae:.3f} | "
                f"Q_MAE(unobs): {val_metrics['flow_unobs'].mae:.4f}"
            )

        # Save best model
        if val_metrics["recon_loss"] < best_val_loss:
            best_val_loss = val_metrics["recon_loss"]
            torch.save(model.state_dict(), run_dir / "best_model.pt")
            torch.save(normalizer.state_dict(), run_dir / "normalizer.pt")

    # Final test evaluation
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", weights_only=True))
    test_metrics = evaluate(model, test_loader, device, incidence, cfg, normalizer,
                            graph_num_edges=graph.num_edges)

    print(f"  Pressure (all):       {test_metrics['pressure_all']}")
    print(f"  Pressure (unobs):     {test_metrics['pressure_unobs']}")
    print(f"  Flow (all):           {test_metrics['flow_all']}")
    print(f"  Flow (unobs):         {test_metrics['flow_unobs']}")

    # Save results
    test_results = {
        "pressure_all": {"mae": test_metrics["pressure_all"].mae,
                         "mse": test_metrics["pressure_all"].mse},
        "pressure_unobs": {"mae": test_metrics["pressure_unobs"].mae,
                           "mse": test_metrics["pressure_unobs"].mse},
        "flow_all": {"mae": test_metrics["flow_all"].mae,
                     "mse": test_metrics["flow_all"].mse},
        "flow_unobs": {"mae": test_metrics["flow_unobs"].mae,
                       "mse": test_metrics["flow_unobs"].mse},
    }

    with open(run_dir / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    main()
