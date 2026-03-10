"""Training script for MultiTaskGNN (joint reconstruction + anomaly detection).

Trains on attack data and evaluates both reconstruction quality and
anomaly detection performance (precision, recall, F1, AUROC).

Usage:
    python -m wdn.train_multitask [--data_dir data/generated_attacks]
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from wdn.dataset import train_val_test_split, create_dataloaders
from wdn.models.multitask import MultiTaskGNN, multitask_loss
from wdn.models.recon import physics_loss
from wdn.metrics import compute_recon_metrics, compute_anomaly_metrics


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(model, loader, optimizer, device, incidence, graph_num_edges,
                    lambda_physics=0.1, lambda_anomaly=1.0):
    """Train for one epoch."""
    model.train()
    total_recon = 0.0
    total_anomaly = 0.0
    total_loss = 0.0
    n = 0

    for batch in loader:
        batch = batch.to(device)

        out = model(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            is_original_edge=batch.is_original_edge,
            batch=batch.batch if hasattr(batch, 'batch') else None,
            pressure_obs=batch.pressure_obs,
            flow_obs=batch.flow_obs,
            pressure_mask=batch.pressure_mask,
            flow_mask=batch.flow_mask,
        )

        losses = multitask_loss(
            out["pressure_pred"], batch.y_pressure,
            out["flow_pred"], batch.y_flow,
            batch.pressure_mask, batch.flow_mask,
            out.get("pressure_anomaly_logits"),
            out.get("flow_anomaly_logits"),
            batch.pressure_anomaly,
            batch.flow_anomaly,
            lambda_anomaly=lambda_anomaly,
            loss_on_all=True,
        )

        # Add physics loss
        p_loss = torch.tensor(0.0, device=device)
        if lambda_physics > 0:
            bs = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
            p_loss = physics_loss(out["flow_pred"], incidence, bs, graph_num_edges)

        loss = losses["total_loss"] + lambda_physics * p_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_recon += losses["recon_loss"].item()
        total_anomaly += losses["anomaly_loss"].item()
        total_loss += loss.item()
        n += 1

    return {
        "recon_loss": total_recon / max(n, 1),
        "anomaly_loss": total_anomaly / max(n, 1),
        "total_loss": total_loss / max(n, 1),
    }


@torch.no_grad()
def evaluate(model, loader, device, incidence, graph_num_edges, normalizer=None,
             lambda_anomaly=1.0):
    """Evaluate reconstruction and anomaly detection."""
    model.eval()
    total_recon = 0.0
    total_anomaly = 0.0
    n = 0

    all_p_pred, all_p_true, all_p_mask = [], [], []
    all_q_pred, all_q_true, all_q_mask = [], [], []
    all_p_anom_logits, all_p_anom_true, all_p_anom_mask = [], [], []
    all_q_anom_logits, all_q_anom_true, all_q_anom_mask = [], [], []

    for batch in loader:
        batch = batch.to(device)

        out = model(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            is_original_edge=batch.is_original_edge,
            batch=batch.batch if hasattr(batch, 'batch') else None,
            pressure_obs=batch.pressure_obs,
            flow_obs=batch.flow_obs,
            pressure_mask=batch.pressure_mask,
            flow_mask=batch.flow_mask,
        )

        losses = multitask_loss(
            out["pressure_pred"], batch.y_pressure,
            out["flow_pred"], batch.y_flow,
            batch.pressure_mask, batch.flow_mask,
            out.get("pressure_anomaly_logits"),
            out.get("flow_anomaly_logits"),
            batch.pressure_anomaly,
            batch.flow_anomaly,
            lambda_anomaly=lambda_anomaly,
        )

        total_recon += losses["recon_loss"].item()
        total_anomaly += losses["anomaly_loss"].item()
        n += 1

        # Reconstruction predictions
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

        # Anomaly predictions
        if "pressure_anomaly_logits" in out:
            all_p_anom_logits.append(out["pressure_anomaly_logits"].cpu())
            all_p_anom_true.append(batch.pressure_anomaly.cpu())
            all_p_anom_mask.append(batch.pressure_mask.cpu())
        if "flow_anomaly_logits" in out:
            all_q_anom_logits.append(out["flow_anomaly_logits"].cpu())
            all_q_anom_true.append(batch.flow_anomaly.cpu())
            all_q_anom_mask.append(batch.flow_mask.cpu())

    # Reconstruction metrics
    p_pred = torch.cat(all_p_pred)
    p_true = torch.cat(all_p_true)
    p_mask = torch.cat(all_p_mask)
    q_pred = torch.cat(all_q_pred)
    q_true = torch.cat(all_q_true)
    q_mask = torch.cat(all_q_mask)

    result = {
        "recon_loss": total_recon / max(n, 1),
        "anomaly_loss": total_anomaly / max(n, 1),
        "pressure_all": compute_recon_metrics(p_pred, p_true),
        "pressure_unobs": compute_recon_metrics(p_pred, p_true, p_mask, only_unobserved=True),
        "flow_all": compute_recon_metrics(q_pred, q_true),
        "flow_unobs": compute_recon_metrics(q_pred, q_true, q_mask, only_unobserved=True),
    }

    # Anomaly detection metrics
    if all_p_anom_logits:
        p_logits = torch.cat(all_p_anom_logits)
        p_labels = torch.cat(all_p_anom_true)
        p_amask = torch.cat(all_p_anom_mask)

        p_scores = torch.sigmoid(p_logits)
        p_pred_labels = (p_scores > 0.5).float()

        result["pressure_anomaly"] = compute_anomaly_metrics(
            p_pred_labels, p_labels, p_scores, p_amask,
        )

    if all_q_anom_logits:
        q_logits = torch.cat(all_q_anom_logits)
        q_labels = torch.cat(all_q_anom_true)
        q_amask = torch.cat(all_q_anom_mask)

        q_scores = torch.sigmoid(q_logits)
        q_pred_labels = (q_scores > 0.5).float()

        result["flow_anomaly"] = compute_anomaly_metrics(
            q_pred_labels, q_labels, q_scores, q_amask,
        )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/generated_attacks")
    parser.add_argument("--gnn_type", type=str, default="GraphSAGE")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lambda_anomaly", type=float, default=1.0)
    parser.add_argument("--lambda_physics", type=float, default=0.1)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Load attack data
    data_dir = Path(args.data_dir)
    with open(data_dir / "graph.pkl", "rb") as f:
        graph = pickle.load(f)
    with open(data_dir / "snapshots.pkl", "rb") as f:
        snapshots = pickle.load(f)
    with open(data_dir / "corrupted.pkl", "rb") as f:
        corrupted = pickle.load(f)

    # Check attack labels
    n_attacked = sum(1 for c in corrupted if c.pressure_anomaly.sum() > 0 or c.flow_anomaly.sum() > 0)
    print(f"Loaded {len(snapshots)} snapshots ({n_attacked} with attacks)")
    print(f"Network: {graph.num_nodes} nodes, {graph.num_edges} edges")

    # Split
    train_s, train_c, val_s, val_c, test_s, test_c = train_val_test_split(
        snapshots, corrupted, 0.70, 0.15, seed=42,
    )
    train_loader, val_loader, test_loader, normalizer = create_dataloaders(
        train_s, train_c, val_s, val_c, test_s, test_c,
        batch_size=8, num_workers=0,
    )

    # Build model
    sample = train_loader.dataset[0]
    model = MultiTaskGNN(
        node_in_dim=sample.x.shape[1],
        edge_in_dim=sample.edge_attr.shape[1],
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
        gnn_type=args.gnn_type,
        heads=4,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: MultiTaskGNN ({args.gnn_type}) with {n_params:,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
    )
    incidence = torch.tensor(graph.incidence_matrix, dtype=torch.float32).to(device)

    # Training
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = []

    print(f"\nTraining for {args.epochs} epochs...")
    print(f"  Lambda anomaly: {args.lambda_anomaly}, Lambda physics: {args.lambda_physics}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_m = train_one_epoch(
            model, train_loader, optimizer, device, incidence,
            graph.num_edges, args.lambda_physics, args.lambda_anomaly,
        )

        val_m = evaluate(
            model, val_loader, device, incidence, graph.num_edges,
            normalizer, args.lambda_anomaly,
        )

        scheduler.step(val_m["recon_loss"])
        elapsed = time.time() - t0

        entry = {
            "epoch": epoch,
            "train_recon": train_m["recon_loss"],
            "train_anomaly": train_m["anomaly_loss"],
            "val_recon": val_m["recon_loss"],
            "val_p_mae_unobs": val_m["pressure_unobs"].mae,
        }
        if "pressure_anomaly" in val_m:
            entry["val_p_anomaly_f1"] = val_m["pressure_anomaly"].f1
            entry["val_p_anomaly_auroc"] = val_m["pressure_anomaly"].auroc

        history.append(entry)

        if epoch % 5 == 0 or epoch == 1:
            anom_str = ""
            if "pressure_anomaly" in val_m:
                anom_str = (f" | P_Anom F1={val_m['pressure_anomaly'].f1:.3f}"
                           f" AUROC={val_m['pressure_anomaly'].auroc:.3f}")
            print(
                f"Epoch {epoch:3d}/{args.epochs} ({elapsed:.1f}s) | "
                f"Train: {train_m['total_loss']:.4f} | "
                f"Val Recon: {val_m['recon_loss']:.4f} | "
                f"P_MAE(unobs): {val_m['pressure_unobs'].mae:.3f}"
                f"{anom_str}"
            )

        if val_m["recon_loss"] < best_val_loss:
            best_val_loss = val_m["recon_loss"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Test evaluation
    print(f"\n{'='*70}")
    print("TEST SET EVALUATION")
    print(f"{'='*70}")

    model.load_state_dict(best_state)
    test_m = evaluate(
        model, test_loader, device, incidence, graph.num_edges,
        normalizer, args.lambda_anomaly,
    )

    print(f"\n  Reconstruction:")
    print(f"    Pressure (all):   {test_m['pressure_all']}")
    print(f"    Pressure (unobs): {test_m['pressure_unobs']}")
    print(f"    Flow (all):       {test_m['flow_all']}")
    print(f"    Flow (unobs):     {test_m['flow_unobs']}")

    if "pressure_anomaly" in test_m:
        print(f"\n  Anomaly Detection:")
        print(f"    Pressure: {test_m['pressure_anomaly']}")
    if "flow_anomaly" in test_m:
        print(f"    Flow:     {test_m['flow_anomaly']}")

    # Save results
    run_dir = Path("runs/multitask") / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    test_results = {
        "model": args.gnn_type,
        "n_params": n_params,
        "reconstruction": {
            "pressure_all": {"mae": test_m["pressure_all"].mae, "rmse": test_m["pressure_all"].rmse},
            "pressure_unobs": {"mae": test_m["pressure_unobs"].mae, "rmse": test_m["pressure_unobs"].rmse},
            "flow_all": {"mae": test_m["flow_all"].mae, "rmse": test_m["flow_all"].rmse},
            "flow_unobs": {"mae": test_m["flow_unobs"].mae, "rmse": test_m["flow_unobs"].rmse},
        },
    }
    if "pressure_anomaly" in test_m:
        test_results["anomaly_detection"] = {
            "pressure": {
                "precision": test_m["pressure_anomaly"].precision,
                "recall": test_m["pressure_anomaly"].recall,
                "f1": test_m["pressure_anomaly"].f1,
                "auroc": test_m["pressure_anomaly"].auroc,
            },
        }
    if "flow_anomaly" in test_m:
        test_results["anomaly_detection"]["flow"] = {
            "precision": test_m["flow_anomaly"].precision,
            "recall": test_m["flow_anomaly"].recall,
            "f1": test_m["flow_anomaly"].f1,
            "auroc": test_m["flow_anomaly"].auroc,
        }

    with open(run_dir / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    torch.save(model.state_dict(), run_dir / "best_model.pt")
    torch.save(normalizer.state_dict(), run_dir / "normalizer.pt")

    print(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    main()
