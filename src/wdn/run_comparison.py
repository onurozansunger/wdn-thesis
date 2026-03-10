"""Compare model performance at different missing data rates.

Generates data at 30% and 50% missing rates, trains the GNN on each,
runs baselines, and produces a comparison table.

Usage:
    python -m wdn.run_comparison
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from wdn.config import (
    GenerateConfig, CorruptionConfig, TrainConfig, ModelConfig,
    load_config, save_config,
)
from wdn.data_generation import generate_dataset
from wdn.corruption import corrupt_all_snapshots
from wdn.dataset import train_val_test_split, create_dataloaders
from wdn.models.recon import ReconGNN, reconstruction_loss, physics_loss
from wdn.metrics import compute_recon_metrics
from wdn.baselines import pseudoinverse_baseline, wls_baseline


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def generate_data(missing_rate: float, seed: int = 42):
    """Generate dataset at a given missing rate."""
    cfg = GenerateConfig(
        network_inp="data/Net1.inp",
        duration_hours=24,
        hydraulic_timestep_minutes=60,
        num_scenarios=50,
        demand_variation=0.2,
        corruption=CorruptionConfig(
            missing_rate_pressure=missing_rate,
            missing_rate_flow=missing_rate,
            noise_sigma_pressure=0.5,
            noise_sigma_flow=0.2,
        ),
        seed=seed,
    )
    graph, snapshots = generate_dataset(cfg)
    corrupted = corrupt_all_snapshots(snapshots, cfg.corruption, seed=seed)
    return graph, snapshots, corrupted


def train_gnn(graph, snapshots, corrupted, device, seed=42):
    """Train ReconGNN and return test metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_s, train_c, val_s, val_c, test_s, test_c = train_val_test_split(
        snapshots, corrupted, 0.70, 0.15, seed,
    )
    train_loader, val_loader, test_loader, normalizer = create_dataloaders(
        train_s, train_c, val_s, val_c, test_s, test_c,
        batch_size=8, num_workers=0,
    )

    sample = train_loader.dataset[0]
    node_in_dim = sample.x.shape[1]
    edge_in_dim = sample.edge_attr.shape[1]

    model = ReconGNN(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
        gnn_type="GAT",
        heads=4,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
    )
    incidence = torch.tensor(graph.incidence_matrix, dtype=torch.float32).to(device)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, 101):
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
            if patience_counter >= 20:
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
        "p_all": compute_recon_metrics(p_pred, p_true),
        "p_unobs": compute_recon_metrics(p_pred, p_true, p_mask, only_unobserved=True),
        "q_all": compute_recon_metrics(q_pred, q_true),
        "q_unobs": compute_recon_metrics(q_pred, q_true, q_mask, only_unobserved=True),
    }


def run_baselines(graph, snapshots, corrupted, seed=42):
    """Run all baselines and return metrics."""
    _, _, _, _, test_s, test_c = train_val_test_split(
        snapshots, corrupted, 0.70, 0.15, seed,
    )

    methods = {
        "Pseudo-inverse": lambda s, c: pseudoinverse_baseline(graph, c.pressure_obs, c.flow_obs, c.pressure_mask, c.flow_mask),
        "WLS": lambda s, c: wls_baseline(graph, c.pressure_obs, c.flow_obs, c.pressure_mask, c.flow_mask),
    }

    results = {}
    for name, method in methods.items():
        all_p_pred, all_p_true, all_p_mask = [], [], []
        all_q_pred, all_q_true, all_q_mask = [], [], []

        for snap, corr in zip(test_s, test_c):
            result = method(snap, corr)
            all_p_pred.append(result.pressure_pred)
            all_p_true.append(snap.pressure_true)
            all_p_mask.append(corr.pressure_mask)
            all_q_pred.append(result.flow_pred)
            all_q_true.append(snap.flow_true)
            all_q_mask.append(corr.flow_mask)

        p_pred = torch.stack(all_p_pred)
        p_true = torch.stack(all_p_true)
        p_mask = torch.stack(all_p_mask)
        q_pred = torch.stack(all_q_pred)
        q_true = torch.stack(all_q_true)
        q_mask = torch.stack(all_q_mask)

        results[name] = {
            "p_all": compute_recon_metrics(p_pred, p_true),
            "p_unobs": compute_recon_metrics(p_pred, p_true, p_mask, only_unobserved=True),
            "q_all": compute_recon_metrics(q_pred, q_true),
            "q_unobs": compute_recon_metrics(q_pred, q_true, q_mask, only_unobserved=True),
        }

    return results


def main():
    device = get_device()
    print(f"Device: {device}\n")

    missing_rates = [0.3, 0.5]
    all_results = {}

    for rate in missing_rates:
        print(f"{'='*70}")
        print(f"  MISSING RATE: {rate*100:.0f}%")
        print(f"{'='*70}")

        # Generate data
        print(f"Generating data...")
        graph, snapshots, corrupted = generate_data(rate)
        print(f"  {len(snapshots)} snapshots generated")

        # Train GNN
        print(f"Training ReconGNN (GAT)...")
        t0 = time.time()
        gnn_metrics = train_gnn(graph, snapshots, corrupted, device)
        print(f"  Done in {time.time()-t0:.0f}s")

        # Run baselines
        print(f"Running baselines...")
        baseline_metrics = run_baselines(graph, snapshots, corrupted)

        all_results[f"{rate*100:.0f}%"] = {
            "ReconGNN (GAT)": {
                "P_MAE_all": gnn_metrics["p_all"].mae,
                "P_MAE_unobs": gnn_metrics["p_unobs"].mae,
                "Q_MAE_all": gnn_metrics["q_all"].mae,
                "Q_MAE_unobs": gnn_metrics["q_unobs"].mae,
            },
        }
        for bname, bmetrics in baseline_metrics.items():
            all_results[f"{rate*100:.0f}%"][bname] = {
                "P_MAE_all": bmetrics["p_all"].mae,
                "P_MAE_unobs": bmetrics["p_unobs"].mae,
                "Q_MAE_all": bmetrics["q_all"].mae,
                "Q_MAE_unobs": bmetrics["q_unobs"].mae,
            }

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"  COMPARISON: 30% vs 50% MISSING DATA")
    print(f"{'='*70}\n")

    methods = ["ReconGNN (GAT)", "Pseudo-inverse", "WLS"]
    header = f"{'Method':<22} | {'30% Missing':>20} | {'50% Missing':>20} |"
    sub    = f"{'':22} | {'P_MAE':>9} {'Q_MAE':>10} | {'P_MAE':>9} {'Q_MAE':>10} |"
    sep = "-" * len(header)

    print("Unobserved sensors only:")
    print(sep)
    print(header)
    print(sub)
    print(sep)

    for method in methods:
        r30 = all_results["30%"].get(method, {})
        r50 = all_results["50%"].get(method, {})
        p30 = r30.get("P_MAE_unobs", float("nan"))
        q30 = r30.get("Q_MAE_unobs", float("nan"))
        p50 = r50.get("P_MAE_unobs", float("nan"))
        q50 = r50.get("Q_MAE_unobs", float("nan"))
        print(f"{method:<22} | {p30:>9.3f} {q30:>10.4f} | {p50:>9.3f} {q50:>10.4f} |")

    print(sep)

    # Improvement factor
    print("\nGNN improvement over best baseline (WLS):")
    for rate_key in ["30%", "50%"]:
        gnn_p = all_results[rate_key]["ReconGNN (GAT)"]["P_MAE_unobs"]
        wls_p = all_results[rate_key]["WLS"]["P_MAE_unobs"]
        gnn_q = all_results[rate_key]["ReconGNN (GAT)"]["Q_MAE_unobs"]
        wls_q = all_results[rate_key]["WLS"]["Q_MAE_unobs"]
        print(f"  {rate_key}: Pressure {wls_p/gnn_p:.1f}x better, Flow {wls_q/gnn_q:.1f}x better")

    # Save results
    out_path = Path("data/comparison_30_50.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
