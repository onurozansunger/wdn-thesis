"""Evaluate baseline methods on the generated dataset.

Usage:
    python -m wdn.eval_baselines
"""

from __future__ import annotations

import pickle
import json
import time
from pathlib import Path

import numpy as np
import torch

from wdn.data_generation import WDNGraph, Snapshot
from wdn.corruption import CorruptedSnapshot
from wdn.baselines import pseudoinverse_baseline, wls_baseline
from wdn.metrics import compute_recon_metrics, ReconMetrics, aggregate_metrics


def evaluate_baseline(
    name: str,
    baseline_fn,
    graph: WDNGraph,
    snapshots: list[Snapshot],
    corrupted: list[CorruptedSnapshot],
    **kwargs,
) -> dict:
    """Run a baseline on all snapshots and compute metrics.

    Returns dict with pressure and flow metrics (all values + unobserved only).
    """
    p_metrics_all = []
    p_metrics_unobs = []
    q_metrics_all = []
    q_metrics_unobs = []

    for snap, corr in zip(snapshots, corrupted):
        result = baseline_fn(
            graph=graph,
            pressure_obs=corr.pressure_obs,
            flow_obs=corr.flow_obs,
            pressure_mask=corr.pressure_mask,
            flow_mask=corr.flow_mask,
            **kwargs,
        )

        # Pressure metrics
        p_all = compute_recon_metrics(result.pressure_pred, snap.pressure_true)
        p_unobs = compute_recon_metrics(
            result.pressure_pred, snap.pressure_true,
            mask=corr.pressure_mask, only_unobserved=True,
        )
        p_metrics_all.append(p_all)
        p_metrics_unobs.append(p_unobs)

        # Flow metrics
        q_all = compute_recon_metrics(result.flow_pred, snap.flow_true)
        q_unobs = compute_recon_metrics(
            result.flow_pred, snap.flow_true,
            mask=corr.flow_mask, only_unobserved=True,
        )
        q_metrics_all.append(q_all)
        q_metrics_unobs.append(q_unobs)

    return {
        "name": name,
        "pressure_all": aggregate_metrics(p_metrics_all),
        "pressure_unobserved": aggregate_metrics(p_metrics_unobs),
        "flow_all": aggregate_metrics(q_metrics_all),
        "flow_unobserved": aggregate_metrics(q_metrics_unobs),
    }


def print_results(results: list[dict]):
    """Pretty-print baseline comparison table."""
    print("\n" + "=" * 80)
    print("BASELINE EVALUATION RESULTS")
    print("=" * 80)

    # Header
    print(f"\n{'Method':<20} {'Metric':<12} {'MAE':>10} {'MSE':>12} {'RMSE':>10}")
    print("-" * 64)

    for r in results:
        name = r["name"]
        for label, key in [
            ("P (all)", "pressure_all"),
            ("P (unobs)", "pressure_unobserved"),
            ("Q (all)", "flow_all"),
            ("Q (unobs)", "flow_unobserved"),
        ]:
            m = r[key]
            prefix = name if label == "P (all)" else ""
            print(f"{prefix:<20} {label:<12} {m.mae:>10.4f} {m.mse:>12.6f} {m.rmse:>10.4f}")
        print("-" * 64)


def main():
    data_dir = Path("data/generated")

    # Load data
    print("Loading generated dataset...")
    with open(data_dir / "graph.pkl", "rb") as f:
        graph = pickle.load(f)
    with open(data_dir / "snapshots.pkl", "rb") as f:
        snapshots = pickle.load(f)
    with open(data_dir / "corrupted.pkl", "rb") as f:
        corrupted = pickle.load(f)

    print(f"  {len(snapshots)} snapshots, {graph.num_nodes} nodes, {graph.num_edges} edges")

    # Use a subset for faster evaluation (or all)
    n_eval = min(len(snapshots), 500)
    snapshots_eval = snapshots[:n_eval]
    corrupted_eval = corrupted[:n_eval]
    print(f"  Evaluating on {n_eval} snapshots")

    results = []

    # --- Pseudo-inverse baseline ---
    print("\nRunning pseudo-inverse baseline...")
    t0 = time.time()
    r_pinv = evaluate_baseline(
        "Pseudo-inverse",
        pseudoinverse_baseline,
        graph, snapshots_eval, corrupted_eval,
        rcond=1e-6,
    )
    print(f"  Done in {time.time() - t0:.2f}s")
    results.append(r_pinv)

    # --- WLS baseline ---
    print("Running WLS baseline...")
    t0 = time.time()
    r_wls = evaluate_baseline(
        "WLS",
        wls_baseline,
        graph, snapshots_eval, corrupted_eval,
        alpha=1.0, beta=0.1,
    )
    print(f"  Done in {time.time() - t0:.2f}s")
    results.append(r_wls)

    # --- Mean imputation baseline (simplest possible) ---
    print("Running mean imputation baseline...")
    t0 = time.time()
    r_mean = evaluate_mean_baseline(graph, snapshots_eval, corrupted_eval)
    print(f"  Done in {time.time() - t0:.2f}s")
    results.append(r_mean)

    # Print comparison
    print_results(results)

    # Save results
    out_path = data_dir / "baseline_results.json"
    save_results = {}
    for r in results:
        save_results[r["name"]] = {
            k: {"mae": v.mae, "mse": v.mse, "rmse": v.rmse}
            for k, v in r.items() if k != "name"
        }
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


def evaluate_mean_baseline(
    graph: WDNGraph,
    snapshots: list[Snapshot],
    corrupted: list[CorruptedSnapshot],
) -> dict:
    """Simplest baseline: replace missing values with mean of observed."""
    from wdn.baselines import BaselineResult

    p_metrics_all = []
    p_metrics_unobs = []
    q_metrics_all = []
    q_metrics_unobs = []

    for snap, corr in zip(snapshots, corrupted):
        p_obs = corr.pressure_obs.numpy()
        p_mask = corr.pressure_mask.numpy().astype(bool)
        q_obs = corr.flow_obs.numpy()
        q_mask = corr.flow_mask.numpy().astype(bool)

        # Mean imputation
        p_pred = p_obs.copy()
        if p_mask.sum() > 0:
            p_pred[~p_mask] = p_obs[p_mask].mean()

        q_pred = q_obs.copy()
        if q_mask.sum() > 0:
            q_pred[~q_mask] = q_obs[q_mask].mean()

        p_pred_t = torch.tensor(p_pred, dtype=torch.float32)
        q_pred_t = torch.tensor(q_pred, dtype=torch.float32)

        p_metrics_all.append(compute_recon_metrics(p_pred_t, snap.pressure_true))
        p_metrics_unobs.append(compute_recon_metrics(
            p_pred_t, snap.pressure_true, corr.pressure_mask, only_unobserved=True))
        q_metrics_all.append(compute_recon_metrics(q_pred_t, snap.flow_true))
        q_metrics_unobs.append(compute_recon_metrics(
            q_pred_t, snap.flow_true, corr.flow_mask, only_unobserved=True))

    return {
        "name": "Mean imputation",
        "pressure_all": aggregate_metrics(p_metrics_all),
        "pressure_unobserved": aggregate_metrics(p_metrics_unobs),
        "flow_all": aggregate_metrics(q_metrics_all),
        "flow_unobserved": aggregate_metrics(q_metrics_unobs),
    }


if __name__ == "__main__":
    main()
