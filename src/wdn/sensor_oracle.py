"""Sensor Oracle: Optimal sensor placement using MC Dropout uncertainty.

Given a trained GNN model, identifies which unmonitored nodes would
benefit most from installing a new sensor. Uses greedy uncertainty
reduction: at each step, "place" a sensor at the node with highest
average uncertainty and measure the improvement.

Algorithm:
    1. Run MC Dropout on test data → per-node uncertainty (std of predictions)
    2. Aggregate uncertainty across all test snapshots per node
    3. Greedy loop:
        a. Find the unmonitored node with highest mean uncertainty
        b. "Install" a sensor there (set its mask to 1, fill obs with true value)
        c. Re-run MC Dropout → measure new total uncertainty
        d. Record uncertainty reduction
    4. Output: ranked placement recommendations + reduction curve

Usage:
    python -m wdn.sensor_oracle [--model_dir runs/multitask/20260310_201113]
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

from wdn.data_generation import WDNGraph
from wdn.dataset import (
    WDNDataset, Normalizer,
    train_val_test_split, create_dataloaders,
)
from wdn.models.multitask import MultiTaskGNN
from wdn.metrics import compute_recon_metrics


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Uncertainty estimation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_node_uncertainty(
    model: MultiTaskGNN,
    loader: PyGDataLoader,
    device: torch.device,
    num_nodes: int,
    n_mc_samples: int = 30,
) -> dict[str, np.ndarray]:
    """Run MC Dropout on a dataset and aggregate per-node uncertainty.

    Returns:
        Dict with:
            - mean_uncertainty: (N,) average std across all test snapshots per node
            - max_uncertainty: (N,) max std across all test snapshots per node
            - mean_error: (N,) average |pred - true| per node (reconstruction error)
            - per_snapshot_std: list of (N,) arrays per snapshot
    """
    model.train()  # Enable dropout for MC sampling

    # Accumulate per-node stats
    node_std_sum = np.zeros(num_nodes, dtype=np.float64)
    node_std_max = np.zeros(num_nodes, dtype=np.float64)
    node_error_sum = np.zeros(num_nodes, dtype=np.float64)
    node_count = np.zeros(num_nodes, dtype=np.float64)

    per_snapshot_std = []

    for batch in loader:
        batch = batch.to(device)
        batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else 1

        # MC Dropout: run multiple forward passes
        p_samples = []
        for _ in range(n_mc_samples):
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
            p_samples.append(out["pressure_pred"].cpu())

        p_stack = torch.stack(p_samples, dim=0)  # (n_mc, N_total)
        p_std = p_stack.std(dim=0).numpy()        # (N_total,)
        p_mean = p_stack.mean(dim=0)
        p_error = torch.abs(p_mean - batch.y_pressure.cpu()).numpy()

        # Decompose batched nodes back to per-graph nodes
        batch_vec = batch.batch.cpu().numpy() if hasattr(batch, 'batch') and batch.batch is not None else np.zeros(len(p_std), dtype=int)

        for g in range(batch_size):
            node_mask = (batch_vec == g)
            g_std = p_std[node_mask]
            g_error = p_error[node_mask]
            n = len(g_std)

            # Map to canonical node indices (0..num_nodes-1)
            for i in range(n):
                node_idx = i  # within a single graph, nodes are 0..N-1
                node_std_sum[node_idx] += g_std[i]
                node_std_max[node_idx] = max(node_std_max[node_idx], g_std[i])
                node_error_sum[node_idx] += g_error[i]
                node_count[node_idx] += 1

            per_snapshot_std.append(g_std)

    # Average
    safe_count = np.maximum(node_count, 1)
    return {
        "mean_uncertainty": node_std_sum / safe_count,
        "max_uncertainty": node_std_max,
        "mean_error": node_error_sum / safe_count,
        "count": node_count,
        "per_snapshot_std": per_snapshot_std,
    }


# ---------------------------------------------------------------------------
# Greedy sensor placement
# ---------------------------------------------------------------------------

def greedy_sensor_placement(
    model: MultiTaskGNN,
    test_snapshots: list,
    test_corrupted: list,
    normalizer: Normalizer,
    graph: WDNGraph,
    device: torch.device,
    max_sensors: int | None = None,
    n_mc_samples: int = 30,
    batch_size: int = 8,
) -> dict:
    """Greedy sensor placement: iteratively add sensors where uncertainty is highest.

    At each step:
        1. Compute per-node uncertainty with current sensor configuration
        2. Pick the unmonitored node with highest mean uncertainty
        3. "Install" sensor: set pressure_mask=1, pressure_obs=true_value for that node
        4. Measure total uncertainty reduction

    Args:
        model: Trained MultiTaskGNN.
        test_snapshots: Test set snapshots (clean).
        test_corrupted: Test set corrupted snapshots (will be modified in-place copies).
        normalizer: Data normalizer.
        graph: WDN graph structure.
        device: Compute device.
        max_sensors: How many sensors to place (default: all unmonitored nodes).
        n_mc_samples: MC Dropout samples per evaluation.
        batch_size: Batch size for evaluation.

    Returns:
        Dict with placement_order, uncertainty_curve, error_curve, node_names.
    """
    num_nodes = graph.num_nodes
    node_names = graph.node_names

    # Deep-copy corrupted snapshots (we'll modify masks)
    import copy
    working_corrupted = [copy.deepcopy(c) for c in test_corrupted]

    # Track which nodes already have sensors (observed in majority of snapshots)
    initial_obs_rate = np.zeros(num_nodes, dtype=np.float64)
    for c in working_corrupted:
        initial_obs_rate += c.pressure_mask.numpy()
    initial_obs_rate /= len(working_corrupted)

    # Nodes that are "mostly unmonitored" (obs rate < 0.6)
    unmonitored = set(np.where(initial_obs_rate < 0.6)[0].tolist())

    if max_sensors is None:
        max_sensors = len(unmonitored)
    max_sensors = min(max_sensors, len(unmonitored))

    print(f"\nSensor Oracle: {len(unmonitored)} unmonitored nodes (of {num_nodes})")
    print(f"Will greedily place up to {max_sensors} sensors\n")

    # --- Baseline: compute uncertainty with current sensor config ---
    def _eval_uncertainty(corrupted_list):
        ds = WDNDataset(test_snapshots, corrupted_list, normalizer)
        loader = PyGDataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        return compute_node_uncertainty(model, loader, device, num_nodes, n_mc_samples)

    baseline_unc = _eval_uncertainty(working_corrupted)
    baseline_total = float(baseline_unc["mean_uncertainty"].sum())
    baseline_mean = float(baseline_unc["mean_uncertainty"].mean())

    print(f"Baseline total uncertainty: {baseline_total:.4f}")
    print(f"Baseline mean uncertainty:  {baseline_mean:.6f}")

    # Also compute baseline reconstruction error
    ds = WDNDataset(test_snapshots, working_corrupted, normalizer)
    loader = PyGDataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    placement_order = []
    uncertainty_curve = [baseline_total]
    mean_uncertainty_curve = [baseline_mean]
    error_curve = [float(baseline_unc["mean_error"].mean())]
    node_details = []

    # --- Greedy loop ---
    for step in range(max_sensors):
        t0 = time.time()

        # Get current uncertainty for unmonitored nodes only
        current_unc = baseline_unc if step == 0 else _eval_uncertainty(working_corrupted)

        # Find best candidate: highest mean uncertainty among unmonitored
        best_node = None
        best_unc = -1.0
        for node_idx in unmonitored:
            if current_unc["mean_uncertainty"][node_idx] > best_unc:
                best_unc = current_unc["mean_uncertainty"][node_idx]
                best_node = node_idx

        if best_node is None:
            break

        # "Install" sensor at best_node: set mask=1, obs=true for all snapshots
        for i, (snap, corr) in enumerate(zip(test_snapshots, working_corrupted)):
            corr.pressure_mask[best_node] = 1.0
            # Set observed value to the RAW true value (dataset handles normalization)
            corr.pressure_obs[best_node] = snap.pressure_true[best_node].item()

        unmonitored.discard(best_node)

        # Evaluate new uncertainty
        new_unc = _eval_uncertainty(working_corrupted)
        new_total = float(new_unc["mean_uncertainty"].sum())
        new_mean = float(new_unc["mean_uncertainty"].mean())
        new_error = float(new_unc["mean_error"].mean())

        reduction = uncertainty_curve[-1] - new_total
        reduction_pct = (reduction / uncertainty_curve[-1]) * 100 if uncertainty_curve[-1] > 0 else 0

        elapsed = time.time() - t0

        detail = {
            "step": step + 1,
            "node_index": int(best_node),
            "node_name": node_names[best_node],
            "node_type": "junction" if graph.node_types[best_node] == 0
                        else "reservoir" if graph.node_types[best_node] == 1
                        else "tank",
            "uncertainty_before_placement": float(best_unc),
            "total_uncertainty_after": new_total,
            "mean_uncertainty_after": new_mean,
            "uncertainty_reduction": float(reduction),
            "uncertainty_reduction_pct": float(reduction_pct),
            "mean_recon_error_after": new_error,
        }
        node_details.append(detail)

        placement_order.append(int(best_node))
        uncertainty_curve.append(new_total)
        mean_uncertainty_curve.append(new_mean)
        error_curve.append(new_error)

        print(
            f"  Step {step+1}/{max_sensors} ({elapsed:.1f}s): "
            f"Place sensor at {node_names[best_node]:>5s} (idx={best_node}) "
            f"| Unc: {best_unc:.5f} → total {new_total:.4f} "
            f"(↓{reduction_pct:.1f}%) | Error: {new_error:.4f}"
        )

        # Update baseline_unc for next iteration
        baseline_unc = new_unc

    # Summary
    total_reduction = uncertainty_curve[0] - uncertainty_curve[-1]
    total_reduction_pct = (total_reduction / uncertainty_curve[0]) * 100 if uncertainty_curve[0] > 0 else 0

    print(f"\n{'='*70}")
    print(f"SENSOR ORACLE SUMMARY")
    print(f"{'='*70}")
    print(f"  Sensors placed: {len(placement_order)}")
    print(f"  Total uncertainty: {uncertainty_curve[0]:.4f} → {uncertainty_curve[-1]:.4f} "
          f"(↓{total_reduction_pct:.1f}%)")
    print(f"  Mean recon error: {error_curve[0]:.4f} → {error_curve[-1]:.4f}")
    print(f"\n  Placement order:")
    for d in node_details:
        print(f"    {d['step']}. {d['node_name']} ({d['node_type']}) "
              f"— uncertainty {d['uncertainty_before_placement']:.5f}, "
              f"reduction {d['uncertainty_reduction_pct']:.1f}%")

    return {
        "placement_order": placement_order,
        "placement_details": node_details,
        "uncertainty_curve": uncertainty_curve,
        "mean_uncertainty_curve": mean_uncertainty_curve,
        "error_curve": error_curve,
        "baseline_total_uncertainty": uncertainty_curve[0],
        "final_total_uncertainty": uncertainty_curve[-1],
        "total_reduction_pct": total_reduction_pct,
        "initial_observation_rates": initial_obs_rate.tolist(),
        "node_names": node_names,
    }


# ---------------------------------------------------------------------------
# Node importance ranking (non-greedy, fast)
# ---------------------------------------------------------------------------

def rank_nodes_by_uncertainty(
    model: MultiTaskGNN,
    test_snapshots: list,
    test_corrupted: list,
    normalizer: Normalizer,
    graph: WDNGraph,
    device: torch.device,
    n_mc_samples: int = 30,
    batch_size: int = 8,
) -> dict:
    """One-shot ranking: compute uncertainty once and rank all nodes.

    This is much faster than greedy placement but doesn't account for
    interactions between sensors (greedy is more accurate).

    Returns:
        Dict with ranking, uncertainties, and per-node details.
    """
    num_nodes = graph.num_nodes

    ds = WDNDataset(test_snapshots, test_corrupted, normalizer)
    loader = PyGDataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    unc = compute_node_uncertainty(model, loader, device, num_nodes, n_mc_samples)

    # Compute observation rate per node
    obs_rate = np.zeros(num_nodes, dtype=np.float64)
    for c in test_corrupted:
        obs_rate += c.pressure_mask.numpy()
    obs_rate /= len(test_corrupted)

    # Build ranking (sort by uncertainty descending)
    ranking = np.argsort(-unc["mean_uncertainty"])

    node_info = []
    for rank, idx in enumerate(ranking):
        node_info.append({
            "rank": rank + 1,
            "node_index": int(idx),
            "node_name": graph.node_names[idx],
            "node_type": "junction" if graph.node_types[idx] == 0
                        else "reservoir" if graph.node_types[idx] == 1
                        else "tank",
            "mean_uncertainty": float(unc["mean_uncertainty"][idx]),
            "max_uncertainty": float(unc["max_uncertainty"][idx]),
            "mean_error": float(unc["mean_error"][idx]),
            "observation_rate": float(obs_rate[idx]),
        })

    print(f"\n{'='*70}")
    print(f"NODE UNCERTAINTY RANKING (one-shot)")
    print(f"{'='*70}")
    print(f"{'Rank':>4s}  {'Node':>6s}  {'Type':>10s}  {'Obs Rate':>8s}  "
          f"{'Mean Unc':>9s}  {'Max Unc':>9s}  {'Mean Err':>9s}")
    print("-" * 70)
    for info in node_info[:num_nodes]:  # show all
        print(f"  {info['rank']:>2d}   {info['node_name']:>6s}  "
              f"{info['node_type']:>10s}  {info['observation_rate']:>8.1%}  "
              f"{info['mean_uncertainty']:>9.5f}  {info['max_uncertainty']:>9.5f}  "
              f"{info['mean_error']:>9.4f}")

    return {
        "ranking": node_info,
        "mean_uncertainties": unc["mean_uncertainty"].tolist(),
        "max_uncertainties": unc["max_uncertainty"].tolist(),
        "mean_errors": unc["mean_error"].tolist(),
        "observation_rates": obs_rate.tolist(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sensor Oracle: optimal sensor placement")
    parser.add_argument("--model_dir", type=str, default="runs/multitask/20260310_201113",
                        help="Directory with trained model (best_model.pt, normalizer.pt)")
    parser.add_argument("--data_dir", type=str, default="data/generated_attacks",
                        help="Directory with data (graph.pkl, snapshots.pkl, corrupted.pkl)")
    parser.add_argument("--gnn_type", type=str, default="GraphSAGE",
                        help="GNN architecture (must match trained model)")
    parser.add_argument("--n_mc_samples", type=int, default=30,
                        help="Number of MC Dropout samples")
    parser.add_argument("--max_sensors", type=int, default=None,
                        help="Max sensors to place (default: all unmonitored)")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["ranking", "greedy", "both"],
                        help="Run mode: fast ranking, greedy placement, or both")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # --- Load data ---
    data_dir = Path(args.data_dir)
    with open(data_dir / "graph.pkl", "rb") as f:
        graph = pickle.load(f)
    with open(data_dir / "snapshots.pkl", "rb") as f:
        snapshots = pickle.load(f)
    with open(data_dir / "corrupted.pkl", "rb") as f:
        corrupted = pickle.load(f)

    print(f"Loaded {len(snapshots)} snapshots ({graph.num_nodes} nodes, {graph.num_edges} edges)")

    # Split (same split as training!)
    train_s, train_c, val_s, val_c, test_s, test_c = train_val_test_split(
        snapshots, corrupted, 0.70, 0.15, seed=42,
    )

    # Fit normalizer on training data
    normalizer = Normalizer()
    normalizer.fit(train_s)

    # Load saved normalizer if available (to match training exactly)
    model_dir = Path(args.model_dir)
    norm_path = model_dir / "normalizer.pt"
    if norm_path.exists():
        norm_state = torch.load(norm_path, map_location="cpu", weights_only=True)
        normalizer.load_state_dict(norm_state)
        print("Loaded saved normalizer")

    # --- Load model ---
    sample_ds = WDNDataset(test_s[:1], test_c[:1], normalizer)
    sample = sample_ds[0]

    model = MultiTaskGNN(
        node_in_dim=sample.x.shape[1],
        edge_in_dim=sample.edge_attr.shape[1],
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
        gnn_type=args.gnn_type,
        heads=4,
    ).to(device)

    state = torch.load(model_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"Loaded model from {model_dir}")

    # --- Output directory ---
    output_dir = Path("runs/sensor_oracle") / time.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Run sensor oracle ---
    results = {}

    if args.mode in ("ranking", "both"):
        print(f"\n{'='*70}")
        print("PHASE 1: One-shot uncertainty ranking")
        print(f"{'='*70}")
        ranking_result = rank_nodes_by_uncertainty(
            model, test_s, test_c, normalizer, graph, device,
            n_mc_samples=args.n_mc_samples,
        )
        results["ranking"] = ranking_result

        with open(output_dir / "node_ranking.json", "w") as f:
            json.dump(ranking_result, f, indent=2)

    if args.mode in ("greedy", "both"):
        print(f"\n{'='*70}")
        print("PHASE 2: Greedy sensor placement")
        print(f"{'='*70}")
        greedy_result = greedy_sensor_placement(
            model, test_s, test_c, normalizer, graph, device,
            max_sensors=args.max_sensors,
            n_mc_samples=args.n_mc_samples,
        )
        results["greedy"] = greedy_result

        with open(output_dir / "greedy_placement.json", "w") as f:
            json.dump(greedy_result, f, indent=2)

    # Save combined results
    with open(output_dir / "sensor_oracle_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save config for reproducibility
    config = {
        "model_dir": str(model_dir),
        "data_dir": str(data_dir),
        "gnn_type": args.gnn_type,
        "n_mc_samples": args.n_mc_samples,
        "max_sensors": args.max_sensors,
        "mode": args.mode,
        "num_test_snapshots": len(test_s),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    return results


if __name__ == "__main__":
    main()
