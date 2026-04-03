"""Pre-compute per-attack-type analysis data for the Modena network dashboard.

Generates each attack type separately on test snapshots and records
the effect on node readings and model detection performance.

Run once before launching the dashboard:
    python dashboard/precompute/export_attack_analysis_modena.py
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

from wdn.corruption import corrupt_snapshot
from wdn.config import CorruptionConfig
from wdn.dataset import WDNDataset, Normalizer, train_val_test_split
from wdn.models.multitask import MultiTaskGNN


ATTACK_TYPES = ["random", "replay", "stealthy", "noise", "targeted"]

ATTACK_LABELS = {
    "random": "Random Falsification",
    "replay": "Replay Attack",
    "stealthy": "Stealthy Bias Injection",
    "noise": "Noise Injection",
    "targeted": "Targeted Attack",
}

ATTACK_DESCRIPTIONS = {
    "random": "Scales and offsets sensor readings with arbitrary values",
    "replay": "Replays legitimate past readings to mask tampering",
    "stealthy": "Injects a slowly increasing bias that's hard to detect",
    "noise": "Adds large Gaussian noise simulating sensor malfunction",
    "targeted": "Attacks the highest-impact sensors in the network",
}


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
    data_dir = PROJECT_ROOT / "data" / "modena_attacks"
    with open(data_dir / "graph.pkl", "rb") as f:
        graph = pickle.load(f)
    with open(data_dir / "snapshots.pkl", "rb") as f:
        snapshots = pickle.load(f)
    with open(data_dir / "corrupted.pkl", "rb") as f:
        corrupted = pickle.load(f)

    print(f"Loaded {len(snapshots)} snapshots ({graph.num_nodes} nodes)")

    # Split
    train_s, train_c, val_s, val_c, test_s, test_c = train_val_test_split(
        snapshots, corrupted, 0.70, 0.15, seed=42,
    )

    # Normalizer
    normalizer = Normalizer()
    norm_path = PROJECT_ROOT / "runs" / "multitask" / "20260403_234741" / "normalizer.pt"
    norm_state = torch.load(norm_path, map_location="cpu", weights_only=True)
    normalizer.load_state_dict(norm_state)

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
        PROJECT_ROOT / "runs" / "multitask" / "20260403_234741" / "best_model.pt",
        map_location=device, weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()
    print("Model loaded")

    N = graph.num_nodes
    n_test = len(test_s)
    # Use a subset for efficiency
    n_samples = min(50, n_test)
    sample_indices = np.linspace(0, n_test - 1, n_samples, dtype=int)

    rng = np.random.default_rng(123)

    # Sweep attack fractions
    fractions = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

    results = {}

    for attack_type in ATTACK_TYPES:
        print(f"\n{'='*50}")
        print(f"Attack type: {ATTACK_LABELS[attack_type]}")
        print(f"{'='*50}")

        attack_results = {
            "label": ATTACK_LABELS[attack_type],
            "description": ATTACK_DESCRIPTIONS[attack_type],
            "fractions": fractions,
            "fraction_data": [],
            # Per-node analysis at 15% fraction
            "node_analysis": None,
        }

        for frac in fractions:
            cfg = CorruptionConfig(
                missing_rate_pressure=0.5,
                missing_rate_flow=0.5,
                noise_sigma_pressure=0.5,
                noise_sigma_flow=0.2,
                attack_enabled=True,
                attack_fraction=frac,
                attack_bias=3.0,
                attack_scale=1.5,
                attack_type=attack_type,
            )

            total_nodes = 0
            total_attacked = 0
            total_detected = 0
            total_true_positive = 0
            total_false_positive = 0
            total_false_negative = 0
            pressure_deviations = []
            node_attack_counts = np.zeros(N)
            node_deviation_sums = np.zeros(N)

            replay_buffer = None

            for idx in sample_indices:
                snap = test_s[idx]
                c = corrupt_snapshot(
                    snap.pressure_true, snap.flow_true,
                    cfg, np.random.default_rng(rng.integers(0, 2**31)),
                    replay_buffer=replay_buffer,
                    snapshot_idx=int(idx),
                )

                # Update replay buffer
                if attack_type in ("replay", "mixed"):
                    replay_buffer = {
                        "pressure": snap.pressure_true.clone(),
                        "flow": snap.flow_true.clone(),
                    }

                # Count attacked nodes
                p_anom = c.pressure_anomaly.numpy()
                n_attacked = int(p_anom.sum())
                total_nodes += int(c.pressure_mask.sum())
                total_attacked += n_attacked

                # Pressure deviation at attacked nodes
                p_true = snap.pressure_true.numpy()
                p_obs = c.pressure_obs.numpy()
                p_mask = c.pressure_mask.numpy()

                for i in range(N):
                    if p_anom[i] > 0 and p_mask[i] > 0:
                        dev = abs(p_obs[i] - p_true[i])
                        pressure_deviations.append(dev)
                        node_attack_counts[i] += 1
                        node_deviation_sums[i] += dev

                # Run model for detection
                ds_single = WDNDataset([snap], [c], normalizer)
                data = ds_single[0].to(device)

                with torch.no_grad():
                    out = model(
                        x=data.x, edge_index=data.edge_index,
                        edge_attr=data.edge_attr,
                        is_original_edge=data.is_original_edge,
                        pressure_obs=data.pressure_obs,
                        flow_obs=data.flow_obs,
                        pressure_mask=data.pressure_mask,
                        flow_mask=data.flow_mask,
                    )

                if "pressure_anomaly_logits" in out:
                    p_prob = torch.sigmoid(out["pressure_anomaly_logits"]).cpu().numpy()
                    threshold = 0.5
                    p_det = (p_prob > threshold).astype(float)

                    for i in range(N):
                        if p_mask[i] > 0:
                            if p_anom[i] > 0 and p_det[i] > 0:
                                total_true_positive += 1
                            elif p_anom[i] == 0 and p_det[i] > 0:
                                total_false_positive += 1
                            elif p_anom[i] > 0 and p_det[i] == 0:
                                total_false_negative += 1

            # Compute metrics for this fraction
            precision = total_true_positive / max(total_true_positive + total_false_positive, 1)
            recall = total_true_positive / max(total_true_positive + total_false_negative, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            pct_affected = total_attacked / max(total_nodes, 1) * 100
            mean_deviation = float(np.mean(pressure_deviations)) if pressure_deviations else 0
            max_deviation = float(np.max(pressure_deviations)) if pressure_deviations else 0

            frac_entry = {
                "fraction": frac,
                "pct_affected": round(pct_affected, 2),
                "n_attacked_total": total_attacked,
                "n_observed_total": total_nodes,
                "mean_deviation_m": round(mean_deviation, 3),
                "max_deviation_m": round(max_deviation, 3),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "true_positives": total_true_positive,
                "false_positives": total_false_positive,
                "false_negatives": total_false_negative,
            }
            attack_results["fraction_data"].append(frac_entry)

            print(f"  frac={frac:.0%}: affected={pct_affected:.1f}%, "
                  f"dev={mean_deviation:.2f}m, P={precision:.2f}, R={recall:.2f}, F1={f1:.3f}")

        # Node-level analysis at 15% attack fraction
        node_analysis = []
        for i in range(N):
            mean_dev = node_deviation_sums[i] / max(node_attack_counts[i], 1)
            node_analysis.append({
                "node_name": graph.node_names[i],
                "times_attacked": int(node_attack_counts[i]),
                "mean_deviation": round(float(mean_dev), 3),
            })
        attack_results["node_analysis"] = node_analysis

        results[attack_type] = attack_results

    # Save
    out_path = PROJECT_ROOT / "dashboard" / "data" / "attack_analysis_modena.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "node_names": graph.node_names,
            "attack_types": ATTACK_TYPES,
            "results": results,
        }, f, indent=2)

    print(f"\nSaved attack analysis to {out_path}")


if __name__ == "__main__":
    main()
