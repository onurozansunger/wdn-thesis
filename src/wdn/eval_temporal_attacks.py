"""Per-attack-type evaluation for TemporalMultiTaskGNN.

Evaluates the temporal model on each attack type separately to see
if GRU temporal context improves detection of time-dependent attacks
(especially replay attacks).

Usage:
    python -m wdn.eval_temporal_attacks --model_dir runs/temporal/XXXXXX --data_dir data/generated_attacks
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from wdn.corruption import corrupt_snapshot, CorruptedSnapshot
from wdn.config import CorruptionConfig
from wdn.dataset import Normalizer
from wdn.temporal_dataset import TemporalWDNDataset, temporal_collate_fn
from wdn.models.temporal_multitask import TemporalMultiTaskGNN
from wdn.metrics import compute_anomaly_metrics


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate_attack_type(
    model, graph, snapshots, attack_type, fractions,
    normalizer, window_size, device, n_samples=50,
):
    """Evaluate model on a specific attack type at various fractions."""
    rng = np.random.default_rng(42)
    results = []

    for frac in fractions:
        cfg = CorruptionConfig(
            missing_rate_pressure=0.5,
            missing_rate_flow=0.5,
            noise_sigma_pressure=0.5,
            noise_sigma_flow=0.2,
            attack_enabled=True,
            attack_fraction=frac,
            attack_type=attack_type,
            attack_bias=2.0,
            attack_scale=1.0,
        )

        all_p_scores, all_p_labels, all_p_masks = [], [], []

        # We need consecutive snapshots for temporal windows
        # Use scenarios that have enough timesteps
        scenarios = {}
        for i, s in enumerate(snapshots):
            if s.scenario_id not in scenarios:
                scenarios[s.scenario_id] = []
            scenarios[s.scenario_id].append(i)

        n_evaluated = 0
        for sid in sorted(scenarios.keys()):
            if n_evaluated >= n_samples:
                break

            indices = sorted(scenarios[sid], key=lambda i: snapshots[i].timestep)
            if len(indices) < window_size:
                continue

            # Corrupt all snapshots in this scenario
            corrupted_list = []
            replay_buffer = {}
            for j, idx in enumerate(indices):
                snap = snapshots[idx]
                corr = corrupt_snapshot(
                    snap.pressure_true, snap.flow_true,
                    cfg, rng, replay_buffer, j,
                )
                corrupted_list.append(corr)

            # Create windows and evaluate
            for start in range(len(indices) - window_size + 1):
                if n_evaluated >= n_samples:
                    break

                window_snaps = [snapshots[indices[start + t]] for t in range(window_size)]
                window_corrs = corrupted_list[start:start + window_size]

                # Build input sequence
                x_seq = []
                for snap, corr in zip(window_snaps, window_corrs):
                    p_obs = normalizer.normalize_pressure(corr.pressure_obs.clone()) * corr.pressure_mask
                    node_feat = torch.cat([
                        snap.node_static,
                        p_obs.unsqueeze(-1),
                        corr.pressure_mask.unsqueeze(-1),
                    ], dim=-1)
                    x_seq.append(node_feat.to(device))

                last_snap = window_snaps[-1]
                last_corr = window_corrs[-1]

                q_obs = normalizer.normalize_flow(last_corr.flow_obs.clone()) * last_corr.flow_mask
                NE = last_snap.flow_true.shape[0]
                edge_static_bi = last_snap.edge_static[last_snap.edge_map]
                q_obs_bi = q_obs[last_snap.edge_map]
                q_mask_bi = last_corr.flow_mask[last_snap.edge_map]
                edge_attr = torch.cat([
                    edge_static_bi, q_obs_bi.unsqueeze(-1), q_mask_bi.unsqueeze(-1),
                ], dim=-1).to(device)

                is_orig = torch.zeros(last_snap.edge_index.shape[1], dtype=torch.bool)
                is_orig[:NE] = True

                p_obs_last = x_seq[-1][:, 5]

                with torch.no_grad():
                    out = model(
                        x_seq=x_seq,
                        edge_index=last_snap.edge_index.to(device),
                        edge_attr=edge_attr,
                        is_original_edge=is_orig.to(device),
                        pressure_obs=p_obs_last,
                        flow_obs=q_obs.to(device),
                        pressure_mask=last_corr.pressure_mask.to(device),
                        flow_mask=last_corr.flow_mask.to(device),
                    )

                if "pressure_anomaly_logits" in out:
                    scores = torch.sigmoid(out["pressure_anomaly_logits"]).cpu()
                    all_p_scores.append(scores)
                    all_p_labels.append(last_corr.pressure_anomaly)
                    all_p_masks.append(last_corr.pressure_mask)

                n_evaluated += 1

        if all_p_scores:
            scores = torch.cat(all_p_scores)
            labels = torch.cat(all_p_labels)
            masks = torch.cat(all_p_masks)
            preds = (scores > 0.5).float()
            metrics = compute_anomaly_metrics(preds, labels, scores, masks)

            results.append({
                "fraction": frac,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "auroc": metrics.auroc,
                "n_samples": n_evaluated,
            })

    return results


ATTACK_INFO = {
    "random": {"label": "Random Falsification", "description": "Scaled and biased sensor readings"},
    "replay": {"label": "Replay Attack", "description": "Past legitimate readings replayed"},
    "stealthy": {"label": "Stealthy Bias Injection", "description": "Gradual drift over time"},
    "noise": {"label": "Noise Injection", "description": "Large random noise (sensor jamming)"},
    "targeted": {"label": "Targeted Attack", "description": "Attacks high-impact sensors"},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/generated_attacks")
    parser.add_argument("--window_size", type=int, default=6)
    parser.add_argument("--n_samples", type=int, default=50)
    args = parser.parse_args()

    device = get_device()
    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)

    # Load data
    with open(data_dir / "graph.pkl", "rb") as f:
        graph = pickle.load(f)
    with open(data_dir / "snapshots.pkl", "rb") as f:
        snapshots = pickle.load(f)

    # Load model
    normalizer = Normalizer()
    normalizer.load_state_dict(torch.load(model_dir / "normalizer.pt", weights_only=True))

    model = TemporalMultiTaskGNN(
        node_in_dim=7, edge_in_dim=8, hidden_dim=64,
        num_layers=2, num_temporal_layers=1,
        window_size=args.window_size, dropout=0.1,
        gnn_type="GraphSAGE", heads=4,
    ).to(device)
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device, weights_only=True))
    model.eval()

    print(f"Model loaded from {model_dir}")
    print(f"Network: {graph.num_nodes} nodes, {graph.num_edges} edges")
    print(f"Window size: {args.window_size}")

    fractions = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    attack_types = ["random", "replay", "stealthy", "noise", "targeted"]

    all_results = {}
    for atype in attack_types:
        print(f"\nEvaluating {ATTACK_INFO[atype]['label']}...")
        frac_results = evaluate_attack_type(
            model, graph, snapshots, atype, fractions,
            normalizer, args.window_size, device, args.n_samples,
        )
        all_results[atype] = {
            "label": ATTACK_INFO[atype]["label"],
            "description": ATTACK_INFO[atype]["description"],
            "fraction_data": frac_results,
        }

        # Print summary at 15%
        rep = next((d for d in frac_results if d["fraction"] == 0.15), frac_results[2] if len(frac_results) > 2 else None)
        if rep:
            print(f"  @15%: P={rep['precision']:.3f} R={rep['recall']:.3f} F1={rep['f1']:.3f} AUROC={rep['auroc']:.3f}")

    # Save
    output = {
        "model": "TemporalMultiTaskGNN",
        "window_size": args.window_size,
        "attack_types": attack_types,
        "results": all_results,
        "node_names": graph.node_names,
    }

    out_path = model_dir / "attack_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
