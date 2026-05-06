"""Evaluate a self-play defender vs the original pretrained defender
on the hand-crafted 5-attack test split.

Usage:
    python scripts/eval_selfplay.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, "src")

from wdn.metrics import compute_anomaly_metrics, compute_recon_metrics
from wdn.models.temporal_moe import TemporalMixtureOfExpertsGNN
from wdn.temporal_dataset import create_temporal_dataloaders


PRETRAINED = "runs/temporal_moe/20260505_144409/best_model.pt"
SELFPLAY = "runs/selfplay/20260505_223529/defender.pt"
DATA_DIR = Path("data/temporal_moe_modena")


def evaluate_on_test(state_dict_path: str, label: str):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    with open(DATA_DIR / "graph.pkl", "rb") as f:
        graph = pickle.load(f)
    with open(DATA_DIR / "snapshots.pkl", "rb") as f:
        snaps = pickle.load(f)
    with open(DATA_DIR / "corrupted.pkl", "rb") as f:
        corr = pickle.load(f)

    n = len(snaps)
    n_train = int(0.7 * n); n_val = int(0.15 * n)
    _, _, test_loader, _ = create_temporal_dataloaders(
        snaps[:n_train], corr[:n_train],
        snaps[n_train:n_train + n_val], corr[n_train:n_train + n_val],
        snaps[n_train + n_val:], corr[n_train + n_val:],
        window_size=6, batch_size=8,
    )

    model = TemporalMixtureOfExpertsGNN(
        node_in_dim=7, edge_in_dim=8, hidden_dim=48,
        num_experts=6, window_size=6, gnn_type="GraphSAGE",
    ).to(device)
    model.load_state_dict(torch.load(state_dict_path, map_location=device))
    model.eval()

    p_logits_all, p_lab_all, p_mask_all = [], [], []
    p_pred_all, p_true_all = [], []
    attack_ids = []

    # Per-attack groupings need attack_type per snapshot. The collate
    # already provides "attack_type" per graph in the batch.
    per_attack_logits = {i: [] for i in range(6)}
    per_attack_labels = {i: [] for i in range(6)}

    with torch.no_grad():
        for raw in test_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor)
                    else [t.to(device) for t in v] if isinstance(v, list) else v)
                for k, v in raw.items()
            }
            out = model(
                x_seq=batch["x_seq"], edge_index=batch["edge_index"],
                edge_attr=batch["edge_attr"],
                is_original_edge=batch["is_original_edge"],
                batch_size=batch["batch_size"],
                num_nodes_per_graph=batch["num_nodes"],
                pressure_obs=batch["pressure_obs"], flow_obs=batch["flow_obs"],
                pressure_mask=batch["pressure_mask"],
                flow_mask=batch["flow_mask"],
            )
            m = batch["pressure_mask"] > 0
            p_logits_all.append(out["pressure_anomaly_logits"][m].cpu())
            p_lab_all.append(batch["pressure_anomaly"][m].cpu())
            p_pred_all.append(out["pressure_pred"].cpu())
            p_true_all.append(batch["y_pressure"].cpu())
            p_mask_all.append(batch["pressure_mask"].cpu())

            B = batch["batch_size"]; N = batch["num_nodes"]
            atk = batch["attack_type"].cpu()
            logits_view = out["pressure_anomaly_logits"].view(B, N).cpu()
            lab_view = batch["pressure_anomaly"].view(B, N).cpu()
            mask_view = batch["pressure_mask"].view(B, N).cpu()
            for b_idx in range(B):
                cls = int(atk[b_idx])
                msel = mask_view[b_idx] > 0
                if msel.sum() == 0:
                    continue
                per_attack_logits[cls].append(logits_view[b_idx][msel])
                per_attack_labels[cls].append(lab_view[b_idx][msel])

    logits = torch.cat(p_logits_all)
    labels = torch.cat(p_lab_all)
    pred_bin = (logits > 0).long()
    lab_bin = (labels > 0.5).long()
    overall = compute_anomaly_metrics(
        pred_bin, lab_bin, scores=torch.sigmoid(logits),
    )
    recon = compute_recon_metrics(
        torch.cat(p_pred_all), torch.cat(p_true_all),
        torch.cat(p_mask_all), only_unobserved=True,
    )

    print(f"\n=== {label} ===")
    print(f"Anomaly F1: {overall.f1:.3f}  AUROC: {overall.auroc:.3f}  "
          f"P: {overall.precision:.3f}  R: {overall.recall:.3f}")
    print(f"P MAE (unobs): {recon.mae:.3f}")

    name_of = {0: "clean", 1: "random", 2: "replay",
               3: "stealthy", 4: "noise", 5: "targeted"}
    print("Per-attack F1:")
    out_per = {}
    for cls, ll in per_attack_logits.items():
        if not ll:
            continue
        lo = torch.cat(ll); la = torch.cat(per_attack_labels[cls])
        if la.sum() == 0:
            continue
        m = compute_anomaly_metrics(
            (lo > 0).long(), (la > 0.5).long(), scores=torch.sigmoid(lo),
        )
        out_per[name_of[cls]] = {"f1": m.f1, "auroc": m.auroc}
        print(f"  {name_of[cls]:10s} F1={m.f1:.3f}  AUROC={m.auroc:.3f}")

    return {"overall_f1": overall.f1, "overall_auroc": overall.auroc,
            "p_mae": recon.mae, "per_attack": out_per}


pre = evaluate_on_test(PRETRAINED, "Pretrained Temporal MoE (no self-play)")
sp = evaluate_on_test(SELFPLAY, "Self-play Defender")

# Save comparison
out = {"pretrained": pre, "selfplay": sp}
with open("runs/selfplay/20260505_223529/eval_comparison.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved comparison to runs/selfplay/20260505_211917/eval_comparison.json")
