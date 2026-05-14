"""Adversarial robustness curve.

For each epsilon in a sweep, re-create the attacker with that
perturbation budget, project onto k attacked sensors, feed the
corrupted snapshot to each of the three defenders, and record the
anomaly F1 and AUROC. The result is the defender's robust radius —
the largest epsilon at which it still performs well.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from wdn.metrics import compute_anomaly_metrics
from wdn.models.attacker import AttackerGNN, StealthBudget, apply_stealth_budget
from wdn.models.temporal_moe import TemporalMixtureOfExpertsGNN
from wdn.temporal_dataset import create_temporal_dataloaders


DD = ROOT / "data" / "temporal_moe_modena"
DEFENDERS = {
    "Pretrained": ROOT / "runs/temporal_moe/20260505_144409/best_model.pt",
    "Self-play single": ROOT / "runs/selfplay/20260505_223529/defender.pt",
    "Self-play MoE":    ROOT / "runs/selfplay/20260506_110630/defender.pt",
}
# Use the attacker from the Self-play single run for a consistent
# attack distribution; we only vary the stealth budget at evaluation.
ATTACKER_CKPT = ROOT / "runs/selfplay/20260505_223529/attacker.pt"
EPSILONS = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]


def make_loader():
    snaps = pickle.load(open(DD / "snapshots.pkl", "rb"))
    corr = pickle.load(open(DD / "corrupted.pkl", "rb"))
    n = len(snaps); nt = int(0.7 * n); nv = int(0.15 * n)
    _, _, tl, _ = create_temporal_dataloaders(
        snaps[:nt], corr[:nt], snaps[nt:nt+nv], corr[nt:nt+nv],
        snaps[nt+nv:], corr[nt+nv:], window_size=6, batch_size=8,
    )
    return tl


def inject_attack_simple(batch, proj):
    """Apply the projected attack to the last timestep of x_seq + the
    observation tensors. Returns updated tensors plus the new anomaly
    label (union of original + attacker-touched sensors)."""
    new_x_seq = list(batch["x_seq"])
    last = new_x_seq[-1].clone()
    last[:, 5] = last[:, 5] + proj["delta_p"]
    last[:, 6] = torch.clamp(last[:, 6] + proj["mask_p"], max=1.0)
    new_x_seq[-1] = last
    p_attacked = (proj["delta_p"].abs() > 1e-6).float()
    p_anom = torch.clamp(batch["pressure_anomaly"] + p_attacked, max=1.0)
    return {**batch, "x_seq": new_x_seq,
            "pressure_obs": last[:, 5],
            "pressure_mask": last[:, 6],
            "pressure_anomaly": p_anom}


@torch.no_grad()
def evaluate(model, attacker, loader, device, budget):
    model.eval(); attacker.eval()
    L, La = [], []
    for raw in loader:
        b = {k: (v.to(device) if isinstance(v, torch.Tensor)
                 else [t.to(device) for t in v] if isinstance(v, list) else v)
             for k, v in raw.items()}
        x_last = b["x_seq"][-1]
        out = attacker(x_last, b["edge_index"], b["edge_attr"],
                       b["is_original_edge"])
        proj = apply_stealth_budget(out, budget, hard=True)
        b2 = inject_attack_simple(b, proj)
        o = model(x_seq=b2["x_seq"], edge_index=b2["edge_index"],
                  edge_attr=b2["edge_attr"],
                  is_original_edge=b2["is_original_edge"],
                  batch_size=b2["batch_size"],
                  num_nodes_per_graph=b2["num_nodes"],
                  pressure_obs=b2["pressure_obs"],
                  flow_obs=b2["flow_obs"],
                  pressure_mask=b2["pressure_mask"],
                  flow_mask=b2["flow_mask"])
        sel = b2["pressure_mask"] > 0
        L.append(o["pressure_anomaly_logits"][sel].cpu())
        La.append(b2["pressure_anomaly"][sel].cpu())
    Lt = torch.cat(L); Lat = torch.cat(La)
    m = compute_anomaly_metrics(
        (Lt > 0).long(), (Lat > 0.5).long(), scores=torch.sigmoid(Lt))
    return {"f1": m.f1, "auroc": m.auroc,
            "precision": m.precision, "recall": m.recall}


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    loader = make_loader()

    # Load attacker once.
    sample = next(iter(loader))
    node_dim = sample["x_seq"][0].shape[1]
    edge_dim = sample["edge_attr"].shape[1]
    attacker = AttackerGNN(
        node_in_dim=node_dim, edge_in_dim=edge_dim, hidden_dim=48,
    ).to(device)
    attacker.load_state_dict(torch.load(ATTACKER_CKPT, map_location=device))

    # Load each defender.
    defenders = {}
    for name, path in DEFENDERS.items():
        m = TemporalMixtureOfExpertsGNN(
            node_in_dim=node_dim, edge_in_dim=edge_dim, hidden_dim=48,
            num_experts=6, window_size=6, gnn_type="GraphSAGE",
        ).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        defenders[name] = m

    results = {name: {"epsilons": EPSILONS, "f1": [], "auroc": []}
               for name in defenders}

    print(f"{'epsilon':>8s} | " + " ".join(f"{n:>20s}" for n in defenders))
    for eps in EPSILONS:
        budget = StealthBudget(epsilon_p=eps, epsilon_q=0.05, k_p=15, k_q=4)
        line = f"{eps:>8.1f} |"
        for name, m in defenders.items():
            r = evaluate(m, attacker, loader, device, budget)
            results[name]["f1"].append(r["f1"])
            results[name]["auroc"].append(r["auroc"])
            line += f"  F1={r['f1']:.3f} AUR={r['auroc']:.3f}  "
        print(line)

    (ROOT / "runs/selfplay/eval_robustness.json").write_text(
        json.dumps(results, indent=2)
    )
    print("\nSaved runs/selfplay/eval_robustness.json")


if __name__ == "__main__":
    main()
