"""Cross-network transfer: does a self-play attacker / defender pair
trained on one network still help on a different one?

We re-use the self-play attacker from the Modena run, project it onto
each test network's input shape (they share `node_in_dim` and
`edge_in_dim`), and evaluate the trained defender of each network on
attacker-generated perturbations. The Modena-trained attacker has no
knowledge of Net3 or Net1 topology — yet if the framework is
generalisable, defenders that were themselves self-played should
still be more robust than the supervised baselines.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from wdn.metrics import compute_anomaly_metrics
from wdn.models.attacker import AttackerGNN, StealthBudget, apply_stealth_budget
from wdn.models.temporal_moe import TemporalMixtureOfExpertsGNN
from wdn.temporal_dataset import create_temporal_dataloaders


NETS = {
    "Net1": {
        "data_dir": "data/moe_net1", "hidden_dim": 32,
        "pretrained": "runs/temporal_moe/20260505_145514/best_model.pt",
        "selfplay":   "runs/selfplay/20260505_233503/defender.pt",
    },
    "Net3": {
        "data_dir": "data/temporal_moe_net3", "hidden_dim": 48,
        "pretrained": "runs/temporal_moe/20260505_150656/best_model.pt",
        "selfplay":   "runs/selfplay/20260505_232120/defender.pt",
    },
    "Modena": {
        "data_dir": "data/temporal_moe_modena", "hidden_dim": 48,
        "pretrained": "runs/temporal_moe/20260505_144409/best_model.pt",
        "selfplay":   "runs/selfplay/20260505_223529/defender.pt",
    },
}
SOURCE_ATTACKER = ROOT / "runs/selfplay/20260505_223529/attacker.pt"
SOURCE_NETWORK = "Modena"


def make_loader(data_dir):
    snaps = pickle.load(open(data_dir / "snapshots.pkl", "rb"))
    corr = pickle.load(open(data_dir / "corrupted.pkl", "rb"))
    n = len(snaps); nt = int(0.7 * n); nv = int(0.15 * n)
    _, _, tl, _ = create_temporal_dataloaders(
        snaps[:nt], corr[:nt], snaps[nt:nt+nv], corr[nt:nt+nv],
        snaps[nt+nv:], corr[nt+nv:], window_size=6, batch_size=8,
    )
    return tl


def inject(b, proj):
    new_x = list(b["x_seq"])
    last = new_x[-1].clone()
    last[:, 5] = last[:, 5] + proj["delta_p"]
    last[:, 6] = torch.clamp(last[:, 6] + proj["mask_p"], max=1.0)
    new_x[-1] = last
    p_att = (proj["delta_p"].abs() > 1e-6).float()
    p_anom = torch.clamp(b["pressure_anomaly"] + p_att, max=1.0)
    return {**b, "x_seq": new_x,
            "pressure_obs": last[:, 5], "pressure_mask": last[:, 6],
            "pressure_anomaly": p_anom}


@torch.no_grad()
def evaluate(defender, attacker, loader, device, budget):
    defender.eval(); attacker.eval()
    L, La = [], []
    for raw in loader:
        b = {k: (v.to(device) if isinstance(v, torch.Tensor)
                 else [t.to(device) for t in v] if isinstance(v, list) else v)
             for k, v in raw.items()}
        x_last = b["x_seq"][-1]
        out = attacker(x_last, b["edge_index"], b["edge_attr"],
                       b["is_original_edge"])
        proj = apply_stealth_budget(out, budget, hard=True)
        b2 = inject(b, proj)
        o = defender(x_seq=b2["x_seq"], edge_index=b2["edge_index"],
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
    return {"f1": m.f1, "auroc": m.auroc}


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Source attacker (Modena self-play single).
    attacker = AttackerGNN(node_in_dim=7, edge_in_dim=8, hidden_dim=48).to(device)
    attacker.load_state_dict(torch.load(SOURCE_ATTACKER, map_location=device))

    budget = StealthBudget(epsilon_p=5.0, epsilon_q=0.05, k_p=15, k_q=4)
    out = {}
    print(f"Attacker trained on {SOURCE_NETWORK}, transferred to:")
    print(f"  {'target':10s}  {'pretrained F1':>14s}  {'self-play F1':>14s}  {'lift':>6s}")
    for tgt, cfg in NETS.items():
        data_dir = ROOT / cfg["data_dir"]
        if not (data_dir / "snapshots.pkl").exists():
            continue
        loader = make_loader(data_dir)
        # K cap by network size (Net1 has only 11 nodes).
        k = min(15, max(2, int(0.2 * (loader.dataset[0]["x_seq"][0].shape[0]))))
        bud = StealthBudget(epsilon_p=5.0, epsilon_q=0.05, k_p=k, k_q=4)
        # Pretrained defender for this target
        d_pre = TemporalMixtureOfExpertsGNN(
            node_in_dim=7, edge_in_dim=8, hidden_dim=cfg["hidden_dim"],
            num_experts=6, window_size=6, gnn_type="GraphSAGE",
        ).to(device)
        d_pre.load_state_dict(torch.load(ROOT / cfg["pretrained"], map_location=device))
        # Self-play defender
        d_sp = TemporalMixtureOfExpertsGNN(
            node_in_dim=7, edge_in_dim=8, hidden_dim=cfg["hidden_dim"],
            num_experts=6, window_size=6, gnn_type="GraphSAGE",
        ).to(device)
        d_sp.load_state_dict(torch.load(ROOT / cfg["selfplay"], map_location=device))

        r_pre = evaluate(d_pre, attacker, loader, device, bud)
        r_sp = evaluate(d_sp, attacker, loader, device, bud)
        out[tgt] = {"k": k, "pretrained": r_pre, "selfplay": r_sp}
        print(f"  {tgt:10s}  {r_pre['f1']:>14.3f}  {r_sp['f1']:>14.3f}  "
              f"{r_sp['f1'] - r_pre['f1']:+6.3f}")

    (ROOT / "runs/selfplay/eval_crossnet.json").write_text(
        json.dumps({"source_network": SOURCE_NETWORK, "targets": out}, indent=2)
    )
    print("\nSaved runs/selfplay/eval_crossnet.json")


if __name__ == "__main__":
    main()
