"""Evaluate self-play defenders for Net1, Net3, Modena vs their
pretrained baselines on the hand-crafted 5-attack test split.
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


CONFIGS = {
    "Net1":   {"data_dir": "data/moe_net1",
               "hidden_dim": 32,
               "pretrained": "runs/temporal_moe/20260505_145514/best_model.pt",
               "selfplay":   "runs/selfplay/20260505_233503/defender.pt"},
    "Net3":   {"data_dir": "data/temporal_moe_net3",
               "hidden_dim": 48,
               "pretrained": "runs/temporal_moe/20260505_150656/best_model.pt",
               "selfplay":   "runs/selfplay/20260505_232120/defender.pt"},
    "Modena": {"data_dir": "data/temporal_moe_modena",
               "hidden_dim": 48,
               "pretrained": "runs/temporal_moe/20260505_144409/best_model.pt",
               "selfplay":   "runs/selfplay/20260505_223529/defender.pt"},
}


def evaluate_one(state_dict_path, data_dir, hidden_dim):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data_dir = Path(data_dir)
    snaps = pickle.load(open(data_dir / "snapshots.pkl", "rb"))
    corr = pickle.load(open(data_dir / "corrupted.pkl", "rb"))
    n = len(snaps); n_train = int(0.7 * n); n_val = int(0.15 * n)
    _, _, test_loader, _ = create_temporal_dataloaders(
        snaps[:n_train], corr[:n_train],
        snaps[n_train:n_train+n_val], corr[n_train:n_train+n_val],
        snaps[n_train+n_val:], corr[n_train+n_val:],
        window_size=6, batch_size=8,
    )
    m = TemporalMixtureOfExpertsGNN(
        node_in_dim=7, edge_in_dim=8, hidden_dim=hidden_dim,
        num_experts=6, window_size=6, gnn_type="GraphSAGE",
    ).to(device)
    m.load_state_dict(torch.load(state_dict_path, map_location=device))
    m.eval()

    p_logits, p_lab, p_pred, p_true, p_mask = [], [], [], [], []
    per_atk = {i: [[], []] for i in range(6)}

    with torch.no_grad():
        for raw in test_loader:
            b = {k: (v.to(device) if isinstance(v, torch.Tensor)
                     else [t.to(device) for t in v] if isinstance(v, list) else v)
                 for k, v in raw.items()}
            o = m(x_seq=b["x_seq"], edge_index=b["edge_index"],
                  edge_attr=b["edge_attr"],
                  is_original_edge=b["is_original_edge"],
                  batch_size=b["batch_size"],
                  num_nodes_per_graph=b["num_nodes"],
                  pressure_obs=b["pressure_obs"], flow_obs=b["flow_obs"],
                  pressure_mask=b["pressure_mask"],
                  flow_mask=b["flow_mask"])
            sel = b["pressure_mask"] > 0
            p_logits.append(o["pressure_anomaly_logits"][sel].cpu())
            p_lab.append(b["pressure_anomaly"][sel].cpu())
            p_pred.append(o["pressure_pred"].cpu())
            p_true.append(b["y_pressure"].cpu())
            p_mask.append(b["pressure_mask"].cpu())
            B, N = b["batch_size"], b["num_nodes"]
            atk = b["attack_type"].cpu()
            lv = o["pressure_anomaly_logits"].view(B, N).cpu()
            la = b["pressure_anomaly"].view(B, N).cpu()
            ma = b["pressure_mask"].view(B, N).cpu()
            for i in range(B):
                cls = int(atk[i]); s = ma[i] > 0
                if s.sum() == 0: continue
                per_atk[cls][0].append(lv[i][s])
                per_atk[cls][1].append(la[i][s])

    L = torch.cat(p_logits); LA = torch.cat(p_lab)
    overall = compute_anomaly_metrics(
        (L > 0).long(), (LA > 0.5).long(), scores=torch.sigmoid(L))
    rec = compute_recon_metrics(
        torch.cat(p_pred), torch.cat(p_true),
        torch.cat(p_mask), only_unobserved=True)
    name = {0:"clean", 1:"random", 2:"replay", 3:"stealthy", 4:"noise", 5:"targeted"}
    out_per = {}
    for cls, (ll, la) in per_atk.items():
        if not ll: continue
        lo = torch.cat(ll); lab = torch.cat(la)
        if lab.sum() == 0: continue
        mm = compute_anomaly_metrics(
            (lo > 0).long(), (lab > 0.5).long(), scores=torch.sigmoid(lo))
        out_per[name[cls]] = {"f1": mm.f1, "auroc": mm.auroc}
    return {"f1": overall.f1, "auroc": overall.auroc,
            "p_mae": rec.mae, "per_attack": out_per}


def fmt(x):
    return "  -" if x is None else f"{x:.3f}"


results = {}
for net, cfg in CONFIGS.items():
    print(f"\n=== {net} ===")
    pre = evaluate_one(cfg["pretrained"], cfg["data_dir"], cfg["hidden_dim"])
    sp = evaluate_one(cfg["selfplay"], cfg["data_dir"], cfg["hidden_dim"])
    results[net] = {"pretrained": pre, "selfplay": sp}
    attacks = ["random", "replay", "stealthy", "noise", "targeted"]
    print(f"  Overall:  F1 {pre['f1']:.3f} -> {sp['f1']:.3f}  |  "
          f"AUROC {pre['auroc']:.3f} -> {sp['auroc']:.3f}  |  "
          f"P MAE {pre['p_mae']:.3f} -> {sp['p_mae']:.3f}")
    for a in attacks:
        pf = pre['per_attack'].get(a, {}).get('f1')
        sf = sp['per_attack'].get(a, {}).get('f1')
        if pf is None or sf is None:
            continue
        delta = sf - pf
        marker = "+" if delta > 0.01 else ("-" if delta < -0.01 else " ")
        print(f"    {a:10s} F1 {pf:.3f} -> {sf:.3f}   {marker}{abs(delta):.3f}")

with open("runs/selfplay/eval_all.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved aggregate results to runs/selfplay/eval_all.json")
