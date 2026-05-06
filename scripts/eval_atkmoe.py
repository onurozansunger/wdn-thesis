"""Compare pretrained, single-attacker self-play, and attacker-MoE
self-play defenders on the Modena hand-crafted test split.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from wdn.metrics import compute_anomaly_metrics, compute_recon_metrics
from wdn.models.temporal_moe import TemporalMixtureOfExpertsGNN
from wdn.temporal_dataset import create_temporal_dataloaders


DD = ROOT / "data" / "temporal_moe_modena"
PRETRAINED = ROOT / "runs/temporal_moe/20260505_144409/best_model.pt"
SP_SINGLE = ROOT / "runs/selfplay/20260505_223529/defender.pt"
SP_MOE = ROOT / "runs/selfplay/20260506_110630/defender.pt"


def evaluate(path):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    snaps = pickle.load(open(DD / "snapshots.pkl", "rb"))
    corr = pickle.load(open(DD / "corrupted.pkl", "rb"))
    n = len(snaps); nt = int(0.7 * n); nv = int(0.15 * n)
    _, _, tl, _ = create_temporal_dataloaders(
        snaps[:nt], corr[:nt], snaps[nt:nt+nv], corr[nt:nt+nv],
        snaps[nt+nv:], corr[nt+nv:], window_size=6, batch_size=8,
    )
    m = TemporalMixtureOfExpertsGNN(
        node_in_dim=7, edge_in_dim=8, hidden_dim=48,
        num_experts=6, window_size=6, gnn_type="GraphSAGE",
    ).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()

    L, La, Pp, Pt, Pm = [], [], [], [], []
    pa = {i: [[], []] for i in range(6)}
    with torch.no_grad():
        for raw in tl:
            b = {k: (v.to(device) if isinstance(v, torch.Tensor)
                     else [t.to(device) for t in v] if isinstance(v, list) else v)
                 for k, v in raw.items()}
            o = m(x_seq=b["x_seq"], edge_index=b["edge_index"],
                  edge_attr=b["edge_attr"],
                  is_original_edge=b["is_original_edge"],
                  batch_size=b["batch_size"],
                  num_nodes_per_graph=b["num_nodes"],
                  pressure_obs=b["pressure_obs"],
                  flow_obs=b["flow_obs"],
                  pressure_mask=b["pressure_mask"],
                  flow_mask=b["flow_mask"])
            sel = b["pressure_mask"] > 0
            L.append(o["pressure_anomaly_logits"][sel].cpu())
            La.append(b["pressure_anomaly"][sel].cpu())
            Pp.append(o["pressure_pred"].cpu())
            Pt.append(b["y_pressure"].cpu())
            Pm.append(b["pressure_mask"].cpu())
            B, N = b["batch_size"], b["num_nodes"]
            atk = b["attack_type"].cpu()
            lv = o["pressure_anomaly_logits"].view(B, N).cpu()
            la = b["pressure_anomaly"].view(B, N).cpu()
            ma = b["pressure_mask"].view(B, N).cpu()
            for i in range(B):
                cls = int(atk[i]); s = ma[i] > 0
                if s.sum() == 0:
                    continue
                pa[cls][0].append(lv[i][s])
                pa[cls][1].append(la[i][s])

    Lt = torch.cat(L); Lat = torch.cat(La)
    over = compute_anomaly_metrics(
        (Lt > 0).long(), (Lat > 0.5).long(), scores=torch.sigmoid(Lt))
    rec = compute_recon_metrics(
        torch.cat(Pp), torch.cat(Pt), torch.cat(Pm), only_unobserved=True)
    name = {0: "clean", 1: "random", 2: "replay",
            3: "stealthy", 4: "noise", 5: "targeted"}
    out_pa = {}
    for cls, (ll, la) in pa.items():
        if not ll:
            continue
        lo = torch.cat(ll); lab = torch.cat(la)
        if lab.sum() == 0:
            continue
        mm = compute_anomaly_metrics(
            (lo > 0).long(), (lab > 0.5).long(), scores=torch.sigmoid(lo))
        out_pa[name[cls]] = {"f1": mm.f1, "auroc": mm.auroc}
    return {"f1": over.f1, "auroc": over.auroc,
            "p_mae": rec.mae, "per_attack": out_pa}


pre = evaluate(PRETRAINED)
sp = evaluate(SP_SINGLE)
moe = evaluate(SP_MOE)

print(f"\n{'metric':12s} {'pretrained':>11s} {'sp single':>11s} {'sp moe':>11s}")
print("-" * 50)
for k in ("f1", "auroc", "p_mae"):
    print(f"{k:12s} {pre[k]:>11.3f} {sp[k]:>11.3f} {moe[k]:>11.3f}")
for a in ("random", "replay", "stealthy", "noise", "targeted"):
    pf = pre["per_attack"].get(a, {}).get("f1", 0)
    sf = sp["per_attack"].get(a, {}).get("f1", 0)
    mf = moe["per_attack"].get(a, {}).get("f1", 0)
    print(f"{a:12s} {pf:>11.3f} {sf:>11.3f} {mf:>11.3f}")

(ROOT / "runs/selfplay/eval_atkmoe.json").write_text(
    json.dumps({"pretrained": pre, "sp_single": sp, "sp_moe": moe}, indent=2)
)
print("\nSaved runs/selfplay/eval_atkmoe.json")
