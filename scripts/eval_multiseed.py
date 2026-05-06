"""Evaluate 3 self-play Modena seeds and report mean ± std vs pretrained."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from statistics import mean, stdev

import torch

sys.path.insert(0, "src")

from wdn.metrics import compute_anomaly_metrics, compute_recon_metrics
from wdn.models.temporal_moe import TemporalMixtureOfExpertsGNN
from wdn.temporal_dataset import create_temporal_dataloaders


_ROOT = Path(__file__).parent.parent
PRETRAINED = str(_ROOT / "runs/temporal_moe/20260505_144409/best_model.pt")
DATA_DIR = _ROOT / "data/temporal_moe_modena"
HIDDEN = 48
SEEDS = [
    str(_ROOT / "runs/selfplay/20260506_003725/defender.pt"),
    str(_ROOT / "runs/selfplay/20260506_005944/defender.pt"),
    str(_ROOT / "runs/selfplay/20260506_012245/defender.pt"),
]
# AttackerMoE-trained defender (single seed, headline run)
ATKMOE = str(_ROOT / "runs/selfplay/20260506_110630/defender.pt")


def evaluate_one(state_dict_path):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    snaps = pickle.load(open(DATA_DIR / "snapshots.pkl", "rb"))
    corr = pickle.load(open(DATA_DIR / "corrupted.pkl", "rb"))
    n = len(snaps); n_train = int(0.7 * n); n_val = int(0.15 * n)
    _, _, test_loader, _ = create_temporal_dataloaders(
        snaps[:n_train], corr[:n_train],
        snaps[n_train:n_train+n_val], corr[n_train:n_train+n_val],
        snaps[n_train+n_val:], corr[n_train+n_val:],
        window_size=6, batch_size=8,
    )
    m = TemporalMixtureOfExpertsGNN(
        node_in_dim=7, edge_in_dim=8, hidden_dim=HIDDEN,
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


def msd(xs):
    if len(xs) <= 1:
        return (xs[0] if xs else 0.0), 0.0
    return mean(xs), stdev(xs)


pre = evaluate_one(PRETRAINED)
sps = [evaluate_one(p) for p in SEEDS]

attacks = ["random", "replay", "stealthy", "noise", "targeted"]
print(f"\n{'metric':12s} {'pretrained':>12s}   {'self-play (mean±std)':>26s}")
print("-" * 60)
for k in ("f1", "auroc", "p_mae"):
    m, s = msd([sp[k] for sp in sps])
    print(f"{k:12s} {pre[k]:>12.3f}   {m:>10.3f} ± {s:.3f}")
for a in attacks:
    pf = pre["per_attack"].get(a, {}).get("f1")
    sf = [sp["per_attack"].get(a, {}).get("f1") for sp in sps]
    sf = [x for x in sf if x is not None]
    if pf is None or not sf:
        continue
    m, s = msd(sf)
    print(f"{a:12s} {pf:>12.3f}   {m:>10.3f} ± {s:.3f}")

(_ROOT / "runs/selfplay/eval_multiseed.json").write_text(
    json.dumps({"pretrained": pre, "selfplay_seeds": sps}, indent=2)
)
print("\nSaved runs/selfplay/eval_multiseed.json")
