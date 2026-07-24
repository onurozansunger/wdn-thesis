"""Seed ensemble as the final deterministic model.

Averages the per-sensor anomaly scores of the N same-config models, then
calibrates one decision threshold on the (ensembled) validation split.
Reports overall and per-attack F1/AUROC and writes a JSON summary.

    python3 scripts/eval_ensemble.py --norm_mode per_node
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from wdn.temporal_dataset import create_temporal_dataloaders
from wdn.models.temporal_moe import TemporalMixtureOfExpertsGNN
from wdn.metrics import compute_anomaly_metrics
from sklearn.metrics import roc_auc_score

ATK = {1: "random", 2: "replay", 3: "stealthy", 4: "noise", 5: "targeted"}


def load(run):
    a = json.load(open(run + "/args.json")); dd = a["data_dir"]
    snaps = pickle.load(open(f"{ROOT}/{dd}/snapshots.pkl", "rb"))
    corr = pickle.load(open(f"{ROOT}/{dd}/corrupted.pkl", "rb"))
    n = len(snaps); ntr = int(0.7 * n); nv = int(0.15 * n)
    _, va, te, _ = create_temporal_dataloaders(
        snaps[:ntr], corr[:ntr], snaps[ntr:ntr + nv], corr[ntr:ntr + nv],
        snaps[ntr + nv:], corr[ntr + nv:], window_size=a["window_size"],
        batch_size=8, norm_mode=a.get("norm_mode", "global"))
    s = next(iter(te))
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    m = TemporalMixtureOfExpertsGNN(
        node_in_dim=s["x_seq"][0].shape[1], edge_in_dim=s["edge_attr"].shape[1],
        hidden_dim=a["hidden_dim"], num_experts=a["num_experts"],
        window_size=a["window_size"], gnn_type=a["gnn_type"],
        router_hidden_dim=a.get("router_hidden_dim", 32)).to(dev)
    m.load_state_dict(torch.load(run + "/best_model.pt", map_location=dev))
    m.eval()
    return m, va, te, dev


@torch.no_grad()
def scores(m, loader, dev):
    S, L, M, TC = [], [], [], []
    for raw in loader:
        b = {k: (v.to(dev) if isinstance(v, torch.Tensor)
                 else [t.to(dev) for t in v] if isinstance(v, list) else v)
             for k, v in raw.items()}
        o = m(x_seq=b["x_seq"], edge_index=b["edge_index"], edge_attr=b["edge_attr"],
              is_original_edge=b["is_original_edge"], batch_size=b["batch_size"],
              num_nodes_per_graph=b["num_nodes"], pressure_obs=b["pressure_obs"],
              flow_obs=b["flow_obs"], pressure_mask=b["pressure_mask"],
              flow_mask=b["flow_mask"])
        S.append(torch.sigmoid(o["pressure_anomaly_logits"]).cpu())
        L.append(b["pressure_anomaly"].cpu()); M.append(b["pressure_mask"].cpu())
        TC.append(b["attack_type"].repeat_interleave(b["num_nodes"]).cpu())
    return [torch.cat(x).numpy() for x in (S, L, M, TC)]


def best_t(sc, lb):
    pos = lb > 0.5; bt, bf = 0.5, -1.0
    for t in np.linspace(0.02, 0.98, 97):
        pr = sc > t
        tp = (pr & pos).sum(); fp = (pr & ~pos).sum(); fn = (~pr & pos).sum()
        d = 2 * tp + fp + fn
        f = 2 * tp / d if d > 0 else 0.0
        if f > bf:
            bf, bt = f, t
    return bt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--norm_mode", default="per_node")
    p.add_argument("--after", default="20260721_113")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    runs = []
    for d in sorted(glob.glob(str(ROOT / "runs" / "temporal_moe" / "2026*"))):
        if os.path.basename(d) < args.after:
            continue
        ap = os.path.join(d, "args.json")
        if not (os.path.exists(ap) and os.path.exists(os.path.join(d, "best_model.pt"))):
            continue
        a = json.load(open(ap))
        if (a.get("norm_mode", "global") == args.norm_mode
                and a.get("seed") in range(1, 6) and a.get("epochs") == 60
                and a.get("data_dir") == "data/temporal_moe_modena"):
            runs.append(d)
    if not runs:
        print("No matching runs."); return

    vS, tS = [], []
    vL = vM = tL = tM = tTC = None
    single = []
    for run in runs:
        m, va, te, dev = load(run)
        vs, vl, vm, _ = scores(m, va, dev)
        ts, tl, tm, ttc = scores(m, te, dev)
        vS.append(vs); tS.append(ts)
        vL, vM, tL, tM, tTC = vl, vm, tl, tm, ttc
        vo, to = vm > 0, tm > 0
        gt = best_t(vs[vo], vl[vo])
        single.append(compute_anomaly_metrics(
            torch.tensor((ts > gt)[to]).float(), torch.tensor(tl[to]),
            torch.tensor(ts[to])).f1)

    vE = np.mean(vS, axis=0); tE = np.mean(tS, axis=0)
    vo, to = vM > 0, tM > 0
    gt = best_t(vE[vo], vL[vo])
    ens = compute_anomaly_metrics(
        torch.tensor((tE > gt)[to]).float(), torch.tensor(tL[to]),
        torch.tensor(tE[to]))

    per_attack = {}
    for c, nm in ATK.items():
        s = to & (tTC == c)
        if s.sum() == 0:
            continue
        f1 = compute_anomaly_metrics(
            torch.tensor((tE > gt)[s]).float(), torch.tensor(tL[s]),
            torch.tensor(tE[s])).f1
        au = roc_auc_score(tL[s], tE[s]) if tL[s].max() > 0 else 0.5
        per_attack[nm] = {"f1": float(f1), "auroc": float(au)}

    out = {
        "n_models": len(runs),
        "single_seed_mean_f1": st.mean(single),
        "single_seed_std_f1": st.stdev(single),
        "ensemble_f1": float(ens.f1),
        "ensemble_precision": float(ens.precision),
        "ensemble_recall": float(ens.recall),
        "ensemble_auroc": float(ens.auroc),
        "threshold": float(gt),
        "per_attack": per_attack,
    }
    dst = args.out or str(ROOT / "runs" / "temporal_moe" / "ensemble_eval.json")
    json.dump(out, open(dst, "w"), indent=2)
    print(f"single-seed mean F1 {out['single_seed_mean_f1']:.4f} "
          f"± {out['single_seed_std_f1']:.4f}")
    print(f"ENSEMBLE ({len(runs)} models) F1 {out['ensemble_f1']:.4f}  "
          f"AUROC {out['ensemble_auroc']:.4f}")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
