"""Does detectability collapse onto displacement alone?

test_generalisation_matched.py held displacement fixed and varied the
attack's shape, and the detector barely noticed: at 2 sd every shape
scored 0.91--0.98, at 8 sd every shape scored 0.95--1.00. That suggests
the family taxonomy the detector is built around is not a variable at all,
and that one number governs detection -- how far the reported value sits
from the network's own estimate, in units of the residual's spread.

This is the test of that claim on real generated data rather than injected
perturbations. For every dataset and every trained family it measures two
things independently:

    displacement  median |p_reported - p_true| for attacked sensors,
                  divided by that sensor's clean residual sd
    AUROC         separation of attacked from clean sensors

If the claim holds, the pairs from thirty different (dataset, family)
combinations -- different networks, different regimes, different attacker
models, different attack mechanisms -- lie on a single increasing curve,
and family identity adds nothing once displacement is known.

    python3 scripts/test_detectability_law.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from wdn.models.temporal_moe import TemporalMixtureOfExpertsGNN
from wdn.corruption import ID_TO_ATTACK_TYPE
from eval_workshop_routing import load_split, to_device
from test_masked_residual import forward, gather, fit

DATASETS = ["data/v2_modena", "data/hard_modena", "data/rec_hard_modena",
            "data/rec_modena", "data/ep_hard_modena",
            "data/v2_ltown", "data/hard_ltown", "data/rec_hard_ltown"]


def robust_fit(resid, node, n_nodes, min_n=30):
    """Median and MAD of the residual per sensor.

    Deliberately label-free: it is computed over every validation sensor,
    attacked ones included. With at most a sixth of sensors attacked the
    median and MAD are unmoved by them, so no attack label is needed and
    the same calibration works on the episode datasets, where no window is
    labelled clean at all.
    """
    K = 1.4826                       # MAD -> sd for a Gaussian
    gmu = float(np.median(resid))
    gsd = float(K * np.median(np.abs(resid - gmu)))
    mu = np.full(n_nodes, gmu)
    sd = np.full(n_nodes, max(gsd, 1e-6))
    for s in range(n_nodes):
        sel = node == s
        if sel.sum() >= min_n:
            m = float(np.median(resid[sel]))
            mu[s] = m
            sd[s] = max(float(K * np.median(np.abs(resid[sel] - m))), 1e-6)
    return mu, sd


def checkpoints(data_dir, max_runs):
    out = []
    for d in sorted((ROOT / "runs" / "temporal_moe").glob("2026*")):
        apth = d / "args.json"
        if not (apth.exists() and (d / "best_model.pt").exists()
                and (d / "test_results.json").exists()):
            continue
        a = json.load(open(apth))
        if (a.get("data_dir") == data_dir and a.get("epochs") == 60
                and a.get("window_size") == 6 and a.get("num_experts", 0) > 1
                and a.get("norm_mode") == "per_node"
                and a.get("lambda_expert") == 0.5
                and not a.get("no_pattern_features")
                and not a.get("no_topology")):
            out.append((d, a))
    return out[:max_runs]


@torch.no_grad()
def collect(model, loader, device, n_nodes):
    """Per observed sensor: sup score, residual, node, label, family, true shift."""
    S, R, I, Y, F, D = [], [], [], [], [], []
    for raw in loader:
        b = to_device(raw, device)
        o = forward(model, b)
        m = (b["pressure_mask"] > 0).cpu()
        N = b["num_nodes"]
        nid = torch.arange(N).repeat(b["batch_size"])
        S.append(torch.sigmoid(o["pressure_anomaly_logits"]).cpu()[m])
        R.append((b["pressure_obs"] - o["pressure_pred"]).cpu()[m])
        I.append(nid[m])
        Y.append(b["pressure_anomaly"].cpu()[m])
        F.append(b["attack_report"].cpu().repeat_interleave(N)[m])
        D.append((b["pressure_obs"] - b["y_pressure"]).abs().cpu()[m])
    cat = lambda xs: torch.cat(xs).numpy()
    return (cat(S), cat(R), cat(I).astype(int), cat(Y).astype(int),
            cat(F).astype(int), cat(D))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_runs", type=int, default=2)
    ap.add_argument("--out", default="runs/detectability_law.json")
    args = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    rows = []
    for dd in DATASETS:
        runs = checkpoints(dd, args.max_runs)
        if not runs or not (ROOT / dd).exists():
            print(f"  skip {dd} (no checkpoints)")
            continue
        loaders = None
        per = {}
        for d, a in runs:
            if loaders is None:
                (tr, va, te, nm), graph, names = load_split(
                    ROOT / dd, a["window_size"], a["batch_size"], a["norm_mode"])
                sample = te.dataset[0]
                n_nodes = graph.num_nodes
                loaders = True
            model = TemporalMixtureOfExpertsGNN(
                node_in_dim=sample["x_seq"][0].shape[1],
                edge_in_dim=sample["edge_attr"].shape[1],
                hidden_dim=a["hidden_dim"], num_experts=a["num_experts"],
                window_size=a["window_size"], gnn_type=a["gnn_type"],
                router_hidden_dim=a["router_hidden_dim"]).to(device)
            model.load_state_dict(torch.load(d / "best_model.pt",
                                             map_location=device))
            model.eval()

            _, cP, _, cI, cY, _ = gather(model, va, device, 1, a["seed"])
            mu, sd = robust_fit(cP, cI, n_nodes)

            S, R, I, Y, F, D = collect(model, te, device, n_nodes)
            Z = np.abs((R - mu[I]) / sd[I])
            Dn = D / sd[I]
            clean = Y == 0
            for kf in sorted(set(F)):
                if kf == 0:
                    continue
                sel = clean | (F == kf)
                y = Y[sel]
                if y.sum() < 20 or (1 - y).sum() < 20:
                    continue
                atk = (F == kf) & (Y > 0)
                k = per.setdefault(ID_TO_ATTACK_TYPE.get(kf, str(kf)),
                                   {"d": [], "sup": [], "z": [], "n": 0})
                k["d"].append(float(np.median(Dn[atk])))
                k["sup"].append(roc_auc_score(y, S[sel]))
                k["z"].append(roc_auc_score(y, Z[sel]))
                k["n"] = int(atk.sum())
        for fam, v in per.items():
            rows.append({"dataset": dd.split("/")[-1], "family": fam,
                         "n": v["n"], "disp": float(np.mean(v["d"])),
                         "sup": float(np.mean(v["sup"])),
                         "z": float(np.mean(v["z"]))})
        print(f"  {dd}: {len(per)} families")

    rows.sort(key=lambda r: r["disp"])
    print(f"\n  {'dataset':18s}{'family':11s}{'displacement':>14s}"
          f"{'AUROC sup':>11s}{'AUROC |z|':>11s}")
    print("  " + "-" * 65)
    for r in rows:
        print(f"  {r['dataset']:18s}{r['family']:11s}{r['disp']:>12.2f} sd"
              f"{r['sup']:>11.3f}{r['z']:>11.3f}")

    d = np.array([r["disp"] for r in rows])
    zz = np.array([r["z"] for r in rows])
    ss = np.array([r["sup"] for r in rows])
    from scipy.stats import spearmanr
    print(f"\n  Spearman(displacement, AUROC):  |z| {spearmanr(d, zz)[0]:+.3f}"
          f"   supervised {spearmanr(d, ss)[0]:+.3f}   (n={len(rows)})")
    hi = d >= 2.0
    lo = d < 1.0
    if hi.any():
        print(f"  displacement >= 2 sd  (n={hi.sum():2d}):  |z| AUROC "
              f"{zz[hi].mean():.3f} +- {zz[hi].std():.3f}")
    if lo.any():
        print(f"  displacement <  1 sd  (n={lo.sum():2d}):  |z| AUROC "
              f"{zz[lo].mean():.3f} +- {zz[lo].std():.3f}")

    json.dump(rows, open(args.out, "w"), indent=2)
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
