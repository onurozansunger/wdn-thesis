"""Is the residual uninformative because the estimator reads the sensor it estimates?

test_oneclass_gate.py found that a one-class score on the spatial residual
p_obs - p_hat loses to the supervised head, and that on replay it is
actively inverted (AUROC 0.216). Both are what you would expect if the
estimator has learned a partial identity map: the observed reading is an
input feature, so p_hat tracks p_obs, so a lie in p_obs moves the
prediction with it and cancels out of the residual.

If that is the cause, the fix is architectural and cheap to test. The
model was trained with 30% of sensors missing, so predicting a sensor it
cannot see is exactly what it already does. Masking a sensor at evaluation
time and reading off its prediction therefore costs no retraining and
stays in distribution.

This computes a leave-one-out residual by folds: the observed sensors are
split into R disjoint groups, and for each group the window is re-scored
with that group's readings removed, so every sensor is predicted once from
its neighbours alone. The resulting residual is a genuine prediction
error.

Compared, on the same windows and the same label-free calibration as the
gate:
    supervised head        the current detector
    |z| plain              residual with the sensor visible
    |z| masked             residual with the sensor hidden

    python3 scripts/test_masked_residual.py --data_dir data/v2_modena
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from wdn.models.temporal_moe import TemporalMixtureOfExpertsGNN
from wdn.corruption import ID_TO_ATTACK_TYPE
from eval_workshop_routing import load_split, to_device
from eval_unseen_attacks import inject

COL_P, COL_M = 5, 6          # pressure_obs, pressure_mask in the (N, 7) rows


def forward(model, b):
    return model(
        x_seq=b["x_seq"], edge_index=b["edge_index"],
        edge_attr=b["edge_attr"], is_original_edge=b["is_original_edge"],
        batch_size=b["batch_size"], num_nodes_per_graph=b["num_nodes"],
        pressure_obs=b["pressure_obs"], flow_obs=b["flow_obs"],
        pressure_mask=b["pressure_mask"], flow_mask=b["flow_mask"])


@torch.no_grad()
def masked_pred(model, b, folds, rng):
    """Prediction for every observed sensor, made with that sensor hidden."""
    obs = (b["pressure_mask"] > 0).nonzero(as_tuple=True)[0]
    order = obs[torch.from_numpy(rng.permutation(len(obs))).to(obs.device)]
    out = torch.full_like(b["pressure_obs"], float("nan"))
    for grp in torch.chunk(order, folds):
        if grp.numel() == 0:
            continue
        bb = dict(b)
        xs = []
        for t in b["x_seq"]:
            xt = t.clone()
            xt[grp, COL_P] = 0.0
            xt[grp, COL_M] = 0.0
            xs.append(xt)
        bb["x_seq"] = xs
        po = b["pressure_obs"].clone(); po[grp] = 0.0
        pm = b["pressure_mask"].clone(); pm[grp] = 0.0
        bb["pressure_obs"], bb["pressure_mask"] = po, pm
        out[grp] = forward(model, bb)["pressure_pred"][grp]
    return out


@torch.no_grad()
def score_batch(model, b, folds, rng):
    """Supervised score, plain residual and masked residual for one batch."""
    o = forward(model, b)
    plain = b["pressure_obs"] - o["pressure_pred"]
    mask_pred = masked_pred(model, b, folds, rng)
    masked = b["pressure_obs"] - mask_pred
    sup = torch.sigmoid(o["pressure_anomaly_logits"])
    return sup, plain, masked


def gather(model, loader, device, folds, seed, clean_only=False,
           unseen=None, k=20, clean_id=0):
    rng = np.random.default_rng(seed)
    S, P, M, I, Y, F = [], [], [], [], [], []
    for raw in loader:
        if unseen is not None or clean_only:
            keep = (raw["attack_type"] == clean_id).nonzero(as_tuple=True)[0]
            if keep.numel() == 0:
                continue
            N = raw["num_nodes"]
            sub = dict(raw)
            nidx = torch.cat([torch.arange(i * N, (i + 1) * N) for i in keep])
            sub["x_seq"] = [t[nidx] for t in raw["x_seq"]]
            for f in ("pressure_obs", "pressure_mask", "pressure_anomaly",
                      "y_pressure"):
                sub[f] = raw[f][nidx]
            sub["batch_size"] = int(keep.numel())
            E = raw["edge_index"].shape[1] // raw["batch_size"]
            ei, ea, eo = [], [], []
            for j, g in enumerate(keep.tolist()):
                ei.append(raw["edge_index"][:, g * E:(g + 1) * E] - g * N + j * N)
                ea.append(raw["edge_attr"][g * E:(g + 1) * E])
                eo.append(raw["is_original_edge"][g * E:(g + 1) * E])
            sub["edge_index"] = torch.cat(ei, 1)
            sub["edge_attr"] = torch.cat(ea, 0)
            sub["is_original_edge"] = torch.cat(eo, 0)
            NE = raw["flow_obs"].shape[0] // raw["batch_size"]
            fidx = torch.cat([torch.arange(g * NE, (g + 1) * NE) for g in keep])
            for f in ("flow_obs", "flow_mask", "y_flow", "flow_anomaly"):
                sub[f] = raw[f][fidx]
            sub["attack_report"] = raw["attack_report"][keep]
            raw = inject(sub, unseen, k, rng) if unseen else sub

        b = to_device(raw, device)
        sup, plain, masked = score_batch(model, b, folds, rng)
        m = (b["pressure_mask"] > 0).cpu()
        N = b["num_nodes"]
        nid = torch.arange(N).repeat(b["batch_size"])
        fam = b["attack_report"].cpu().repeat_interleave(N)
        S.append(sup.cpu()[m]); P.append(plain.cpu()[m]); M.append(masked.cpu()[m])
        I.append(nid[m]); Y.append(b["pressure_anomaly"].cpu()[m]); F.append(fam[m])
    cat = lambda xs: torch.cat(xs).numpy()
    return (cat(S), cat(P), cat(M), cat(I).astype(int),
            cat(Y).astype(int), cat(F).astype(int))


def fit(resid, node, n_nodes, min_n=30):
    gmu, gsd = float(np.nanmean(resid)), float(np.nanstd(resid))
    mu = np.full(n_nodes, gmu); sd = np.full(n_nodes, gsd)
    for s in range(n_nodes):
        sel = node == s
        if sel.sum() >= min_n:
            mu[s] = np.nanmean(resid[sel]); sd[s] = np.nanstd(resid[sel])
    return mu, np.maximum(sd, 1e-6)


def z(resid, node, mu, sd):
    return np.abs((resid - mu[node]) / sd[node])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/v2_modena")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--max_runs", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    runs = []
    for d in sorted((ROOT / "runs" / "temporal_moe").glob("2026*")):
        apth = d / "args.json"
        if not (apth.exists() and (d / "best_model.pt").exists()
                and (d / "test_results.json").exists()):
            continue
        a = json.load(open(apth))
        if (a.get("data_dir") == args.data_dir and a.get("epochs") == 60
                and a.get("window_size") == 6 and a.get("num_experts", 0) > 1
                and a.get("norm_mode") == "per_node"
                and a.get("lambda_expert") == 0.5
                and not a.get("no_pattern_features")
                and not a.get("no_topology")):
            runs.append((d, a))
    runs = runs[: args.max_runs]
    if not runs:
        raise SystemExit("no checkpoints for " + args.data_dir)
    print(f"  {args.data_dir}: {len(runs)} checkpoints, {args.folds} folds\n")

    loaders = None
    acc = {}

    def add(name, y, sup, zp, zm):
        r = acc.setdefault(name, {"sup": [], "zp": [], "zm": [],
                                  "sup_ap": [], "zp_ap": [], "zm_ap": []})
        r["sup"].append(roc_auc_score(y, sup))
        r["zp"].append(roc_auc_score(y, zp))
        r["zm"].append(roc_auc_score(y, zm))
        r["sup_ap"].append(average_precision_score(y, sup))
        r["zp_ap"].append(average_precision_score(y, zp))
        r["zm_ap"].append(average_precision_score(y, zm))

    for d, a in runs:
        if loaders is None:
            (tr, va, te, nm), graph, names = load_split(
                ROOT / args.data_dir, a["window_size"], a["batch_size"],
                a["norm_mode"])
            sample = te.dataset[0]
            n_nodes = graph.num_nodes
            loaders = True
        model = TemporalMixtureOfExpertsGNN(
            node_in_dim=sample["x_seq"][0].shape[1],
            edge_in_dim=sample["edge_attr"].shape[1],
            hidden_dim=a["hidden_dim"], num_experts=a["num_experts"],
            window_size=a["window_size"], gnn_type=a["gnn_type"],
            router_hidden_dim=a["router_hidden_dim"]).to(device)
        model.load_state_dict(torch.load(d / "best_model.pt", map_location=device))
        model.eval()

        # label-free calibration on clean validation windows
        _, cP, cM, cI, cY, _ = gather(model, va, device, args.folds,
                                      a["seed"], clean_only=True)
        ok = cY == 0
        mu_p, sd_p = fit(cP[ok], cI[ok], n_nodes)
        mu_m, sd_m = fit(cM[ok], cI[ok], n_nodes)
        print(f"  seed {a['seed']:>3}  clean residual sd: visible "
              f"{np.nanstd(cP[ok]):.4f}   hidden {np.nanstd(cM[ok]):.4f}")

        S, P, M, I, Y, F = gather(model, te, device, args.folds, a["seed"])
        zp, zm = z(P, I, mu_p, sd_p), z(M, I, mu_m, sd_m)
        add("in-dist (all)", Y, S, zp, zm)
        clean = Y == 0
        for kf in sorted(set(F)):
            if kf == 0:
                continue
            sel = clean | (F == kf)
            if Y[sel].sum() < 20:
                continue
            add(f"  in-dist: {ID_TO_ATTACK_TYPE.get(kf, kf)}",
                Y[sel], S[sel], zp[sel], zm[sel])

        for kind in ("sinusoidal", "swap", "slowdrift"):
            uS, uP, uM, uI, uY, _ = gather(model, te, device, args.folds,
                                           a["seed"], unseen=kind, k=args.k)
            add(f"UNSEEN: {kind}", uY, uS,
                z(uP, uI, mu_p, sd_p), z(uM, uI, mu_m, sd_m))

    print(f"\n  {'':30s}{'AUROC':>25s}   {'avg precision':>25s}")
    print(f"  {'':30s}{'sup':>8}{'|z| vis':>9}{'|z| hid':>9}   "
          f"{'sup':>8}{'|z| vis':>9}{'|z| hid':>9}")
    print("  " + "-" * 84)
    for k, v in acc.items():
        m = {kk: float(np.mean(vv)) for kk, vv in v.items()}
        star = "  <--" if m["zm"] > max(m["sup"], m["zp"]) + 0.02 else ""
        print(f"  {k:30s}{m['sup']:>8.3f}{m['zp']:>9.3f}{m['zm']:>9.3f}   "
              f"{m['sup_ap']:>8.3f}{m['zp_ap']:>9.3f}{m['zm_ap']:>9.3f}{star}")

    if args.out:
        json.dump({k: {kk: list(map(float, vv)) for kk, vv in v.items()}
                   for k, v in acc.items()}, open(args.out, "w"), indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
