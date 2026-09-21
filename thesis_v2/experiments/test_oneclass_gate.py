"""Gate experiment: does a label-free one-class score beat the supervised head?

The measured failure of the current detector is generalisation, not
accuracy: F1 0.84 in distribution against 0.001--0.022 on attack families
it never trained on (thesis/workshop/tables/unseen.tex). The diagnosis is
that a head trained to name five generator families learns the generator.
A score that models only the *clean* residual cannot make that mistake,
because it never sees an attack at all.

This tests the diagnosis without training anything. The trained detector
already produces a spatial residual p_obs - p_hat for every observed
sensor. We calibrate a one-class score on that residual using clean
validation windows only -- per-sensor mean and sd, no attack label
anywhere -- and then compare it against the model's own supervised head on
the same test windows.

Thresholds are chosen label-free too: the one-class threshold is the
quantile of the clean calibration score that matches the supervised head's
false-alarm rate on the same clean windows, so neither score gets to peek
at attacked data to set its operating point.

If |z| clearly beats the head on unseen families, the architecture bet is
confirmed and worth building. If it does not, the bet is dead and we say
so.

    python3 scripts/test_oneclass_gate.py --data_dir data/v2_modena
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from wdn.models.temporal_moe import TemporalMixtureOfExpertsGNN
from wdn.corruption import ID_TO_ATTACK_TYPE
from eval_workshop_routing import load_split, to_device
from eval_unseen_attacks import inject

COL_P, COL_M = -2, -1


def forward(model, b):
    return model(
        x_seq=b["x_seq"], edge_index=b["edge_index"],
        edge_attr=b["edge_attr"], is_original_edge=b["is_original_edge"],
        batch_size=b["batch_size"], num_nodes_per_graph=b["num_nodes"],
        pressure_obs=b["pressure_obs"], flow_obs=b["flow_obs"],
        pressure_mask=b["pressure_mask"], flow_mask=b["flow_mask"])


@torch.no_grad()
def collect(model, loader, device, clean_only=False, clean_id=0):
    """Per observed sensor: supervised score, residual, node id, label, family."""
    S, R, I, Y, F = [], [], [], [], []
    for raw in loader:
        if clean_only:
            keep = (raw["attack_type"] == clean_id)
            if not keep.any():
                continue
        b = to_device(raw, device)
        out = forward(model, b)
        m = (b["pressure_mask"] > 0).cpu()
        N = b["num_nodes"]
        nid = torch.arange(N).repeat(b["batch_size"])
        fam = b["attack_report"].cpu().repeat_interleave(N)
        lab = b["pressure_anomaly"].cpu()
        if clean_only:
            gk = keep.repeat_interleave(N)
            m = m & gk
        S.append(torch.sigmoid(out["pressure_anomaly_logits"]).cpu()[m])
        R.append((b["pressure_obs"] - out["pressure_pred"]).cpu()[m])
        I.append(nid[m]); Y.append(lab[m]); F.append(fam[m])
    cat = lambda xs: torch.cat(xs).numpy()
    return cat(S), cat(R), cat(I).astype(int), cat(Y).astype(int), cat(F).astype(int)


@torch.no_grad()
def collect_unseen(model, loader, device, kind, k, seed, clean_id=0):
    """Same, but with an unseen family injected into clean windows."""
    rng = np.random.default_rng(seed)
    S, R, I, Y = [], [], [], []
    for raw in loader:
        keep = (raw["attack_type"] == clean_id).nonzero(as_tuple=True)[0]
        if keep.numel() == 0:
            continue
        N = raw["num_nodes"]
        sub = dict(raw)
        node_idx = torch.cat([torch.arange(i * N, (i + 1) * N) for i in keep])
        sub["x_seq"] = [t[node_idx] for t in raw["x_seq"]]
        for f in ("pressure_obs", "pressure_mask", "pressure_anomaly", "y_pressure"):
            sub[f] = raw[f][node_idx]
        sub["batch_size"] = int(keep.numel())
        E = raw["edge_index"].shape[1] // raw["batch_size"]
        eidx, eattr, eorig = [], [], []
        for j, g in enumerate(keep.tolist()):
            eidx.append(raw["edge_index"][:, g * E:(g + 1) * E] - g * N + j * N)
            eattr.append(raw["edge_attr"][g * E:(g + 1) * E])
            eorig.append(raw["is_original_edge"][g * E:(g + 1) * E])
        sub["edge_index"] = torch.cat(eidx, 1)
        sub["edge_attr"] = torch.cat(eattr, 0)
        sub["is_original_edge"] = torch.cat(eorig, 0)
        NE = raw["flow_obs"].shape[0] // raw["batch_size"]
        fidx = torch.cat([torch.arange(g * NE, (g + 1) * NE) for g in keep])
        for f in ("flow_obs", "flow_mask", "y_flow", "flow_anomaly"):
            sub[f] = raw[f][fidx]

        b = to_device(inject(sub, kind, k, rng), device)
        out = forward(model, b)
        m = (b["pressure_mask"] > 0).cpu()
        nid = torch.arange(N).repeat(b["batch_size"])
        S.append(torch.sigmoid(out["pressure_anomaly_logits"]).cpu()[m])
        R.append((b["pressure_obs"] - out["pressure_pred"]).cpu()[m])
        I.append(nid[m]); Y.append(b["pressure_anomaly"].cpu()[m])
    cat = lambda xs: torch.cat(xs).numpy()
    return cat(S), cat(R), cat(I).astype(int), cat(Y).astype(int)


def fit_oneclass(resid, node, n_nodes, min_n=30):
    """Per-sensor mean/sd of the clean residual, global fallback when sparse."""
    gmu, gsd = float(resid.mean()), float(resid.std())
    mu = np.full(n_nodes, gmu, dtype=np.float64)
    sd = np.full(n_nodes, gsd, dtype=np.float64)
    for s in range(n_nodes):
        sel = node == s
        if sel.sum() >= min_n:
            mu[s] = resid[sel].mean()
            sd[s] = resid[sel].std()
    sd = np.maximum(sd, 1e-6)
    return mu, sd


def zscore(resid, node, mu, sd):
    return np.abs((resid - mu[node]) / sd[node])


def prf(y, pred):
    return float(f1_score(y, pred, zero_division=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/v2_modena")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--max_runs", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    runs = []
    for d in sorted((ROOT / "runs" / "temporal_moe").glob("2026*")):
        apth, tp = d / "args.json", d / "test_results.json"
        if not (apth.exists() and tp.exists() and (d / "best_model.pt").exists()):
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
    print(f"  {args.data_dir}: {len(runs)} checkpoints\n")

    loaders = None
    acc = {}
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
        thr = json.load(open(d / "test_results.json")).get("threshold", 0.5)

        # --- calibrate on CLEAN VALIDATION windows only, no labels used ---
        cS, cR, cI, cY, _ = collect(model, va, device, clean_only=True)
        keep = cY == 0                       # drop residual leakage from partial attacks
        mu, sd = fit_oneclass(cR[keep], cI[keep], n_nodes)
        cz = zscore(cR[keep], cI[keep], mu, sd)
        fpr_sup = float((cS[keep] > thr).mean())
        zthr = float(np.quantile(cz, 1.0 - fpr_sup)) if fpr_sup > 0 else float(cz.max())

        # --- in-distribution test ---
        S, R, I, Y, F = collect(model, te, device)
        Z = zscore(R, I, mu, sd)
        rec = acc.setdefault("in-dist (all)", {"sup_a": [], "z_a": [], "sup_f": [], "z_f": []})
        rec["sup_a"].append(roc_auc_score(Y, S)); rec["z_a"].append(roc_auc_score(Y, Z))
        rec["sup_f"].append(prf(Y, (S > thr).astype(int)))
        rec["z_f"].append(prf(Y, (Z > zthr).astype(int)))

        clean = Y == 0
        for kf in sorted(set(F)):
            if kf == 0:
                continue
            sel = clean | (F == kf)
            y = Y[sel]
            if y.sum() < 20:
                continue
            nm_ = f"  in-dist: {ID_TO_ATTACK_TYPE.get(kf, kf)}"
            r = acc.setdefault(nm_, {"sup_a": [], "z_a": [], "sup_f": [], "z_f": []})
            r["sup_a"].append(roc_auc_score(y, S[sel])); r["z_a"].append(roc_auc_score(y, Z[sel]))
            r["sup_f"].append(prf(y, (S[sel] > thr).astype(int)))
            r["z_f"].append(prf(y, (Z[sel] > zthr).astype(int)))

        # --- unseen families ---
        for kind in ("sinusoidal", "swap", "slowdrift"):
            uS, uR, uI, uY = collect_unseen(model, te, device, kind, args.k, a["seed"])
            uZ = zscore(uR, uI, mu, sd)
            r = acc.setdefault(f"UNSEEN: {kind}", {"sup_a": [], "z_a": [], "sup_f": [], "z_f": []})
            r["sup_a"].append(roc_auc_score(uY, uS)); r["z_a"].append(roc_auc_score(uY, uZ))
            r["sup_f"].append(prf(uY, (uS > thr).astype(int)))
            r["z_f"].append(prf(uY, (uZ > zthr).astype(int)))
        print(f"  seed {a['seed']:>3}  done   (sup FPR on clean {fpr_sup:.3f}, "
              f"z threshold {zthr:.2f})")

    print(f"\n  {'':30s}{'F1 sup':>9}{'F1 |z|':>9}{'  ':2}"
          f"{'AUROC sup':>11}{'AUROC |z|':>11}")
    print("  " + "-" * 72)
    for k, v in acc.items():
        f_s, f_z = np.mean(v["sup_f"]), np.mean(v["z_f"])
        a_s, a_z = np.mean(v["sup_a"]), np.mean(v["z_a"])
        star = "  <--" if f_z > f_s + 0.02 else ""
        print(f"  {k:30s}{f_s:>9.3f}{f_z:>9.3f}  {a_s:>11.3f}{a_z:>11.3f}{star}")

    if args.out:
        json.dump({k: {kk: list(map(float, vv)) for kk, vv in v.items()}
                   for k, v in acc.items()}, open(args.out, "w"), indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
