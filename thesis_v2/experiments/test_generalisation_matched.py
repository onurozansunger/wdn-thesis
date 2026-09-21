"""Does the detector generalise, once magnitude is held fixed?

measure_unseen_magnitude.py showed the unseen-family test was confounded:
the injected families displace a sensor by 0.08--0.43 of its clean
residual sd, while the trained families displace it by 2.38--17.55. The
resulting F1 of 0.001--0.022 is therefore not evidence that the detector
memorised the generator. It is consistent with the perturbation simply
being below anyone's floor.

This removes the confound. Every family -- seen and unseen -- is injected
through the same code path, into the same clean windows, scaled so the
displacement is a prescribed multiple of that sensor's own clean residual
sd. Family shape is then the only thing that varies at a given magnitude.

    bias        additive offset. This is the shape the trained "random"
                and "targeted" families have, so it is the SEEN control.
    sinusoidal  oscillatory; no trained family oscillates.
    slowdrift   linear ramp an order of magnitude slower than the trained
                stealthy family.
    swap        two sensors exchange readings, amplified to reach the
                target displacement; each value stays plausible alone.

If seen and unseen curves coincide, the detector responds to displacement
and not to family identity, and there is no generalisation gap to fix. If
the unseen curves sit below the seen one at matched magnitude, the gap is
real and worth an architecture.

    python3 scripts/test_generalisation_matched.py --data_dir data/v2_modena
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
from eval_workshop_routing import load_split, to_device
from test_masked_residual import forward, gather, fit

COL_P, COL_M = 5, 6
FAMILIES = ("bias", "sinusoidal", "slowdrift", "swap")
SEEN = {"bias"}


def inject_scaled(batch, kind, k, mag, sd, rng):
    """Displace k sensors per graph by `mag` times their clean residual sd."""
    x = [t.clone() for t in batch["x_seq"]]
    B, N, T = batch["batch_size"], batch["num_nodes"], len(batch["x_seq"])
    anom = torch.zeros_like(batch["pressure_anomaly"])

    for b in range(B):
        off = b * N
        obs = (batch["pressure_mask"][off:off + N] > 0).nonzero(
            as_tuple=True)[0].numpy()
        if len(obs) < 2:
            continue
        sensors = rng.choice(obs, size=min(k, len(obs)), replace=False)

        if kind == "bias":
            for s in sensors:
                d = float(mag * sd[s]) * (1.0 if rng.random() < 0.5 else -1.0)
                for t in range(T):
                    x[t][off + s, COL_P] += d
                anom[off + s] = 1.0

        elif kind == "sinusoidal":
            # RMS of a sine is amp/sqrt(2); scale so RMS displacement = mag*sd.
            omega = 2 * np.pi / T
            for s in sensors:
                amp = float(mag * sd[s] * np.sqrt(2.0))
                phi = rng.uniform(0, 2 * np.pi)
                for t in range(T):
                    x[t][off + s, COL_P] += amp * float(np.sin(omega * t + phi))
                anom[off + s] = 1.0

        elif kind == "slowdrift":
            # Linear ramp; mean |displacement| over the window is rate*(T-1)/2.
            for s in sensors:
                rate = float(mag * sd[s] * 2.0 / max(T - 1, 1))
                for t in range(T):
                    x[t][off + s, COL_P] += rate * t
                anom[off + s] = 1.0

        elif kind == "swap":
            # Exchange, amplified along the a->c direction so the final
            # displacement is the requested size while the reading stays a
            # blend of two genuine sensor values.
            pairs = sensors[: 2 * (len(sensors) // 2)]
            for i in range(0, len(pairs), 2):
                a, c = int(pairs[i]), int(pairs[i + 1])
                for t in range(T):
                    va = x[t][off + a, COL_P].clone()
                    vc = x[t][off + c, COL_P].clone()
                    gap = float(vc - va)
                    if abs(gap) < 1e-6:
                        continue
                    alpha_a = float(mag * sd[a]) / abs(gap)
                    alpha_c = float(mag * sd[c]) / abs(gap)
                    x[t][off + a, COL_P] = va + alpha_a * (vc - va)
                    x[t][off + c, COL_P] = vc + alpha_c * (va - vc)
                anom[off + a] = anom[off + c] = 1.0
        else:
            raise ValueError(kind)

    return {**batch, "x_seq": x,
            "pressure_obs": x[-1][:, COL_P],
            "pressure_mask": batch["pressure_mask"],
            "pressure_anomaly": anom}


def clean_subset(raw, clean_id=0):
    keep = (raw["attack_type"] == clean_id).nonzero(as_tuple=True)[0]
    if keep.numel() == 0:
        return None
    N = raw["num_nodes"]
    sub = dict(raw)
    nidx = torch.cat([torch.arange(i * N, (i + 1) * N) for i in keep])
    sub["x_seq"] = [t[nidx] for t in raw["x_seq"]]
    for f in ("pressure_obs", "pressure_mask", "pressure_anomaly", "y_pressure"):
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
    return sub


@torch.no_grad()
def evaluate(model, loader, device, kind, mag, k, sd, mu, seed):
    rng = np.random.default_rng(seed)
    S, Z, Y = [], [], []
    for raw in loader:
        sub = clean_subset(raw)
        if sub is None:
            continue
        b = to_device(inject_scaled(sub, kind, k, mag, sd, rng), device)
        o = forward(model, b)
        m = (b["pressure_mask"] > 0).cpu()
        N = b["num_nodes"]
        nid = torch.arange(N).repeat(b["batch_size"])[m]
        r = (b["pressure_obs"] - o["pressure_pred"]).cpu()[m].numpy()
        S.append(torch.sigmoid(o["pressure_anomaly_logits"]).cpu()[m].numpy())
        Z.append(np.abs((r - mu[nid]) / sd[nid]))
        Y.append(b["pressure_anomaly"].cpu()[m].numpy())
    s, z, y = np.concatenate(S), np.concatenate(Z), np.concatenate(Y).astype(int)
    if y.sum() < 20 or (1 - y).sum() < 20:
        return None
    return roc_auc_score(y, s), roc_auc_score(y, z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/v2_modena")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--max_runs", type=int, default=3)
    ap.add_argument("--mags", default="0.5,1,2,4,8,16")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    mags = [float(x) for x in args.mags.split(",")]
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

        _, cP, _, cI, cY, _ = gather(model, va, device, 1, a["seed"],
                                     clean_only=True)
        ok = cY == 0
        mu, sd = fit(cP[ok], cI[ok], n_nodes)

        for kind in FAMILIES:
            for mg in mags:
                r = evaluate(model, te, device, kind, mg, args.k, sd, mu,
                             a["seed"])
                if r is None:
                    continue
                acc.setdefault((kind, mg), {"sup": [], "z": []})
                acc[(kind, mg)]["sup"].append(r[0])
                acc[(kind, mg)]["z"].append(r[1])
        print(f"  seed {a['seed']:>3}  done")

    print(f"\n  AUROC of the supervised head, by family shape and displacement")
    print(f"  (displacement in multiples of that sensor's clean residual sd)\n")
    hdr = "".join(f"{m:>9.1f}" for m in mags)
    print(f"  {'family':16s}{hdr}")
    print("  " + "-" * (16 + 9 * len(mags)))
    for kind in FAMILIES:
        tag = kind + (" *" if kind in SEEN else "")
        cells = "".join(
            f"{np.mean(acc[(kind, m)]['sup']):>9.3f}" if (kind, m) in acc
            else f"{'--':>9}" for m in mags)
        print(f"  {tag:16s}{cells}")
    print("\n  same, for the label-free |z| score on the residual\n")
    print(f"  {'family':16s}{hdr}")
    print("  " + "-" * (16 + 9 * len(mags)))
    for kind in FAMILIES:
        tag = kind + (" *" if kind in SEEN else "")
        cells = "".join(
            f"{np.mean(acc[(kind, m)]['z']):>9.3f}" if (kind, m) in acc
            else f"{'--':>9}" for m in mags)
        print(f"  {tag:16s}{cells}")
    print("\n  * = shape the detector was trained on (the control)")

    if args.out:
        json.dump({f"{k[0]}@{k[1]}": {kk: list(map(float, vv))
                                      for kk, vv in v.items()}
                   for k, v in acc.items()}, open(args.out, "w"), indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
