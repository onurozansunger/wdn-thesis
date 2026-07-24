"""Sweep the anomaly decision threshold and record the operating curve.

The detector emits a per-sensor anomaly probability; turning that into a
decision needs a threshold. The default 0.5 is arbitrary — and badly
mis-calibrated under per-node normalisation, which shifts the score
distribution. This script measures, for each normalisation mode:

    overall F1  and  replay F1   as a function of the threshold

which exposes the operating trade-off (a high threshold buys precision on
the easy classes but discards the weak replay signal) and shows where the
validation-calibrated threshold lands on that curve.

Writes runs/temporal_moe/threshold_sweep.json.

    python3 scripts/eval_threshold_sweep.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from wdn.temporal_dataset import create_temporal_dataloaders
from wdn.models.temporal_moe import TemporalMixtureOfExpertsGNN

ATTACK_ID = {"clean": 0, "random": 1, "replay": 2,
             "stealthy": 3, "noise": 4, "targeted": 5}


def load_run(run_dir: Path):
    a = json.load(open(run_dir / "args.json"))
    dd = ROOT / a["data_dir"]
    snaps = pickle.load(open(dd / "snapshots.pkl", "rb"))
    corr = pickle.load(open(dd / "corrupted.pkl", "rb"))
    n = len(snaps); ntr = int(0.7 * n); nv = int(0.15 * n)
    _, val, test, norm = create_temporal_dataloaders(
        snaps[:ntr], corr[:ntr], snaps[ntr:ntr + nv], corr[ntr:ntr + nv],
        snaps[ntr + nv:], corr[ntr + nv:],
        window_size=a["window_size"], batch_size=8,
        norm_mode=a.get("norm_mode", "global"),
    )
    sample = next(iter(test))
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = TemporalMixtureOfExpertsGNN(
        node_in_dim=sample["x_seq"][0].shape[1],
        edge_in_dim=sample["edge_attr"].shape[1],
        hidden_dim=a["hidden_dim"], num_experts=a["num_experts"],
        window_size=a["window_size"], gnn_type=a["gnn_type"],
        router_hidden_dim=a.get("router_hidden_dim", 32),
    ).to(dev)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=dev))
    model.eval()
    return model, val, test, dev, a


@torch.no_grad()
def collect(model, loader, dev):
    S, L, M, C = [], [], [], []
    for raw in loader:
        b = {k: (v.to(dev) if isinstance(v, torch.Tensor)
                 else [t.to(dev) for t in v] if isinstance(v, list) else v)
             for k, v in raw.items()}
        o = model(
            x_seq=b["x_seq"], edge_index=b["edge_index"],
            edge_attr=b["edge_attr"], is_original_edge=b["is_original_edge"],
            batch_size=b["batch_size"], num_nodes_per_graph=b["num_nodes"],
            pressure_obs=b["pressure_obs"], flow_obs=b["flow_obs"],
            pressure_mask=b["pressure_mask"], flow_mask=b["flow_mask"],
        )
        S.append(torch.sigmoid(o["pressure_anomaly_logits"]).cpu())
        L.append(b["pressure_anomaly"].cpu())
        M.append(b["pressure_mask"].cpu())
        C.append(b["attack_type"].repeat_interleave(b["num_nodes"]).cpu())
    return [torch.cat(x).numpy() for x in (S, L, M, C)]


def f1_at(scores, labels, t):
    pred = scores > t
    pos = labels > 0.5
    tp = np.logical_and(pred, pos).sum()
    fp = np.logical_and(pred, ~pos).sum()
    fn = np.logical_and(~pred, pos).sum()
    d = 2 * tp + fp + fn
    return float(2 * tp / d) if d > 0 else 0.0


def main():
    # Latest 5-seed final runs, one per normalisation mode.
    runs = {}
    for d in sorted((ROOT / "runs" / "temporal_moe").glob("2026*")):
        ap = d / "args.json"
        if not (ap.exists() and (d / "best_model.pt").exists()):
            continue
        a = json.load(open(ap))
        if (a.get("data_dir") != "data/temporal_moe_modena"
                or a.get("epochs") != 60 or a.get("hidden_dim") != 64
                or a.get("seed") != 1):
            continue
        runs[a.get("norm_mode", "global")] = d      # newest wins
    if not runs:
        print("No matching runs found."); return

    grid = np.linspace(0.02, 0.98, 97)
    out = {}
    for mode, run_dir in runs.items():
        print(f"[{mode}] {run_dir.name}")
        model, val, test, dev, a = load_run(run_dir)
        vS, vL, vM, _ = collect(model, val, dev)
        tS, tL, tM, tC = collect(model, test, dev)
        vo, to = vM > 0, tM > 0
        rep = to & (tC == ATTACK_ID["replay"])

        # Calibrate on validation, evaluate the whole curve on test.
        val_f1 = [f1_at(vS[vo], vL[vo], t) for t in grid]
        tau = float(grid[int(np.argmax(val_f1))])
        out[mode] = {
            "run": run_dir.name,
            "threshold": grid.tolist(),
            "overall_f1": [f1_at(tS[to], tL[to], t) for t in grid],
            "replay_f1": [f1_at(tS[rep], tL[rep], t) for t in grid],
            "calibrated_tau": tau,
            "f1_at_calibrated": f1_at(tS[to], tL[to], tau),
            "replay_at_calibrated": f1_at(tS[rep], tL[rep], tau),
            "f1_at_default": f1_at(tS[to], tL[to], 0.5),
            "replay_at_default": f1_at(tS[rep], tL[rep], 0.5),
        }
        r = out[mode]
        print(f"   tau={tau:.2f}  F1 {r['f1_at_default']:.4f} -> "
              f"{r['f1_at_calibrated']:.4f}   replay "
              f"{r['replay_at_default']:.3f} -> {r['replay_at_calibrated']:.3f}")

    p = ROOT / "runs" / "temporal_moe" / "threshold_sweep.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nWrote {p}")


if __name__ == "__main__":
    main()
