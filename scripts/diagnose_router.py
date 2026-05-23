"""Router-misdirection diagnostic.

The supervisors suspect the Mixture-of-Experts router sends attacks to
the wrong experts. This script settles it with hard numbers:

  1. Confusion matrix — true attack class vs router argmax — on the
     test split of each network.
  2. Per-class routing accuracy (diagonal of the matrix).
  3. Router confidence — softmax mass on the chosen expert — split by
     correct vs incorrect routing.

A healthy router has a near-diagonal matrix and high confidence on the
correct class. A misdirecting router shows systematic off-diagonal
mass (e.g. replay always routed to the noise expert).
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from wdn.models.temporal_moe import TemporalMixtureOfExpertsGNN
from wdn.temporal_dataset import create_temporal_dataloaders

CLASS_NAMES = ["clean", "random", "replay", "stealthy", "noise", "targeted"]

NETS = {
    "Modena": {
        "data_dir": "data/temporal_moe_modena", "hidden_dim": 48,
        "ckpt": "runs/temporal_moe/20260522_200642/best_model.pt",
    },
    "Net3": {
        "data_dir": "data/temporal_moe_net3", "hidden_dim": 48,
        "ckpt": "runs/temporal_moe/20260522_195527/best_model.pt",
    },
}


def make_test_loader(data_dir):
    snaps = pickle.load(open(data_dir / "snapshots.pkl", "rb"))
    corr = pickle.load(open(data_dir / "corrupted.pkl", "rb"))
    n = len(snaps); nt = int(0.7 * n); nv = int(0.15 * n)
    _, _, tl, _ = create_temporal_dataloaders(
        snaps[:nt], corr[:nt], snaps[nt:nt+nv], corr[nt:nt+nv],
        snaps[nt+nv:], corr[nt+nv:], window_size=6, batch_size=8,
    )
    return tl


@torch.no_grad()
def diagnose(name, cfg):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data_dir = ROOT / cfg["data_dir"]
    loader = make_test_loader(data_dir)

    model = TemporalMixtureOfExpertsGNN(
        node_in_dim=7, edge_in_dim=8, hidden_dim=cfg["hidden_dim"],
        num_experts=6, window_size=6, gnn_type="GraphSAGE",
    ).to(device)
    model.load_state_dict(torch.load(ROOT / cfg["ckpt"], map_location=device))
    model.eval()

    n_cls = 6
    confusion = np.zeros((n_cls, n_cls), dtype=int)   # rows=true, cols=pred
    conf_correct, conf_wrong = [], []

    for raw in loader:
        b = {k: (v.to(device) if isinstance(v, torch.Tensor)
                 else [t.to(device) for t in v] if isinstance(v, list) else v)
             for k, v in raw.items()}
        out = model(
            x_seq=b["x_seq"], edge_index=b["edge_index"],
            edge_attr=b["edge_attr"], is_original_edge=b["is_original_edge"],
            batch_size=b["batch_size"], num_nodes_per_graph=b["num_nodes"],
            pressure_obs=b["pressure_obs"], flow_obs=b["flow_obs"],
            pressure_mask=b["pressure_mask"], flow_mask=b["flow_mask"],
        )
        logits = out["router_logits"].cpu()                  # (B, 6)
        probs = F.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1)
        true = b["attack_type"].cpu()
        for t, p, pr in zip(true.tolist(), pred.tolist(), probs):
            confusion[t, p] += 1
            (conf_correct if t == p else conf_wrong).append(float(pr[p]))

    total = confusion.sum()
    overall_acc = np.trace(confusion) / max(total, 1)

    print(f"\n{'='*64}\n{name}  —  router diagnostic  (test windows: {total})\n{'='*64}")
    # Confusion matrix
    hdr = "true \\ pred  " + "".join(f"{c[:8]:>9s}" for c in CLASS_NAMES)
    print(hdr)
    for i, cname in enumerate(CLASS_NAMES):
        row = confusion[i]
        rsum = row.sum()
        cells = "".join(f"{v:>9d}" for v in row)
        diag_acc = row[i] / max(rsum, 1)
        print(f"{cname:11s}{cells}   acc={diag_acc:.3f} (n={rsum})")
    print(f"\nOverall router accuracy: {overall_acc:.3f}")

    # Confidence split
    cc = np.mean(conf_correct) if conf_correct else 0.0
    cw = np.mean(conf_wrong) if conf_wrong else 0.0
    print(f"Mean confidence — correct routing: {cc:.3f}  |  "
          f"wrong routing: {cw:.3f}")

    # Biggest off-diagonal leak
    leak = confusion.copy()
    np.fill_diagonal(leak, 0)
    if leak.sum() > 0:
        i, j = np.unravel_index(leak.argmax(), leak.shape)
        print(f"Largest misroute: {CLASS_NAMES[i]} -> {CLASS_NAMES[j]} "
              f"({leak[i, j]} windows, "
              f"{leak[i, j] / max(confusion[i].sum(), 1):.0%} of {CLASS_NAMES[i]})")

    return {
        "overall_acc": float(overall_acc),
        "confusion": confusion.tolist(),
        "per_class_acc": {
            CLASS_NAMES[i]: float(confusion[i, i] / max(confusion[i].sum(), 1))
            for i in range(n_cls)
        },
        "conf_correct": float(cc),
        "conf_wrong": float(cw),
    }


def main():
    out = {}
    for name, cfg in NETS.items():
        if not (ROOT / cfg["data_dir"] / "snapshots.pkl").exists():
            continue
        out[name] = diagnose(name, cfg)
    (ROOT / "runs" / "selfplay" / "router_diagnostic.json").write_text(
        json.dumps(out, indent=2)
    )
    print(f"\nSaved runs/selfplay/router_diagnostic.json")


if __name__ == "__main__":
    main()
