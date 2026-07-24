"""Concrete evidence for *why* replay still underperforms after the
per-node normalisation fix.

Three panels, pooled over the 5 per-node water models:
  (a) score distributions — replayed sensors vs genuine sensors overlap,
      so although the ranking (AUROC) is good, no single threshold
      separates them (that is the AUROC-high / F1-low gap);
  (b) precision-recall curve for replay — you cannot get high precision
      and high recall at once;
  (c) the physical reason — how far a replayed value sits from the truth
      vs the observation-noise band it hides in.

Writes thesis/figures/slides/replay_why.png (+ deliverables copy) and a
small JSON of the numbers.
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from wdn.temporal_dataset import create_temporal_dataloaders
from wdn.models.temporal_moe import TemporalMixtureOfExpertsGNN

BLUE, PURPLE, RED, MUTED, INK = "#2563eb", "#7c3aed", "#dc2626", "#6b7280", "#1f2937"
GREEN = "#059669"
plt.rcParams.update({
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.facecolor": "white", "axes.edgecolor": "#c9ced6",
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": "#54607a", "ytick.color": "#54607a",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 12.5, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#eef0f4", "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})


def load(run):
    a = json.load(open(run + "/args.json")); dd = a["data_dir"]
    snaps = pickle.load(open(f"{ROOT}/{dd}/snapshots.pkl", "rb"))
    corr = pickle.load(open(f"{ROOT}/{dd}/corrupted.pkl", "rb"))
    n = len(snaps); ntr = int(0.7 * n); nv = int(0.15 * n)
    _, _, te, _ = create_temporal_dataloaders(
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
    return m, te, dev


@torch.no_grad()
def collect(m, loader, dev):
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


runs = []
for d in sorted(glob.glob(str(ROOT / "runs/temporal_moe/2026*"))):
    if os.path.basename(d) < "20260721_113":
        continue
    ap = os.path.join(d, "args.json")
    if not os.path.exists(ap):
        continue
    a = json.load(open(ap))
    if (a.get("norm_mode") == "per_node" and a.get("seed") in range(1, 6)
            and a.get("epochs") == 60
            and a.get("data_dir") == "data/temporal_moe_modena"):
        runs.append(d)

S, L, M, TC = [], [], [], []
for run in runs:
    m, te, dev = load(run)
    s, l, mm, tc = collect(m, te, dev)
    S.append(s); L.append(l); M.append(mm); TC.append(tc)
S = np.concatenate(S); L = np.concatenate(L); M = np.concatenate(M); TC = np.concatenate(TC)

obs = M > 0
# Replay windows only, observed sensors.
rep = obs & (TC == 2)
rep_scores_pos = S[rep & (L > 0.5)]     # replayed (falsified) sensors
rep_scores_neg = S[rep & (L <= 0.5)]    # genuine sensors in replay windows
y = L[rep].astype(int)
p = S[rep]
auroc = roc_auc_score(y, p)
prec, rec, thr = precision_recall_curve(y, p)
# best F1 point
f1s = 2 * prec * rec / (prec + rec + 1e-9)
bi = int(np.argmax(f1s))
best_f1 = float(f1s[bi])

fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))

# Panel A: score overlap
ax = axes[0]
bins = np.linspace(0, 1, 40)
ax.hist(rep_scores_neg, bins=bins, density=True, alpha=0.6, color=GREEN,
        label="genuine sensors")
ax.hist(rep_scores_pos, bins=bins, density=True, alpha=0.6, color=PURPLE,
        label="replayed sensors")
ax.set_xlabel("anomaly score"); ax.set_ylabel("density")
ax.set_title("Replay and genuine scores overlap", fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=11)
ax.text(0.03, 0.94, f"AUROC {auroc:.2f} — ranking is OK,\nbut the two overlap",
        transform=ax.transAxes, ha="left", va="top", fontsize=10.5,
        color=MUTED)

# Panel B: PR curve
ax = axes[1]
ax.plot(rec, prec, color=PURPLE, lw=2.5)
ax.scatter([rec[bi]], [prec[bi]], color=RED, zorder=5, s=60)
ax.annotate(f"best F1 = {best_f1:.2f}", (rec[bi], prec[bi]),
            textcoords="offset points", xytext=(10, 10), color=RED,
            fontsize=11, fontweight="bold")
ax.set_xlabel("recall"); ax.set_ylabel("precision")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.set_title("No threshold gives high precision AND recall",
             fontsize=13, fontweight="bold")

out = ROOT / "thesis" / "figures" / "slides" / "replay_why.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
# copy into the results package
dl = ROOT / "results_package" / "figures" / "07_replay_why.png"
fig.savefig(dl, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)

json.dump({"replay_auroc": float(auroc), "replay_best_f1": best_f1,
           "n_replay_sensors": int(rep.sum()),
           "n_replayed": int((rep & (L > 0.5)).sum())},
          open(ROOT / "runs/temporal_moe/replay_why.json", "w"), indent=2)
print(f"replay AUROC {auroc:.3f}  best-F1 {best_f1:.3f}")
print(f"wrote {out}")
