"""Cluster the perturbations produced by a trained attacker (or
attacker-MoE) so we can show the experts have specialised on different
attack patterns and discover modes that don't match any of the five
hand-crafted classes.

Workflow:
    1. Load defender + attacker checkpoints.
    2. Iterate over the test set; record one (delta_p_vector,
       attack_class_label) per snapshot from the attacker output.
    3. Project to 2D with t-SNE.
    4. Cluster with HDBSCAN; report cluster -> attack_class purity.
    5. Save the projection PNG.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from wdn.models.attacker import AttackerGNN, StealthBudget, apply_stealth_budget
from wdn.models.attacker_moe import MixtureOfAttackersGNN
from wdn.temporal_dataset import create_temporal_dataloaders


THEME = {
    "BG": "#0f172a", "WHITE": "#f1f5f9", "GRAY": "#64748b",
    "DIM": "#475569",
    "PALETTE": ["#60a5fa", "#a78bfa", "#4ade80", "#fb923a",
                "#f87171", "#22d3ee"],
}
plt.rcParams.update({
    "figure.facecolor": THEME["BG"], "axes.facecolor": THEME["BG"],
    "axes.edgecolor": THEME["DIM"], "text.color": THEME["WHITE"],
    "axes.labelcolor": THEME["GRAY"],
    "xtick.color": THEME["GRAY"], "ytick.color": THEME["GRAY"],
    "font.family": "sans-serif", "font.size": 10,
    "axes.grid": True, "grid.color": "#1e293b", "grid.alpha": 0.6,
})

ATTACK_NAMES = {
    0: "clean", 1: "random", 2: "replay",
    3: "stealthy", 4: "noise", 5: "targeted",
}


def collect(attacker, loader, device, budget, num_nodes):
    """Run the attacker over the test loader and pad each batch's
    delta_p back to a per-graph (B, N) tensor so every snapshot
    becomes a fixed-length vector for clustering."""
    deltas, labels, expert_picks = [], [], []
    is_moe = isinstance(attacker, MixtureOfAttackersGNN)
    with torch.no_grad():
        for raw in loader:
            b = {k: (v.to(device) if isinstance(v, torch.Tensor)
                     else [t.to(device) for t in v] if isinstance(v, list) else v)
                 for k, v in raw.items()}
            x_last = b["x_seq"][-1]
            if is_moe:
                out = attacker(
                    x_last, b["edge_index"], b["edge_attr"],
                    b["is_original_edge"],
                    num_nodes_per_graph=b["num_nodes"],
                )
                router_pick = out["router_probs"].argmax(dim=-1).cpu().numpy()
            else:
                out = attacker(
                    x_last, b["edge_index"], b["edge_attr"],
                    b["is_original_edge"],
                )
                router_pick = np.full(b["batch_size"], -1)
            proj = apply_stealth_budget(out, budget, hard=True)
            dp = proj["delta_p"].view(b["batch_size"], num_nodes).cpu().numpy()
            deltas.append(dp)
            labels.append(b["attack_type"].cpu().numpy())
            expert_picks.append(router_pick)
    return (np.concatenate(deltas, axis=0),
            np.concatenate(labels, axis=0),
            np.concatenate(expert_picks, axis=0))


def main(
    attacker_ckpt: str, data_dir: str, hidden_dim: int,
    is_moe: bool, num_experts: int, out_name: str,
):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data_dir = Path(data_dir)
    snaps = pickle.load(open(data_dir / "snapshots.pkl", "rb"))
    corr = pickle.load(open(data_dir / "corrupted.pkl", "rb"))
    n = len(snaps); n_train = int(0.7 * n); n_val = int(0.15 * n)
    _, _, test_loader, _ = create_temporal_dataloaders(
        snaps[:n_train], corr[:n_train],
        snaps[n_train:n_train+n_val], corr[n_train:n_train+n_val],
        snaps[n_train+n_val:], corr[n_train+n_val:],
        window_size=6, batch_size=8,
    )

    sample = next(iter(test_loader))
    node_dim = sample["x_seq"][0].shape[1]
    edge_dim = sample["edge_attr"].shape[1]
    N = sample["num_nodes"]

    if is_moe:
        attacker = MixtureOfAttackersGNN(
            node_in_dim=node_dim, edge_in_dim=edge_dim,
            hidden_dim=hidden_dim, num_experts=num_experts,
        ).to(device)
    else:
        attacker = AttackerGNN(
            node_in_dim=node_dim, edge_in_dim=edge_dim,
            hidden_dim=hidden_dim,
        ).to(device)
    attacker.load_state_dict(torch.load(attacker_ckpt, map_location=device))
    attacker.eval()

    budget = StealthBudget(epsilon_p=5.0, epsilon_q=0.05, k_p=15, k_q=4)
    deltas, labels, picks = collect(attacker, test_loader, device, budget, N)

    print(f"Collected {len(deltas)} attack vectors of dim {deltas.shape[1]}")
    print(f"Attack-class distribution: "
          f"{np.bincount(labels.astype(int), minlength=6).tolist()}")
    if is_moe:
        print(f"Router-pick distribution: "
              f"{np.bincount(picks.astype(int), minlength=num_experts).tolist()}")

    # ---- 2D projection ----
    try:
        from sklearn.manifold import TSNE
        emb = TSNE(n_components=2, random_state=0,
                   perplexity=min(30, max(5, len(deltas) // 5)),
                   init="pca", learning_rate="auto").fit_transform(deltas)
    except Exception as e:
        print(f"t-SNE fell back to PCA: {e}")
        from sklearn.decomposition import PCA
        emb = PCA(n_components=2).fit_transform(deltas)

    fig, axes = plt.subplots(1, 2 if is_moe else 1,
                             figsize=(11 if is_moe else 5.5, 4.2))
    if not is_moe:
        axes = [axes]

    # Left: coloured by ground-truth attack class
    ax = axes[0]
    for cls in sorted(set(labels.tolist())):
        sel = labels == cls
        ax.scatter(emb[sel, 0], emb[sel, 1], s=10, alpha=0.7,
                   color=THEME["PALETTE"][int(cls) % len(THEME["PALETTE"])],
                   label=ATTACK_NAMES.get(int(cls), str(cls)))
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.set_title("Coloured by ground-truth attack class",
                 fontsize=10.5, fontweight="bold", color=THEME["WHITE"])
    ax.legend(fontsize=8, framealpha=0, loc="best", ncol=2)

    if is_moe:
        ax = axes[1]
        for ex in range(num_experts):
            sel = picks == ex
            if sel.sum() == 0:
                continue
            ax.scatter(emb[sel, 0], emb[sel, 1], s=10, alpha=0.7,
                       color=THEME["PALETTE"][ex % len(THEME["PALETTE"])],
                       label=f"Expert {ex}")
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        ax.set_title("Coloured by attacker expert (router pick)",
                     fontsize=10.5, fontweight="bold", color=THEME["WHITE"])
        ax.legend(fontsize=8, framealpha=0, loc="best")

    fig.suptitle("Attack vocabulary — t-SNE projection",
                 fontsize=11.5, fontweight="bold", color=THEME["WHITE"], y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = ROOT / "presentation" / "charts" / out_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

    # ---- Expert -> attack class purity table ----
    if is_moe:
        print("\nExpert <-> attack class purity:")
        for ex in range(num_experts):
            sel = picks == ex
            if sel.sum() == 0:
                continue
            counts = np.bincount(labels[sel].astype(int), minlength=6)
            top = int(counts.argmax())
            purity = counts[top] / counts.sum()
            print(f"  Expert {ex}: n={sel.sum()}  top={ATTACK_NAMES[top]} "
                  f"({purity:.0%})  counts={counts.tolist()}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--attacker_ckpt", required=True)
    p.add_argument("--data_dir", default="data/temporal_moe_modena")
    p.add_argument("--hidden_dim", type=int, default=48)
    p.add_argument("--moe", action="store_true")
    p.add_argument("--num_experts", type=int, default=4)
    p.add_argument("--out", default="vocab_attackmoe.png")
    a = p.parse_args()
    main(a.attacker_ckpt, a.data_dir, a.hidden_dim,
         a.moe, a.num_experts, a.out)
