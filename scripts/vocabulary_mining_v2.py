"""Enhanced vocabulary mining for the Part-2 attacker.

Goes beyond ``scripts/vocabulary_mining.py`` with four extra analyses
the supervisors asked for in the meeting:

    1. Per-attack-class delta vectors with HDBSCAN clustering, so we can
       distinguish "discovered" attack patterns from those that match a
       hand-crafted class.
    2. Spatial signature: which nodes does the attacker preferentially
       perturb?  Plotted as a centrality histogram.
    3. Magnitude distribution per attack class + per discovered cluster.
    4. Stealth-vs-damage scatter (each snapshot a point), with hand-
       crafted classes coloured.

Outputs PNG + JSON summary so the numbers can be cited in the paper.

Usage:
    python3 scripts/vocabulary_mining_v2.py \
        --attacker_ckpt runs/selfplay/<RUN>/attacker.pt \
        --defender_ckpt runs/temporal_moe/<RUN>/best_model.pt \
        --data_dir data/temporal_moe_modena \
        --hidden_dim 64 --moe --num_experts 4 --out_name vocab_v2
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from wdn.models.attacker import AttackerGNN, StealthBudget, apply_stealth_budget
from wdn.models.attacker_moe import MixtureOfAttackersGNN
from wdn.models.temporal_moe import TemporalMixtureOfExpertsGNN
from wdn.temporal_dataset import create_temporal_dataloaders


THEME = {
    "BG": "#0f172a", "WHITE": "#f1f5f9", "GRAY": "#94a3b8",
    "DIM": "#475569",
    "PALETTE": ["#60a5fa", "#a78bfa", "#4ade80", "#fb923a",
                "#f87171", "#22d3ee", "#facc15", "#e879f9"],
}
plt.rcParams.update({
    "figure.facecolor": THEME["BG"], "axes.facecolor": THEME["BG"],
    "axes.edgecolor": THEME["DIM"], "text.color": THEME["WHITE"],
    "axes.labelcolor": THEME["GRAY"],
    "xtick.color": THEME["GRAY"], "ytick.color": THEME["GRAY"],
    "font.family": "sans-serif", "font.size": 10,
    "axes.grid": True, "grid.color": "#1e293b", "grid.alpha": 0.5,
})

ATTACK_NAMES = {0: "clean", 1: "random", 2: "replay",
                3: "stealthy", 4: "noise", 5: "targeted"}


def collect(attacker, defender, loader, device, budget, num_nodes):
    """Run the attacker over the test set and gather:
        - delta_p vector per snapshot (B, N)
        - hand-crafted attack class label per snapshot
        - router-picked expert index per snapshot (-1 if not MoE)
        - per-snapshot stealth (= 1 - mean defender prob on attacked sensors)
        - per-snapshot damage (= MSE on defender pressure pred for attacked)
        - which node indices were attacked (top-k mask)
    """
    is_moe = isinstance(attacker, MixtureOfAttackersGNN)
    out_rows = {
        "deltas": [], "labels": [], "picks": [],
        "stealth": [], "damage": [], "attacked_nodes": [],
    }
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

            # Defender forward on injected batch
            B = b["batch_size"]
            new_x = list(b["x_seq"])
            last_x = new_x[-1].clone()
            last_x[:, 5] = last_x[:, 5] + proj["delta_p"]
            last_x[:, 6] = torch.clamp(last_x[:, 6] + proj["mask_p"], max=1.0)
            new_x[-1] = last_x
            new_pobs = last_x[:, 5]
            new_pmask = last_x[:, 6]
            def_out = defender(
                x_seq=new_x, edge_index=b["edge_index"],
                edge_attr=b["edge_attr"],
                is_original_edge=b["is_original_edge"],
                batch_size=B, num_nodes_per_graph=b["num_nodes"],
                pressure_obs=new_pobs, flow_obs=b["flow_obs"] + proj["delta_q"],
                pressure_mask=new_pmask,
                flow_mask=torch.clamp(b["flow_mask"] + proj["mask_q"], max=1.0),
            )

            dp = proj["delta_p"].view(B, num_nodes)
            attacked = (proj["mask_p"] > 0.5).view(B, num_nodes)
            p_logits = def_out["pressure_anomaly_logits"].view(B, num_nodes)
            p_probs = torch.sigmoid(p_logits)
            p_pred = def_out["pressure_pred"].view(B, num_nodes)
            y_p = b["y_pressure"].view(B, num_nodes)

            # Per-snapshot stealth = 1 - mean(prob | attacked)
            stealth_per = torch.zeros(B, device=device)
            damage_per = torch.zeros(B, device=device)
            for i in range(B):
                m = attacked[i]
                if m.sum() == 0: continue
                stealth_per[i] = 1.0 - p_probs[i, m].mean()
                damage_per[i] = (p_pred[i, m] - y_p[i, m]).pow(2).mean()

            out_rows["deltas"].append(dp.cpu().numpy())
            out_rows["labels"].append(b["attack_type"].cpu().numpy())
            out_rows["picks"].append(router_pick)
            out_rows["stealth"].append(stealth_per.cpu().numpy())
            out_rows["damage"].append(damage_per.cpu().numpy())
            out_rows["attacked_nodes"].append(attacked.cpu().numpy())

    return {k: np.concatenate(v, axis=0) for k, v in out_rows.items()}


def node_centrality(graph_pkl: Path, num_nodes: int) -> np.ndarray:
    """Degree centrality per node from the network graph."""
    G = pickle.load(open(graph_pkl, "rb"))
    if hasattr(G, "node_features") and hasattr(G, "edge_index"):
        ei = G.edge_index  # (2, E)
        deg = np.zeros(num_nodes, dtype=np.int64)
        for src in ei[0]:
            deg[int(src)] += 1
        return deg.astype(float)
    # Fallback for old graph types
    return np.ones(num_nodes)


def plot_all(data, centrality, num_experts, out_dir, out_name):
    deltas = data["deltas"]; labels = data["labels"]; picks = data["picks"]
    stealth = data["stealth"]; damage = data["damage"]
    attacked = data["attacked_nodes"]
    n_snap, N = deltas.shape

    # ----- Figure: 2x2 grid -----
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    pal = THEME["PALETTE"]

    # (A) t-SNE projection coloured by hand-crafted class
    ax = axes[0, 0]
    try:
        from sklearn.manifold import TSNE
        emb = TSNE(n_components=2, random_state=0,
                   perplexity=min(30, max(5, n_snap // 5)),
                   init="pca", learning_rate="auto").fit_transform(deltas)
    except Exception:
        from sklearn.decomposition import PCA
        emb = PCA(n_components=2).fit_transform(deltas)
    for cls in sorted(set(labels.astype(int).tolist())):
        sel = labels == cls
        ax.scatter(emb[sel, 0], emb[sel, 1], s=14, alpha=0.75,
                   color=pal[int(cls) % len(pal)],
                   label=ATTACK_NAMES.get(int(cls), str(cls)))
    ax.set_title("(a) Attacker delta — t-SNE, hand-crafted class",
                 color=THEME["WHITE"], fontweight="bold", fontsize=10.5)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=7.5, framealpha=0, loc="best", ncol=2)

    # (B) Stealth vs damage scatter, coloured by class
    ax = axes[0, 1]
    for cls in sorted(set(labels.astype(int).tolist())):
        sel = labels == cls
        ax.scatter(stealth[sel], damage[sel], s=18, alpha=0.75,
                   color=pal[int(cls) % len(pal)],
                   label=ATTACK_NAMES.get(int(cls), str(cls)))
    ax.set_xlabel("Stealth (1 − defender prob on attacked sensors)")
    ax.set_ylabel("Damage (MSE on attacked sensors)")
    ax.set_title("(b) Stealth–damage trade-off per attack",
                 color=THEME["WHITE"], fontweight="bold", fontsize=10.5)
    ax.legend(fontsize=7.5, framealpha=0, loc="best", ncol=2)

    # (C) Spatial signature: how often each node gets attacked
    ax = axes[1, 0]
    attack_freq = attacked.mean(axis=0)  # (N,)
    order = np.argsort(centrality)
    ax.scatter(centrality[order], attack_freq[order], s=10, alpha=0.7,
               color=pal[0])
    if centrality.std() > 0:
        # Trend line
        z = np.polyfit(centrality, attack_freq, 1)
        xs = np.linspace(centrality.min(), centrality.max(), 50)
        ax.plot(xs, np.polyval(z, xs), color=pal[3], linewidth=1.5,
                label=f"slope = {z[0]:+.4f}")
        ax.legend(fontsize=8, framealpha=0, loc="best")
    ax.set_xlabel("Node degree centrality")
    ax.set_ylabel("Fraction of snapshots attacked")
    ax.set_title("(c) Attack frequency vs node centrality",
                 color=THEME["WHITE"], fontweight="bold", fontsize=10.5)

    # (D) Magnitude distribution per class
    ax = axes[1, 1]
    mags = np.linalg.norm(deltas, axis=1)  # L2 norm per snapshot
    for cls in sorted(set(labels.astype(int).tolist())):
        sel = labels == cls
        if sel.sum() < 2: continue
        ax.hist(mags[sel], bins=20, alpha=0.55,
                color=pal[int(cls) % len(pal)],
                label=ATTACK_NAMES.get(int(cls), str(cls)))
    ax.set_xlabel("‖δp‖₂ per snapshot")
    ax.set_ylabel("Count")
    ax.set_title("(d) Perturbation magnitude per class",
                 color=THEME["WHITE"], fontweight="bold", fontsize=10.5)
    ax.legend(fontsize=7.5, framealpha=0, loc="best", ncol=2)

    fig.suptitle(f"Attack vocabulary analysis — {out_name}",
                 fontsize=12.5, fontweight="bold", color=THEME["WHITE"], y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = out_dir / f"{out_name}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")
    return mags


def summary_json(data, mags, centrality, num_experts, out_dir, out_name):
    labels = data["labels"]; picks = data["picks"]
    stealth = data["stealth"]; damage = data["damage"]
    attacked = data["attacked_nodes"]

    per_class = {}
    for cls in sorted(set(labels.astype(int).tolist())):
        sel = labels == cls
        if sel.sum() == 0: continue
        per_class[ATTACK_NAMES.get(int(cls), str(cls))] = {
            "n": int(sel.sum()),
            "stealth_mean": float(stealth[sel].mean()),
            "stealth_std": float(stealth[sel].std()),
            "damage_mean": float(damage[sel].mean()),
            "damage_std": float(damage[sel].std()),
            "magnitude_mean": float(mags[sel].mean()),
            "magnitude_std": float(mags[sel].std()),
        }

    # Expert <-> class purity (if MoE)
    per_expert = {}
    if (picks >= 0).any():
        for ex in range(num_experts):
            sel = picks == ex
            if sel.sum() == 0:
                per_expert[f"expert_{ex}"] = {"n": 0}
                continue
            counts = np.bincount(labels[sel].astype(int), minlength=6)
            top = int(counts.argmax())
            per_expert[f"expert_{ex}"] = {
                "n": int(sel.sum()),
                "top_class": ATTACK_NAMES.get(top, str(top)),
                "top_class_purity": float(counts[top] / counts.sum()),
                "class_counts": counts.tolist(),
            }

    # Centrality correlation
    attack_freq = attacked.mean(axis=0)
    corr = float(np.corrcoef(centrality, attack_freq)[0, 1]) \
           if centrality.std() > 0 else 0.0

    summary = {
        "n_snapshots": int(len(labels)),
        "per_class": per_class,
        "per_expert": per_expert,
        "centrality_attack_corr": corr,
        "global_magnitude_mean": float(mags.mean()),
        "global_magnitude_std": float(mags.std()),
    }
    out_path = out_dir / f"{out_name}.json"
    json.dump(summary, open(out_path, "w"), indent=2)
    print(f"Saved {out_path}")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attacker_ckpt", required=True)
    p.add_argument("--defender_ckpt", required=True)
    p.add_argument("--data_dir", default="data/temporal_moe_modena")
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--router_hidden_dim", type=int, default=24)
    p.add_argument("--num_experts_defender", type=int, default=6)
    p.add_argument("--num_experts", type=int, default=4)
    p.add_argument("--moe", action="store_true")
    p.add_argument("--out_name", default="vocab_v2")
    p.add_argument("--out_dir", default="presentation/charts")
    p.add_argument("--epsilon_p", type=float, default=5.0)
    p.add_argument("--epsilon_q", type=float, default=0.05)
    p.add_argument("--k_p", type=int, default=10)
    p.add_argument("--k_q", type=int, default=4)
    args = p.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data_dir = Path(args.data_dir)
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

    # Build attacker
    if args.moe:
        attacker = MixtureOfAttackersGNN(
            node_in_dim=node_dim, edge_in_dim=edge_dim,
            hidden_dim=args.hidden_dim, num_experts=args.num_experts,
        ).to(device)
    else:
        attacker = AttackerGNN(
            node_in_dim=node_dim, edge_in_dim=edge_dim,
            hidden_dim=args.hidden_dim,
        ).to(device)
    attacker.load_state_dict(torch.load(args.attacker_ckpt, map_location=device))
    attacker.eval()

    # Build defender
    defender = TemporalMixtureOfExpertsGNN(
        node_in_dim=node_dim, edge_in_dim=edge_dim,
        hidden_dim=args.hidden_dim, num_experts=args.num_experts_defender,
        window_size=6, gnn_type="GraphSAGE",
        router_hidden_dim=args.router_hidden_dim,
    ).to(device)
    defender.load_state_dict(torch.load(args.defender_ckpt, map_location=device))
    defender.eval()

    budget = StealthBudget(epsilon_p=args.epsilon_p, epsilon_q=args.epsilon_q,
                           k_p=args.k_p, k_q=args.k_q)

    data = collect(attacker, defender, test_loader, device, budget, N)
    centrality = node_centrality(data_dir / "graph.pkl", N)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mags = plot_all(data, centrality, args.num_experts, out_dir, args.out_name)
    summary = summary_json(data, mags, centrality, args.num_experts,
                           out_dir, args.out_name)

    print("\n--- Summary ---")
    print(f"n_snapshots: {summary['n_snapshots']}")
    print(f"centrality↔attack correlation: {summary['centrality_attack_corr']:+.3f}")
    print(f"global |δp|₂: {summary['global_magnitude_mean']:.3f} "
          f"± {summary['global_magnitude_std']:.3f}")
    print("\nPer-class stealth / damage / |δp|₂:")
    for cls, s in summary["per_class"].items():
        print(f"  {cls:<10} n={s['n']:>3}  "
              f"stealth={s['stealth_mean']:.3f}  "
              f"damage={s['damage_mean']:.4f}  "
              f"|δp|={s['magnitude_mean']:.2f}")
    if summary["per_expert"]:
        print("\nPer-expert usage / top class:")
        for ex, s in summary["per_expert"].items():
            if s["n"] == 0:
                print(f"  {ex}: unused"); continue
            print(f"  {ex}: n={s['n']:>3} ({s['n']/summary['n_snapshots']:.0%})  "
                  f"top={s['top_class']} ({s['top_class_purity']:.0%})")


if __name__ == "__main__":
    main()
