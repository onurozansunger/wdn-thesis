"""Generate the killer self-play plots: co-evolution loss curves and
attacker-vs-defender F1 trajectories. Reads ``history.json`` produced
by ``train_selfplay`` and writes PNGs to ``presentation/charts/``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
OUT = ROOT / "presentation" / "charts"
OUT.mkdir(exist_ok=True)

# Theme (same as generate_charts.py)
BG = "#0f172a"; CARD = "#1e293b"
BLUE = "#60a5fa"; PURPLE = "#a78bfa"; GREEN = "#4ade80"
ORANGE = "#fb923a"; RED = "#f87171"; WHITE = "#f1f5f9"
GRAY = "#64748b"; DIMGRAY = "#475569"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": DIMGRAY, "text.color": WHITE,
    "axes.labelcolor": GRAY, "xtick.color": GRAY, "ytick.color": GRAY,
    "font.family": "sans-serif", "font.size": 10,
    "axes.grid": True, "grid.color": "#1e293b", "grid.alpha": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})


def plot_coevolution(history_path: Path, out_name: str, title: str):
    h = json.load(open(history_path))
    epochs = [e["epoch"] for e in h]
    hand = [e["hand_f1"] for e in h]
    adv = [e["adv_f1"] for e in h]
    dmg = [e["atk_damage"] for e in h]
    recon = [e["def_recon"] for e in h]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))

    # ---- Left: F1 curves ----
    ax1.plot(epochs, hand, color=BLUE, lw=2.2, label="Defender F1 — hand-crafted")
    ax1.plot(epochs, adv, color=PURPLE, lw=2.2, label="Defender F1 — vs attacker")
    ax1.fill_between(epochs, hand, adv,
                     where=[a >= h_ for a, h_ in zip(adv, hand)],
                     color=PURPLE, alpha=0.12)
    # Annotate curriculum jumps if k_p changes
    ks = [e.get("k_p", None) for e in h]
    last_k = ks[0] if ks else None
    for i, k in enumerate(ks):
        if k is not None and last_k is not None and k > last_k:
            ax1.axvline(epochs[i], color=GREEN, lw=0.8, ls="--", alpha=0.6)
            last_k = k
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("F1")
    ax1.set_ylim(0.4, 1.0)
    ax1.set_title("Co-evolution: defender F1 vs attacker pressure",
                  fontsize=10.5, fontweight="bold", color=WHITE, pad=6)
    ax1.legend(fontsize=9, framealpha=0, loc="lower right")

    # ---- Right: damage + reconstruction ----
    ax2.plot(epochs, dmg, color=RED, lw=2.0, label="Attacker damage (MSE)")
    ax2b = ax2.twinx()
    ax2b.plot(epochs, recon, color=GREEN, lw=2.0, label="Defender recon loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Attacker damage", color=RED)
    ax2b.set_ylabel("Defender recon loss", color=GREEN)
    ax2.tick_params(axis="y", labelcolor=RED)
    ax2b.tick_params(axis="y", labelcolor=GREEN)
    ax2b.spines["top"].set_visible(False)
    ax2.set_title("Adversarial dynamics", fontsize=10.5,
                  fontweight="bold", color=WHITE, pad=6)
    # Combined legend
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=8.5, framealpha=0, loc="upper right")

    fig.suptitle(title, fontsize=11.5, fontweight="bold",
                 color=WHITE, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / out_name, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved {out_name}")


def plot_per_attack_compare(eval_path: Path, out_name: str):
    d = json.load(open(eval_path))
    pre = d["pretrained"]["per_attack"]
    sp = d["selfplay"]["per_attack"]
    attacks = ["random", "replay", "stealthy", "noise", "targeted"]
    pre_f1 = [pre.get(a, {}).get("f1", 0.0) for a in attacks]
    sp_f1 = [sp.get(a, {}).get("f1", 0.0) for a in attacks]

    fig, ax = plt.subplots(figsize=(6.5, 3))
    x = np.arange(len(attacks)); w = 0.34
    ax.bar(x - w/2, pre_f1, w, label="Pretrained", color=BLUE, alpha=0.6)
    ax.bar(x + w/2, sp_f1, w, label="+ Self-play", color=GREEN, alpha=0.9)
    for i, (a_, b_) in enumerate(zip(pre_f1, sp_f1)):
        ax.text(i - w/2, a_ + 0.01, f"{a_:.2f}", ha="center", fontsize=8, color=WHITE)
        ax.text(i + w/2, b_ + 0.01, f"{b_:.2f}", ha="center",
                fontsize=8, fontweight="bold", color=WHITE)
    ax.set_xticks(x); ax.set_xticklabels(attacks, fontsize=9)
    ax.set_ylabel("F1"); ax.set_ylim(0, 1.22)
    ax.set_title("Per-attack F1 — Self-play vs Pretrained (Modena)",
                 fontsize=11, fontweight="bold", color=WHITE, pad=26)
    ax.legend(fontsize=9, framealpha=0, loc="lower center",
              bbox_to_anchor=(0.5, 1.01), ncol=2, columnspacing=1.8)
    plt.tight_layout()
    fig.savefig(OUT / out_name, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved {out_name}")


def plot_three_network_compare(eval_path: Path, out_name: str):
    """Bar chart: Pretrained vs Self-play across Net1, Net3, Modena."""
    d = json.load(open(eval_path))
    nets = ["Net1", "Net3", "Modena"]
    metrics = [("f1", "Anomaly F1"), ("auroc", "AUROC")]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for ax, (mkey, mlabel) in zip(axes, metrics):
        pre = [d[n]["pretrained"][mkey] for n in nets]
        sp = [d[n]["selfplay"][mkey] for n in nets]
        x = np.arange(len(nets)); w = 0.34
        ax.bar(x - w/2, pre, w, label="Pretrained", color=BLUE, alpha=0.6)
        ax.bar(x + w/2, sp, w, label="+ Self-play", color=GREEN, alpha=0.9)
        for i, (p, s) in enumerate(zip(pre, sp)):
            ax.text(i - w/2, p + 0.005, f"{p:.2f}", ha="center", fontsize=8, color=WHITE)
            ax.text(i + w/2, s + 0.005, f"{s:.2f}", ha="center",
                    fontsize=8, fontweight="bold", color=WHITE)
        ax.set_xticks(x); ax.set_xticklabels(nets)
        ax.set_ylabel(mlabel); ax.set_ylim(0.5, 1.05)
        ax.set_title(mlabel, fontsize=10.5, fontweight="bold", color=WHITE)
        ax.legend(fontsize=9, framealpha=0, loc="lower right")
    fig.suptitle("Self-play across three networks",
                 fontsize=11.5, fontweight="bold", color=WHITE, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / out_name, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved {out_name}")


def plot_multiseed(eval_path: Path, out_name: str):
    """Per-attack F1 with error bars across 3 seeds."""
    d = json.load(open(eval_path))
    pre = d["pretrained"]
    sps = d["selfplay_seeds"]
    attacks = ["random", "replay", "stealthy", "noise", "targeted"]
    pre_f1 = [pre["per_attack"].get(a, {}).get("f1", 0.0) for a in attacks]
    sp_arr = []
    for sp in sps:
        sp_arr.append([sp["per_attack"].get(a, {}).get("f1", 0.0) for a in attacks])
    sp_arr = np.array(sp_arr)
    sp_mean = sp_arr.mean(axis=0)
    sp_std = sp_arr.std(axis=0, ddof=1) if sp_arr.shape[0] > 1 else np.zeros(len(attacks))

    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(attacks)); w = 0.34
    ax.bar(x - w/2, pre_f1, w, label="Pretrained", color=BLUE, alpha=0.6)
    ax.bar(x + w/2, sp_mean, w, yerr=sp_std, capsize=3,
           label="Self-play (3 seeds)", color=GREEN, alpha=0.9,
           error_kw={"ecolor": WHITE, "lw": 1.2})
    for i, (p, m_, s_) in enumerate(zip(pre_f1, sp_mean, sp_std)):
        ax.text(i - w/2, p + 0.01, f"{p:.2f}", ha="center", fontsize=8, color=WHITE)
        ax.text(i + w/2, m_ + s_ + 0.015, f"{m_:.2f}", ha="center",
                fontsize=8, fontweight="bold", color=WHITE)
    ax.set_xticks(x); ax.set_xticklabels(attacks, fontsize=9)
    ax.set_ylabel("F1"); ax.set_ylim(0, 1.25)
    ax.set_title("Per-attack F1 — Self-play vs Pretrained (Modena, 3 seeds)",
                 fontsize=11, fontweight="bold", color=WHITE, pad=26)
    ax.legend(fontsize=9, framealpha=0, loc="lower center",
              bbox_to_anchor=(0.5, 1.01), ncol=2, columnspacing=1.8)
    plt.tight_layout()
    fig.savefig(OUT / out_name, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved {out_name}")


def plot_heldout(eval_path: Path, out_name: str):
    """Bar chart: defenders evaluated on novel attacks not in training."""
    d = json.load(open(eval_path))
    kinds = list(d.keys())
    defs = ["pretrained", "sp_single", "sp_moe"]
    labels = ["Pretrained", "SP Single", "SP MoE"]
    colors = [BLUE, PURPLE, GREEN]

    fig, axes = plt.subplots(1, len(kinds), figsize=(5 * len(kinds), 3.4))
    if len(kinds) == 1:
        axes = [axes]
    for ax, kind in zip(axes, kinds):
        f1 = [d[kind][n]["f1"] for n in defs]
        auroc = [d[kind][n]["auroc"] for n in defs]
        x = np.arange(len(defs)); w = 0.34
        ax.bar(x - w/2, f1, w, label="F1",
               color=[BLUE]*len(defs), alpha=0.6)
        ax.bar(x + w/2, auroc, w, label="AUROC",
               color=[PURPLE]*len(defs), alpha=0.85)
        for i, (a, b) in enumerate(zip(f1, auroc)):
            ax.text(i - w/2, a + 0.005, f"{a:.3f}", ha="center",
                    fontsize=8, color=WHITE)
            ax.text(i + w/2, b + 0.005, f"{b:.3f}", ha="center",
                    fontsize=8, fontweight="bold", color=WHITE)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0.6, 1.05)
        ax.set_title(f"Novel attack: {kind}",
                     fontsize=10.5, fontweight="bold", color=WHITE, pad=24)
        ax.legend(fontsize=8, framealpha=0, loc="lower center",
                  bbox_to_anchor=(0.5, 1.01), ncol=2, columnspacing=1.6)
    plt.tight_layout()
    fig.savefig(OUT / out_name, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved {out_name}")



def plot_multiseed_compare(net1_path: Path | None, net3_path: Path,
                            modena_path: Path, out_name: str):
    """Multi-seed per-attack F1 across all three networks (mean ± std).
    Net1 path is allowed to be None — only Modena and Net3 had three
    seeds run."""
    nets = []
    if modena_path.exists():
        nets.append(("Modena", json.load(open(modena_path))))
    if net3_path.exists():
        nets.append(("Net3", json.load(open(net3_path))))
    if net1_path and Path(net1_path).exists():
        nets.append(("Net1", json.load(open(net1_path))))

    if not nets:
        print(f"skip {out_name} — no eval files")
        return

    attacks = ["random", "replay", "stealthy", "noise", "targeted"]
    fig, axes = plt.subplots(1, len(nets), figsize=(4.5 * len(nets), 3.3),
                             sharey=True)
    if len(nets) == 1:
        axes = [axes]
    for ax, (name, d) in zip(axes, nets):
        pre = d["pretrained"]
        sps = d["selfplay_seeds"]
        pre_f1 = [pre["per_attack"].get(a, {}).get("f1", 0.0) for a in attacks]
        sp_arr = np.array([
            [sp["per_attack"].get(a, {}).get("f1", 0.0) for a in attacks]
            for sp in sps
        ])
        sp_mean = sp_arr.mean(axis=0)
        sp_std = sp_arr.std(axis=0, ddof=1) if sp_arr.shape[0] > 1 else np.zeros(len(attacks))
        x = np.arange(len(attacks)); w = 0.34
        ax.bar(x - w/2, pre_f1, w, label="Pretrained",
               color=BLUE, alpha=0.55)
        ax.bar(x + w/2, sp_mean, w, yerr=sp_std, capsize=3,
               label="Self-play (3 seeds)",
               color=GREEN, alpha=0.9,
               error_kw={"ecolor": WHITE, "lw": 1.0})
        ax.set_xticks(x); ax.set_xticklabels(attacks, fontsize=8, rotation=20)
        ax.set_title(name, fontsize=10.5, fontweight="bold", color=WHITE)
        ax.set_ylim(0, 1.30)
        if ax is axes[0]:
            ax.set_ylabel("F1")
    # Single shared legend below the panels — never overlaps bars.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, framealpha=0,
               loc="lower center", bbox_to_anchor=(0.5, -0.04),
               ncol=2, columnspacing=2.0)
    fig.suptitle("Multi-seed per-attack F1 across the three networks",
                 fontsize=11.5, fontweight="bold", color=WHITE, y=1.0)
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(OUT / out_name, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved {out_name}")


def plot_classical(eval_path: Path, out_name: str):
    """Classical baselines (Mean / Pseudo-inv / WLS) vs GNN MoE."""
    d = json.load(open(eval_path))
    GNN_MAE = {"Net1": 1.526, "Net3": 1.082, "Modena": 0.435}
    GNN_F1 = {"Net1": 0.650, "Net3": 0.724, "Modena": 0.767}
    networks = [n for n in ["Net1", "Net3", "Modena"] if n in d]
    methods = ["Mean", "PseudoInv", "WLS"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))

    ax = axes[0]
    x = np.arange(len(networks)); w = 0.20
    for i, m in enumerate(methods):
        vals = [d[n]["results"][m]["p_mae"] for n in networks]
        ax.bar(x + (i - 1.5) * w, vals, w, label=m,
               color=[GRAY, DIMGRAY, BLUE][i], alpha=0.85)
    ax.bar(x + 1.5 * w, [GNN_MAE[n] for n in networks], w,
           label="GNN MoE", color=GREEN, alpha=0.95)
    ax.set_xticks(x); ax.set_xticklabels(networks)
    ax.set_yscale("log"); ax.set_ylabel("P MAE (m, log scale)")
    ax.set_title("Reconstruction error", fontsize=10.5,
                 fontweight="bold", color=WHITE)
    ax.legend(fontsize=8.5, framealpha=0, loc="upper right")

    ax = axes[1]
    for i, m in enumerate(methods):
        vals = [d[n]["results"][m]["f1"] for n in networks]
        ax.bar(x + (i - 1.5) * w, vals, w, label=m,
               color=[GRAY, DIMGRAY, BLUE][i], alpha=0.85)
    ax.bar(x + 1.5 * w, [GNN_F1[n] for n in networks], w,
           label="GNN MoE", color=GREEN, alpha=0.95)
    ax.set_xticks(x); ax.set_xticklabels(networks)
    ax.set_ylabel("Anomaly F1"); ax.set_ylim(0, 1.0)
    ax.set_title("Anomaly detection", fontsize=10.5,
                 fontweight="bold", color=WHITE)
    ax.legend(fontsize=8.5, framealpha=0, loc="upper left")

    fig.suptitle("Classical baselines vs GNN Mixture-of-Experts",
                 fontsize=11.5, fontweight="bold", color=WHITE, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / out_name, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved {out_name}")


def plot_robustness(eval_path: Path, out_name: str):
    d = json.load(open(eval_path))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.3))
    colors = {"Pretrained": BLUE, "Self-play single": PURPLE,
              "Self-play MoE": GREEN}
    for ax, key, ylabel in zip(axes, ["f1", "auroc"], ["F1", "AUROC"]):
        for name, series in d.items():
            ax.plot(series["epsilons"], series[key], "o-",
                    label=name, color=colors.get(name, GRAY), lw=2.2,
                    markersize=6)
        ax.set_xlabel("Stealth budget ε (m)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs attack budget",
                     fontsize=10.5, fontweight="bold", color=WHITE)
        ax.legend(fontsize=8.5, framealpha=0, loc="lower right")
    fig.suptitle("Adversarial robustness — three defenders, six budgets",
                 fontsize=11.5, fontweight="bold", color=WHITE, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / out_name, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved {out_name}")


def plot_crossnet(eval_path: Path, out_name: str):
    d = json.load(open(eval_path))
    src = d["source_network"]; tgt = d["targets"]
    networks = list(tgt.keys())
    pre_f1 = [tgt[n]["pretrained"]["f1"] for n in networks]
    sp_f1 = [tgt[n]["selfplay"]["f1"] for n in networks]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(networks)); w = 0.34
    ax.bar(x - w/2, pre_f1, w, label="Pretrained", color=BLUE, alpha=0.6)
    ax.bar(x + w/2, sp_f1, w, label="Self-play", color=GREEN, alpha=0.9)
    for i, (p, s) in enumerate(zip(pre_f1, sp_f1)):
        ax.text(i - w/2, p + 0.01, f"{p:.2f}",
                ha="center", fontsize=9, color=WHITE)
        ax.text(i + w/2, s + 0.01, f"{s:.2f}",
                ha="center", fontsize=9, fontweight="bold", color=WHITE)
    ax.set_xticks(x); ax.set_xticklabels(networks)
    ax.set_ylabel("Anomaly F1"); ax.set_ylim(0, 1.0)
    ax.set_title(f"Cross-network transfer — attacker trained on {src}",
                 fontsize=11, fontweight="bold", color=WHITE, pad=8)
    ax.legend(fontsize=9, framealpha=0, loc="lower right")
    plt.tight_layout()
    fig.savefig(OUT / out_name, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved {out_name}")


if __name__ == "__main__":
    plot_coevolution(
        ROOT / "runs/selfplay/20260505_215710/history.json",
        "selfplay_coevolution_scratch.png",
        "Self-play co-evolution — scratch defender (Modena)",
    )
    plot_coevolution(
        ROOT / "runs/selfplay/20260505_223529/history.json",
        "selfplay_coevolution_finetune.png",
        "Self-play co-evolution — pretrained + fine-tune (Modena)",
    )
    plot_per_attack_compare(
        ROOT / "runs/selfplay/20260505_223529/eval_comparison.json",
        "selfplay_per_attack.png",
    )
    if (ROOT / "runs/selfplay/eval_all.json").exists():
        plot_three_network_compare(
            ROOT / "runs/selfplay/eval_all.json",
            "selfplay_three_networks.png",
        )
    if (ROOT / "runs/selfplay/eval_multiseed.json").exists():
        plot_multiseed(
            ROOT / "runs/selfplay/eval_multiseed.json",
            "selfplay_multiseed.png",
        )
    if (ROOT / "runs/selfplay/eval_heldout.json").exists():
        plot_heldout(
            ROOT / "runs/selfplay/eval_heldout.json",
            "selfplay_heldout.png",
        )
    if (ROOT / "runs/selfplay/eval_classical.json").exists():
        plot_classical(
            ROOT / "runs/selfplay/eval_classical.json",
            "classical_baselines.png",
        )
    if (ROOT / "runs/selfplay/eval_robustness.json").exists():
        plot_robustness(
            ROOT / "runs/selfplay/eval_robustness.json",
            "selfplay_robustness.png",
        )
    if (ROOT / "runs/selfplay/eval_crossnet.json").exists():
        plot_crossnet(
            ROOT / "runs/selfplay/eval_crossnet.json",
            "selfplay_crossnet.png",
        )
    plot_multiseed_compare(
        None,
        ROOT / "runs/selfplay/eval_multiseed_net3.json",
        ROOT / "runs/selfplay/eval_multiseed.json",
        "selfplay_multiseed_compare.png",
    )
