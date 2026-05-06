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
    ax.set_ylabel("F1"); ax.set_ylim(0, 1.1)
    ax.set_title("Per-attack F1 — Self-play vs Pretrained (Modena)",
                 fontsize=11, fontweight="bold", color=WHITE, pad=8)
    ax.legend(fontsize=9, framealpha=0, loc="lower left", ncol=2)
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
    ax.set_ylabel("F1"); ax.set_ylim(0, 1.15)
    ax.set_title("Per-attack F1 — Self-play vs Pretrained (Modena, 3 seeds)",
                 fontsize=11, fontweight="bold", color=WHITE, pad=8)
    ax.legend(fontsize=9, framealpha=0, loc="lower left", ncol=2)
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
        ax.set_ylim(0.6, 1.0)
        ax.set_title(f"Novel attack: {kind}",
                     fontsize=10.5, fontweight="bold", color=WHITE, pad=6)
        ax.legend(fontsize=8, framealpha=0, loc="lower right", ncol=2)
    fig.suptitle("Held-out generalization — novel attacks never seen in training",
                 fontsize=11.5, fontweight="bold", color=WHITE, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
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
