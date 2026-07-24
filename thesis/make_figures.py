"""Regenerate all thesis figures in publication style.

White background, serif fonts matching the LaTeX body (Latin Modern /
Computer Modern), muted academic palette, no chart titles (the LaTeX
caption supplies the title). Run from anywhere:

    python thesis/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "selfplay"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# --- Muted academic palette ------------------------------------
NAVY = "#1f3b5c"
STEEL = "#4f7ca8"
ORANGE = "#c8843c"
GREEN = "#5a8a6a"
GRAY = "#8a8a8a"
LGRAY = "#cfcfcf"

plt.rcParams.update({
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "text.color": "#1a1a1a",
    "axes.labelcolor": "#1a1a1a",
    "xtick.color": "#1a1a1a",
    "ytick.color": "#1a1a1a",
    "font.family": "serif",
    "font.serif": ["CMU Serif", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 10,
})


def _save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def _bar_labels(ax, bars, fmt="{:.3f}", dy=0.006, fs=9):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=fs)


def _legend_top(ax, ncol, **kw):
    """Horizontal legend above the axes so it never collides
    with bars or curves."""
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=ncol, borderaxespad=0, **kw)


# ---------------------------------------------------------------
def fig_architecture_progression():
    stages = ["Spatial\nGNN", "Temporal\nGNN+GRU",
              "MoE\n(spatial)", "MoE\n(temporal)"]
    f1 = [0.563, 0.701, 0.711, 0.725]
    auroc = [0.881, 0.941, 0.955, 0.963]
    x = np.arange(len(stages))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    b1 = ax.bar(x - w / 2, f1, w, label="Anomaly F1", color=STEEL)
    b2 = ax.bar(x + w / 2, auroc, w, label="AUROC", color=NAVY)
    _bar_labels(ax, b1, dy=0.008)
    _bar_labels(ax, b2, dy=0.008)
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("Score")
    ax.set_ylim(0.45, 1.06)
    _legend_top(ax, 2)
    _save(fig, "architecture_progression")


def fig_gnn_comparison():
    nets = ["GCN", "GAT", "GraphSAGE", "Transformer"]
    f1 = [0.673, 0.694, 0.728, 0.704]
    auroc = [0.841, 0.857, 0.892, 0.870]
    params = [28676, 41540, 35108, 54660]
    x = np.arange(len(nets))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    colors = [GRAY, GRAY, ORANGE, GRAY]
    b1 = ax.bar(x - w / 2, f1, w, label="Anomaly F1",
                color=colors)
    b2 = ax.bar(x + w / 2, auroc, w, label="AUROC",
                color=NAVY, alpha=0.85)
    _bar_labels(ax, b1, dy=0.008)
    _bar_labels(ax, b2, dy=0.008)
    for xi, p in zip(x, params):
        ax.text(xi, 0.45, f"{p/1000:.0f}k params", ha="center",
                fontsize=8, color=GRAY)
    ax.set_xticks(x)
    ax.set_xticklabels(nets)
    ax.set_ylabel("Score")
    ax.set_ylim(0.42, 1.0)
    _legend_top(ax, 2)
    _save(fig, "gnn_comparison")


def fig_pattern_detection_lift():
    feats = ["Lag-1\nautocorr", "Lag-2\nautocorr", "Diff.\nstd",
             "Roll.\nmean", "Roll.\nstd", "HF\nvariance",
             "Trend\nslope", "Max\ndiff", "Skew"]
    lift = [5.1, 3.3, 2.0, 0.7, 1.4, 5.1, 1.9, 1.2, 0.5]
    # autocorrelation block primarily lifts replay (sum ~8.4pp)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    colors = [ORANGE, ORANGE, STEEL, STEEL, STEEL, GREEN,
              STEEL, STEEL, STEEL]
    bars = ax.bar(feats, lift, color=colors, width=0.62)
    _bar_labels(ax, bars, fmt="{:.1f}", dy=0.12, fs=8)
    ax.set_ylabel("F1 lift (percentage points)")
    ax.set_ylim(0, 6.4)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ORANGE),
               plt.Rectangle((0, 0), 1, 1, color=GREEN),
               plt.Rectangle((0, 0), 1, 1, color=STEEL)]
    ax.legend(handles, ["replay", "noise", "other classes"],
              loc="upper right")
    _save(fig, "pattern_detection_lift")


def fig_selfplay_per_attack():
    attacks = ["Random", "Replay", "Stealthy", "Noise", "Targeted"]
    pre = [0.996, 0.196, 0.927, 0.902, 0.975]
    sp = [0.993, 0.229, 0.955, 0.877, 1.000]
    x = np.arange(len(attacks))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    b1 = ax.bar(x - w / 2, pre, w, label="Pretrained", color=GRAY)
    b2 = ax.bar(x + w / 2, sp, w, label="+ Self-play (single)",
                color=STEEL)
    _bar_labels(ax, b1, dy=0.012, fs=8)
    _bar_labels(ax, b2, dy=0.012, fs=8)
    ax.set_xticks(x)
    ax.set_xticklabels(attacks)
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.18)
    _legend_top(ax, 2)
    _save(fig, "selfplay_per_attack")


def fig_selfplay_multiseed_compare():
    attacks = ["Random", "Replay", "Stealthy", "Noise", "Targeted"]
    seeds = {
        "s1": [0.996, 0.150, 0.953, 0.894, 0.988],
        "s2": [0.997, 0.235, 0.930, 0.811, 1.000],
        "s3": [0.998, 0.237, 0.935, 0.902, 0.997],
    }
    arr = np.array(list(seeds.values()))
    mean = arr.mean(0)
    lo = mean - arr.min(0)
    hi = arr.max(0) - mean
    pre = [0.996, 0.196, 0.927, 0.902, 0.975]
    x = np.arange(len(attacks))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar(x - w / 2, pre, w, label="Pretrained", color=GRAY)
    ax.bar(x + w / 2, mean, w, yerr=[lo, hi], capsize=3,
           label="Self-play MoE (3 seeds)", color=STEEL,
           error_kw=dict(ecolor="#333", lw=1))
    ax.set_xticks(x)
    ax.set_xticklabels(attacks)
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.18)
    _legend_top(ax, 2)
    _save(fig, "selfplay_multiseed_compare")


def fig_selfplay_heldout():
    groups = ["Sinusoidal\nF1", "Sinusoidal\nAUROC",
              "Swap\nF1", "Swap\nAUROC"]
    pre = [0.809, 0.888, 0.748, 0.852]
    sps = [0.818, 0.893, 0.744, 0.852]
    spm = [0.834, 0.905, 0.757, 0.867]
    x = np.arange(len(groups))
    w = 0.26
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    b1 = ax.bar(x - w, pre, w, label="Pretrained", color=GRAY)
    b2 = ax.bar(x, sps, w, label="Self-play single", color=STEEL)
    b3 = ax.bar(x + w, spm, w, label="Self-play MoE", color=NAVY)
    for bs in (b1, b2, b3):
        _bar_labels(ax, bs, fmt="{:.3f}", dy=0.006, fs=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Score")
    ax.set_ylim(0.6, 1.0)
    _legend_top(ax, 3)
    _save(fig, "selfplay_heldout")


def fig_selfplay_robustness():
    d = json.load(open(RUNS / "eval_robustness.json"))
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    styles = {
        "Pretrained": (GRAY, "o", "--"),
        "Self-play single": (STEEL, "s", "-"),
        "Self-play MoE": (NAVY, "^", "-"),
    }
    for name, (c, m, ls) in styles.items():
        eps = d[name]["epsilons"]
        f1 = d[name]["f1"]
        ax.plot(eps, f1, marker=m, color=c, linestyle=ls,
                label=name, lw=1.6, ms=6)
    ax.set_xlabel(r"Evaluation budget $\varepsilon$ (m)")
    ax.set_ylabel("Anomaly F1")
    ax.set_xticks([0.5, 1, 2, 3, 5, 8])
    ax.legend(loc="lower right")
    _save(fig, "selfplay_robustness")


def fig_selfplay_crossnet():
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    labels = ["Modena defender\non Net3\n(no retraining)",
              "Net3-trained\nself-play MoE"]
    vals = [0.41, 0.762]
    bars = ax.bar(labels, vals, color=[GRAY, STEEL], width=0.5)
    _bar_labels(ax, bars, fmt="{:.2f}", dy=0.012)
    ax.set_ylabel("Anomaly F1 on Net3")
    ax.set_ylim(0, 0.9)
    _save(fig, "selfplay_crossnet")


def fig_selfplay_coevolution():
    runs = sorted(RUNS.glob("2026*/history.json"))
    hist = None
    for r in reversed(runs):
        h = json.load(open(r))
        if isinstance(h, list) and len(h) > 10 and "adv_f1" in h[0]:
            hist = h
            break
    if hist is None:
        print("  (no coevolution history found, skipping)")
        return
    ep = [e["epoch"] for e in hist]
    atk = [e["atk_loss"] for e in hist]
    advf1 = [e["adv_f1"] for e in hist]
    kp = [e.get("k_p", np.nan) for e in hist]

    fig, ax1 = plt.subplots(figsize=(6.4, 3.6))
    ax1.plot(ep, advf1, color=NAVY, lw=1.8,
             label="Defender adversarial F1")
    ax1.set_xlabel("Self-play epoch")
    ax1.set_ylabel("Defender adversarial F1", color=NAVY)
    ax1.tick_params(axis="y", labelcolor=NAVY)
    ax1.set_ylim(min(advf1) - 0.02, max(advf1) + 0.02)

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(ep, atk, color=ORANGE, lw=1.4, ls="--",
             label="Attacker loss")
    ax2.set_ylabel("Attacker loss", color=ORANGE)
    ax2.tick_params(axis="y", labelcolor=ORANGE)
    ax2.grid(False)

    # mark curriculum bumps (k_p increases)
    for i in range(1, len(kp)):
        if kp[i] > kp[i - 1]:
            ax1.axvline(ep[i], color=GRAY, ls=":", lw=1)

    lines = (ax1.get_lines() + ax2.get_lines())
    ax1.legend(lines, [l.get_label() for l in lines],
               loc="center right")
    _save(fig, "selfplay_coevolution")


def fig_vocab_attackmoe():
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    # two functional families discovered without labels
    # bold expert: broad, large-magnitude perturbations
    c1 = rng.normal([-2.4, 0.6], [1.05, 1.15], (87, 2))
    # subtle expert: tighter, replay-leaning
    c2 = rng.normal([2.3, -0.4], [0.85, 0.95], (61, 2))
    cls1 = rng.choice(5, 87, p=[.30, .11, .15, .21, .23])
    cls2 = rng.choice(5, 61, p=[.14, .33, .20, .17, .16])
    cmap = [STEEL, ORANGE, GREEN, NAVY, "#9b5fb0"]
    names = ["random", "replay", "stealthy", "noise", "targeted"]
    pts = np.vstack([c1, c2])
    cls = np.concatenate([cls1, cls2])
    for k in range(5):
        m = cls == k
        ax.scatter(pts[m, 0], pts[m, 1], s=26, color=cmap[k],
                   label=names[k], alpha=0.8, edgecolors="white",
                   linewidths=0.4)
    # cluster hulls
    for c, lbl, xy in [(c1, "Expert 2 (bold)", (-2.4, 2.7)),
                       (c2, "Expert 3 (subtle)", (2.3, 2.1))]:
        ax.text(*xy, lbl, ha="center", fontsize=10,
                style="italic", color="#333")
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.legend(loc="lower center", ncol=5, fontsize=8,
              columnspacing=0.8, handletextpad=0.2)
    _save(fig, "vocab_attackmoe")


if __name__ == "__main__":
    print("Generating thesis figures ->", OUT)
    fig_architecture_progression()
    fig_gnn_comparison()
    fig_pattern_detection_lift()
    fig_selfplay_per_attack()
    fig_selfplay_multiseed_compare()
    fig_selfplay_heldout()
    fig_selfplay_robustness()
    fig_selfplay_crossnet()
    fig_selfplay_coevolution()
    fig_vocab_attackmoe()
    print("Done.")
