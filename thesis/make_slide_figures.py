"""Generate presentation-quality figures for the supervisor slides.

Distinct from make_figures.py (which targets the LaTeX thesis in a serif,
publication style). These are tuned for projection: white background,
sans-serif, large type, the validated categorical palette, 16:9-friendly
aspect ratios, 200 dpi.

    python3 thesis/make_slide_figures.py

Writes to thesis/figures/slides/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "figures" / "slides"
OUT.mkdir(parents=True, exist_ok=True)

# --- Validated categorical palette (same as the dashboard) ------
BLUE = "#2563eb"
ORANGE = "#ea580c"
GREEN = "#059669"
PURPLE = "#7c3aed"
RED = "#dc2626"
CYAN = "#0891b2"
YELLOW = "#ca8a04"
INK = "#1f2937"
MUTED = "#6b7280"
FAINT = "#e5e7eb"

plt.rcParams.update({
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#c9ced6",
    "axes.linewidth": 1.0,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": "#54607a",
    "ytick.color": "#54607a",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 13,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#eef0f4",
    "grid.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 12,
})

ATTACKS = ["random", "targeted", "stealthy", "noise", "replay"]
ALABEL = {"random": "Random", "targeted": "Targeted", "stealthy": "Stealthy",
          "noise": "Noise", "replay": "Replay"}
ACOLOR = {"random": BLUE, "targeted": GREEN, "stealthy": ORANGE,
          "noise": CYAN, "replay": PURPLE}
DCOLOR = {"water": BLUE, "power": ORANGE, "traffic": GREEN}


def _save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"  wrote slides/{name}.png")


def _box(ax, x, y, w, h, title, sub=None, color=BLUE, fs=12, alpha=0.10):
    """Rounded box with a bold title and an optional muted sub-line.

    Vertical offsets are in data units; one unit ~= 1 inch ~= 72 pt, so a
    12 pt line needs ~0.17 units of separation to clear its neighbour.
    """
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.035",
        linewidth=1.8, edgecolor=color, facecolor=color, alpha=alpha,
        mutation_aspect=1))
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.035",
        linewidth=1.8, edgecolor=color, facecolor="none", mutation_aspect=1))
    if sub:
        ax.text(x + w / 2, y + h / 2 + 0.17, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h / 2 - 0.17, sub, ha="center", va="center",
                fontsize=fs - 2.5, color=MUTED)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=INK)


def _arrow(ax, x0, y0, x1, y1, color="#8b93a3", lw=1.8, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle=style, mutation_scale=14,
        linewidth=lw, color=color, shrinkA=0, shrinkB=0))


# ===============================================================
# 1. THE architecture figure — temporal Mixture-of-Experts GNN
# ===============================================================
def fig_architecture():
    fig, ax = plt.subplots(figsize=(13.6, 6.35))
    ax.set_xlim(0, 13.6); ax.set_ylim(0, 6.35)
    ax.axis("off")

    # --- Input: sliding window of graph snapshots ---
    ax.text(1.15, 6.15, "Input", fontsize=12.5, fontweight="bold", color=INK,
            ha="center")
    for i, dx in enumerate([0.0, 0.16, 0.32]):
        ax.add_patch(FancyBboxPatch(
            (0.35 + dx, 3.55 + dx * 0.9), 1.35, 1.75,
            boxstyle="round,pad=0.01,rounding_size=0.06",
            linewidth=1.6, edgecolor=BLUE,
            facecolor="white" if i < 2 else "#eaf1ff", zorder=i))
    ax.text(1.35, 4.85, "graph\nsnapshots", ha="center", va="center",
            fontsize=11, fontweight="bold", color=INK, zorder=5)
    ax.text(1.35, 4.15, "6-step\nwindow", ha="center", va="center",
            fontsize=9.5, color=MUTED, zorder=5)
    ax.text(1.15, 3.15, "pressure · flow · mask\n(sparse, 50% missing)",
            ha="center", va="top", fontsize=9.5, color=MUTED)

    # --- Router branch (top) ---
    _box(ax, 3.15, 5.05, 2.5, 1.05, "Router (small GNN)",
         "classifies attack class", color=PURPLE, fs=11.5)
    _box(ax, 6.35, 5.05, 2.5, 1.05, "Confidence gate",
         r"$T = 1 + \alpha(1-\max p)$", color=PURPLE, fs=11.5)
    ax.text(7.05, 4.72, r"unsure $\rightarrow$ spread the mix", ha="left",
            fontsize=9, color=MUTED, style="italic")

    # --- Experts (middle stack) ---
    ax.text(4.4, 4.35, "6 attack-specialised experts", fontsize=11.5,
            fontweight="bold", color=INK, ha="center")
    ecolors = [BLUE, GREEN, ORANGE, CYAN, RED, YELLOW]
    enames = ["clean", "random", "replay", "stealthy", "noise", "targeted"]
    for i, (c, nm) in enumerate(zip(ecolors, enames)):
        y = 3.72 - i * 0.60
        _box(ax, 3.15, y, 2.5, 0.48, "", None, color=c, fs=10, alpha=0.13)
        ax.text(3.55, y + 0.24, f"Expert {i+1}", ha="left", va="center",
                fontsize=10, fontweight="bold", color=INK)
        ax.text(5.45, y + 0.24, nm, ha="right", va="center",
                fontsize=9.5, color=MUTED)
    ax.text(4.4, 0.02,
            r"each:  GraphSAGE $\rightarrow$ GRU $\rightarrow$ 4 heads"
            "     (hidden 64)",
            ha="center", fontsize=9.5, color=MUTED)

    # --- Mixing ---
    ax.add_patch(plt.Circle((6.55, 2.35), 0.30, facecolor="white",
                            edgecolor=INK, linewidth=1.8, zorder=4))
    ax.text(6.55, 2.35, "×", ha="center", va="center", fontsize=20,
            color=INK, zorder=5)
    ax.text(6.55, 1.85, "weighted\nmix", ha="center", va="top", fontsize=9.5,
            color=MUTED)

    # --- Outputs ---
    _box(ax, 8.0, 3.35, 2.35, 0.85, "Reconstruction", "pressure · flow",
         color=BLUE, fs=11.5)
    _box(ax, 8.0, 2.05, 2.35, 0.85, "Anomaly flags", "per-sensor",
         color=RED, fs=11.5)

    # --- Losses panel ---
    _box(ax, 10.95, 1.95, 2.45, 2.25, "", None, color=GREEN, fs=11, alpha=0.07)
    ax.text(12.17, 3.95, "Training signal", ha="center", fontsize=11.5,
            fontweight="bold", color=INK)
    loss_lines = [
        ("reconstruction", "MSE on observed"),
        ("anomaly", "BCE per sensor"),
        ("router CE", "+ balance entropy"),
        ("physics", r"$\|B\mathbf{q}\|^2$ mass cons."),
        ("direct expert", "each expert on its class"),
    ]
    for i, (a, b) in enumerate(loss_lines):
        y = 3.62 - i * 0.34
        ax.text(11.12, y, "•", fontsize=11, color=GREEN, va="center")
        ax.text(11.30, y, a, fontsize=10, fontweight="bold", color=INK,
                va="center")
        ax.text(11.30, y - 0.145, b, fontsize=8.5, color=MUTED, va="center")

    # --- Arrows ---
    _arrow(ax, 1.95, 5.05, 3.15, 5.57)          # input → router
    _arrow(ax, 5.65, 5.57, 6.35, 5.57)          # router → gate
    _arrow(ax, 1.95, 4.45, 3.15, 3.55)          # input → experts
    # Mix weights drop down the left of the gate so they clear the caption.
    _arrow(ax, 6.80, 5.05, 6.58, 2.68)          # gate → mix (weights)
    _arrow(ax, 5.65, 2.35, 6.25, 2.35)          # experts → mix
    _arrow(ax, 6.85, 2.50, 8.00, 3.60)          # mix → recon
    _arrow(ax, 6.85, 2.28, 8.00, 2.45)          # mix → anomaly
    _arrow(ax, 10.35, 3.05, 10.95, 3.05)        # outputs → losses

    # No in-figure title: the slide's frame title already names it.
    _save(fig, "architecture_moe")


# ===============================================================
# 2. Pipeline overview
# ===============================================================
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(13.6, 2.5))
    ax.set_xlim(0, 13.6); ax.set_ylim(0, 2.5)
    ax.axis("off")
    steps = [
        ("Network", "EPANET / pandapower\n/ road graph", BLUE),
        ("Simulation", "time-series per\nnode & edge", CYAN),
        ("Corruption", "5 attacks +\nnoise + missing", RED),
        ("Detector", "temporal MoE\nGNN", PURPLE),
        ("Evaluation", "per-attack F1\nmulti-seed", GREEN),
    ]
    w, gap = 2.24, 0.50
    for i, (t, s, c) in enumerate(steps):
        x = 0.35 + i * (w + gap)
        # Empty box, then place title and the two-line sub with enough
        # vertical separation that they cannot collide.
        _box(ax, x, 0.55, w, 1.35, "", None, color=c, fs=13)
        ax.text(x + w / 2, 1.48, t, ha="center", va="center", fontsize=13,
                fontweight="bold", color=INK)
        ax.text(x + w / 2, 0.98, s, ha="center", va="center", fontsize=9.5,
                color=MUTED)
        if i < len(steps) - 1:
            _arrow(ax, x + w + 0.06, 1.22, x + w + gap - 0.06, 1.22)
    _save(fig, "pipeline")


# ===============================================================
# 3. Part 1 — per-attack F1 with seed error bars
# ===============================================================
def fig_part1_perattack(cd):
    w = cd["water_modena_10seed"]
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    means = [w[a][0] for a in ATTACKS]
    stds = [w[a][1] for a in ATTACKS]
    cols = [ACOLOR[a] for a in ATTACKS]
    bars = ax.bar([ALABEL[a] for a in ATTACKS], means, color=cols, width=0.62,
                  yerr=stds, capsize=4,
                  error_kw=dict(ecolor="#7b8494", lw=1.3))
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.03, f"{m:.3f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color=INK)
    ax.set_ylim(0, 1.14)
    ax.set_ylabel("Detection F1")
    ax.axhline(1.0, color=FAINT, lw=1)
    # Point at the left shoulder of the replay bar so the arrow head stays
    # clear of the centred value label.
    ax.annotate("information-theoretic\nceiling", xy=(3.72, 0.035),
                xytext=(3.05, 0.42), fontsize=10.5, color=PURPLE,
                ha="center",
                arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.5,
                                shrinkA=2, shrinkB=3))
    _save(fig, "part1_per_attack")


# ===============================================================
# 4. Router diagnostic — Modena healthy vs Net3 mis-routing
# ===============================================================
def fig_router(diag):
    names = ["clean", "random", "replay", "stealthy", "noise", "targeted"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    for ax, net in zip(axes, ["Modena", "Net3"]):
        conf = np.array(diag[net]["confusion"], dtype=float)[1:, 1:]
        rows = conf.sum(1, keepdims=True); rows[rows == 0] = 1
        norm = conf / rows
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        lbl = names[1:]
        ax.set_xticks(range(5)); ax.set_xticklabels(lbl, rotation=30, ha="right")
        ax.set_yticks(range(5)); ax.set_yticklabels(lbl)
        for i in range(5):
            for j in range(5):
                v = int(conf[i, j])
                if v:
                    ax.text(j, i, v, ha="center", va="center", fontsize=11,
                            color="white" if norm[i, j] > 0.55 else INK,
                            fontweight="bold" if i == j else "normal")
        acc = diag[net]["overall_acc"]
        ax.set_title(f"{net} — router accuracy {acc:.0%}", fontsize=13,
                     fontweight="bold", pad=10)
        ax.set_xlabel("routed to expert"); ax.grid(False)
        if net == "Modena":
            ax.set_ylabel("true attack class")
        else:
            # Highlight the stealthy(row 2) -> replay(col 1) cell. Cell (i,j)
            # spans (j-0.5, i-0.5) to (j+0.5, i+0.5).
            ax.add_patch(Rectangle((0.5, 1.5), 1, 1, fill=False,
                                   edgecolor=RED, lw=3, zorder=5))
            ax.annotate(
                "77% of stealthy windows" "\n" r"$\rightarrow$ the replay expert",
                xy=(1.52, 2.0), xytext=(2.75, 0.95),
                fontsize=10.5, color=RED, fontweight="bold",
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=2,
                                connectionstyle="arc3,rad=-0.25",
                                shrinkA=2, shrinkB=2))
    _save(fig, "router_diagnostic")


# ===============================================================
# 5. Replay ceiling — the loss-weight Pareto front
# ===============================================================
def fig_replay_pareto(rw):
    ws = sorted(rw.keys(), key=float)
    x = [float(k) for k in ws]
    overall = [rw[k]["pressure_f1"]["mean"] for k in ws]
    o_err = [rw[k]["pressure_f1"]["std"] for k in ws]
    replay = [rw[k]["per_attack"]["replay"]["mean"] for k in ws]
    r_err = [rw[k]["per_attack"]["replay"]["std"] for k in ws]
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    ax.errorbar(x, overall, yerr=o_err, marker="o", ms=9, lw=2.5, color=BLUE,
                capsize=4, label="Overall F1")
    ax.errorbar(x, replay, yerr=r_err, marker="s", ms=9, lw=2.5, color=PURPLE,
                capsize=4, label="Replay F1")
    ax.set_xlabel("Replay loss weight")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right")
    ax.annotate("pushing replay costs\noverall performance",
                xy=(2.5, 0.715), xytext=(3.6, 0.50), fontsize=10.5,
                color=MUTED, ha="left",
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4))
    _save(fig, "replay_pareto")


# ===============================================================
# 6. Cross-domain — overall F1 per domain
# ===============================================================
def fig_crossdomain_f1(cd):
    keys = [("water", "water_modena_10seed", "Water\nModena · 272 buses"),
            ("power", "power_ieee118_5seed", "Power\nIEEE 118-bus"),
            ("traffic", "traffic_200_5seed", "Traffic\n200 sensors")]
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    f1 = [cd[k]["F1"][0] for _, k, _ in keys]
    err = [cd[k]["F1"][1] for _, k, _ in keys]
    cols = [DCOLOR[d] for d, _, _ in keys]
    bars = ax.bar([lab for _, _, lab in keys], f1, color=cols, width=0.55,
                  yerr=err, capsize=5, error_kw=dict(ecolor="#7b8494", lw=1.4))
    for i, (m, e) in enumerate(zip(f1, err)):
        ax.text(i, m + e + 0.028, f"{m:.3f}", ha="center", va="bottom",
                fontsize=15, fontweight="bold", color=INK)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Detection F1")
    # Widen the x range so the band caption has clear space to the right of
    # the last bar instead of overlapping it.
    ax.set_xlim(-0.6, 3.35)
    ax.axhspan(0.84, 0.88, color=GREEN, alpha=0.07)
    ax.text(2.45, 0.86, "same band\nacross domains", fontsize=10.5,
            color=GREEN, va="center", ha="left", fontweight="bold")
    _save(fig, "crossdomain_f1")


# ===============================================================
# 7. THE insight — replay F1 vs signal speed
# ===============================================================
def fig_replay_vs_signal(cd):
    order = [("water", "water_modena_10seed", "Water\npressure"),
             ("power", "power_ieee118_5seed", "Power\nvoltage"),
             ("traffic", "traffic_200_5seed", "Traffic\nspeed")]
    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    vals = [cd[k]["replay"][0] for _, k, _ in order]
    errs = [cd[k]["replay"][1] for _, k, _ in order]
    cols = [DCOLOR[d] for d, _, _ in order]
    bars = ax.bar([lab for _, _, lab in order], vals, color=cols, width=0.52,
                  yerr=errs, capsize=5, error_kw=dict(ecolor="#7b8494", lw=1.4))
    # Stack the value label above the error-bar cap, and the note above the
    # value, so nothing can collide regardless of bar height or spread.
    notes = ["hides inside\nthe noise band", "at the edge of\ndetectability",
             "stale value\nstands out"]
    for i, (m, e, n) in enumerate(zip(vals, errs, notes)):
        top = m + e
        ax.text(i, top + 0.045, f"{m:.3f}", ha="center", va="bottom",
                fontsize=16, fontweight="bold", color=INK)
        ax.text(i, top + 0.155, n, ha="center", va="bottom", fontsize=10.5,
                color=MUTED)
    ax.set_ylim(0, 1.30)
    ax.set_ylabel("Replay-attack F1")
    ax.set_xlabel("signal changes slowly   $\\longrightarrow$   signal changes fast",
                  labelpad=12)
    _save(fig, "replay_vs_signal")


# ===============================================================
# 8. Difficulty profile heatmap (domain x attack)
# ===============================================================
def fig_difficulty_heatmap(cd):
    doms = [("Water", "water_modena_10seed"), ("Power", "power_ieee118_5seed"),
            ("Traffic", "traffic_200_5seed")]
    z = np.array([[cd[k][a][0] for a in ATTACKS] for _, k in doms])
    fig, ax = plt.subplots(figsize=(10.4, 3.9))
    im = ax.imshow(z, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ATTACKS)))
    ax.set_xticklabels([ALABEL[a] for a in ATTACKS], fontsize=13)
    ax.set_yticks(range(len(doms)))
    ax.set_yticklabels([d for d, _ in doms], fontsize=13, fontweight="bold")
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, f"{z[i, j]:.2f}", ha="center", va="center",
                    fontsize=14,
                    color="white" if z[i, j] > 0.55 else INK,
                    fontweight="bold")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("F1", fontsize=11)
    cb.outline.set_visible(False)
    # highlight the inverting column
    ax.add_patch(Rectangle((3.5, -0.5), 1, 3, fill=False, edgecolor=PURPLE,
                           lw=3))
    _save(fig, "difficulty_heatmap")


# ===============================================================
# 9. Cascade architecture — router ranks -> expert -> feedback -> re-route
# ===============================================================
def fig_cascade_architecture():
    fig, ax = plt.subplots(figsize=(13.6, 6.4))
    ax.set_xlim(0, 13.6); ax.set_ylim(0, 6.4)
    ax.axis("off")

    # Input snapshots
    for i, dx in enumerate([0.0, 0.14, 0.28]):
        ax.add_patch(FancyBboxPatch(
            (0.3 + dx, 3.05 + dx * 0.9), 1.25, 1.6,
            boxstyle="round,pad=0.01,rounding_size=0.06",
            linewidth=1.6, edgecolor=BLUE,
            facecolor="white" if i < 2 else "#eaf1ff", zorder=i))
    ax.text(1.2, 4.2, "graph\nsnapshots", ha="center", va="center",
            fontsize=11, fontweight="bold", color=INK, zorder=5)
    ax.text(1.2, 3.5, "6-step window", ha="center", va="center",
            fontsize=8.5, color=MUTED, zorder=5)

    # Boxes on a common centre-line y=3.85.
    cy = 3.85
    # Router (ranks experts)
    _box(ax, 2.65, cy - 0.75, 2.5, 1.5, "Router",
         "very fast classifier\nranks the 6 experts", color=PURPLE, fs=12.5)
    # Selected expert
    _box(ax, 5.95, cy - 0.6, 2.2, 1.2, "Run expert k", "k-th most likely",
         color=BLUE, fs=12)
    # Feedback (single title, spaced sub-lines, taller box)
    ax.add_patch(FancyBboxPatch(
        (8.95, cy - 0.95), 2.75, 1.9,
        boxstyle="round,pad=0.012,rounding_size=0.03",
        linewidth=1.8, edgecolor=GREEN, facecolor=GREEN, alpha=0.09))
    ax.add_patch(FancyBboxPatch(
        (8.95, cy - 0.95), 2.75, 1.9,
        boxstyle="round,pad=0.012,rounding_size=0.03",
        linewidth=1.8, edgecolor=GREEN, facecolor="none"))
    ax.text(10.33, cy + 0.62, "Feedback check", ha="center", fontsize=12.5,
            fontweight="bold", color=INK)
    ax.text(10.33, cy + 0.28, "(label-free)", ha="center", fontsize=8.5,
            color=MUTED, style="italic")
    ax.text(10.33, cy - 0.06, "clean-sensor", ha="center", fontsize=9, color=MUTED)
    ax.text(10.33, cy - 0.30, "agreement", ha="center", fontsize=9, color=MUTED)
    ax.text(10.33, cy - 0.64, r"+ physics $\|B\mathbf{q}\|^2$", ha="center",
            fontsize=9, color=MUTED)
    # Accept -> output
    _box(ax, 12.15, cy - 0.5, 1.25, 1.0, "Accept", "use this\nexpert",
         color=BLUE, fs=11)

    # Forward arrows on the centre line.
    _arrow(ax, 1.72, cy, 2.65, cy)
    _arrow(ax, 5.15, cy, 5.95, cy)
    _arrow(ax, 8.15, cy, 8.95, cy)
    _arrow(ax, 11.70, cy, 12.15, cy, color=GREEN)
    ax.text(11.92, cy + 0.28, "pass", ha="center", fontsize=8.5, color=GREEN,
            fontweight="bold")

    # Negative-feedback loop: from the feedback box bottom, arc well below
    # all boxes, back up into the router. Clears the middle boxes entirely.
    ax.add_patch(FancyArrowPatch(
        (10.33, cy - 0.95), (3.9, cy - 0.78),
        connectionstyle="arc3,rad=-0.42",
        arrowstyle="-|>", mutation_scale=16, linewidth=2.2, color=ORANGE))
    ax.text(7.1, 0.72, r"negative feedback  $\rightarrow$  next most likely expert",
            ha="center", fontsize=11.5, color=ORANGE, fontweight="bold")
    ax.text(7.1, 0.36, "loop until an expert is accepted (or the ranking is exhausted)",
            ha="center", fontsize=9, color=MUTED, style="italic")

    ax.text(6.8, 6.05,
            r"Cascade routing  (router $\rightarrow$ expert $\rightarrow$ feedback $\rightarrow$ re-route)",
            ha="center", fontsize=15, fontweight="bold", color=INK)
    ax.text(6.8, 5.62,
            "interpretable: every decision shows which expert ran and why it was kept or rejected",
            ha="center", fontsize=10, color=MUTED, style="italic")
    _save(fig, "cascade_architecture")


# ===============================================================
# 10. Normalisation result — the replay rescue
# ===============================================================
def fig_normalisation(norm):
    by = {r["config"]: r for r in norm}
    order = ["Global (per-signal)", "Per-node"]
    cols = [MUTED, PURPLE]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
    panels = [("overall_f1", "Overall F1", 1.0),
              ("replay_auroc", "Replay AUROC\n(ranking quality)", 1.0),
              ("replay_f1", "Replay F1\n(at calibrated threshold)", 0.2)]
    for ax, (key, title, ymax) in zip(axes, panels):
        vals = [by[o][key] for o in order]
        bars = ax.bar(range(2), vals, color=cols, width=0.6)
        for i, v in enumerate(vals):
            ax.text(i, v + ymax * 0.02, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=14, fontweight="bold", color=INK)
        ax.set_xticks(range(2)); ax.set_xticklabels(order, fontsize=11)
        ax.set_title(title, fontsize=12.5, fontweight="bold", color=INK)
        ax.set_ylim(0, ymax * 1.2)
        if key == "replay_auroc":
            ax.axhline(0.5, color=RED, lw=1.2, ls="--")
            ax.text(1.4, 0.52, "chance", fontsize=8.5, color=RED, ha="right")
    _save(fig, "normalisation")


# ===============================================================
# 11. Routing schemes — soft vs cascade vs hard ceiling
# ===============================================================
def fig_routing_schemes(casc):
    import statistics as _st
    order = [("soft", "Soft mixture", BLUE),
             ("router_top1", "Router top-1", MUTED),
             ("cascade", "Cascade\n(feedback)", ORANGE),
             ("oracle", "Oracle\n(hard ceiling)", GREEN)]
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    for i, (k, lab, c) in enumerate(order):
        v = [r[k]["f1"] for r in casc]
        m = _st.mean(v); s = _st.stdev(v) if len(v) > 1 else 0.0
        ax.bar(i, m, color=c, width=0.6, yerr=s, capsize=5,
               error_kw=dict(ecolor="#7b8494", lw=1.4))
        ax.text(i, m + s + 0.02, f"{m:.3f}", ha="center", va="bottom",
                fontsize=14, fontweight="bold", color=INK)
    ax.set_xticks(range(4)); ax.set_xticklabels([o[1] for o in order], fontsize=11)
    ax.set_ylabel("Detection F1"); ax.set_ylim(0, 1.0)
    _save(fig, "routing_schemes")


def main():
    cd = json.load(open(ROOT / "runs" / "temporal_moe" / "crossdomain_summary.json"))
    rw = json.load(open(ROOT / "runs" / "temporal_moe" / "rw_sweep_summary.json"))
    diag = json.load(open(ROOT / "runs" / "selfplay" / "router_diagnostic.json"))
    norm = json.load(open(ROOT / "runs" / "temporal_moe" / "norm_compare.json"))
    casc = json.load(open(ROOT / "runs" / "temporal_moe" / "cascade_eval.json"))
    print("Generating slide figures ...")
    fig_cascade_architecture()
    fig_normalisation(norm)
    fig_routing_schemes(casc)
    fig_architecture()
    fig_pipeline()
    fig_part1_perattack(cd)
    fig_router(diag)
    fig_replay_pareto(rw)
    fig_crossdomain_f1(cd)
    fig_replay_vs_signal(cd)
    fig_difficulty_heatmap(cd)
    print(f"Done -> {OUT}")


if __name__ == "__main__":
    main()
