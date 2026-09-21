"""Create the conceptual overview figure used in the thesis introduction."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


V2 = Path(__file__).resolve().parents[1]
OUT = V2 / "outputs" / "figures"


def box(ax, x, y, width, height, title, lines, facecolor, edgecolor):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.5,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height - 0.095,
        title,
        ha="center",
        va="top",
        fontsize=11.2,
        fontweight="bold",
        color="#1f2933",
    )
    ax.text(
        x + 0.035,
        y + height - 0.205,
        "\n".join(f"• {line}" for line in lines),
        ha="left",
        va="top",
        fontsize=8.5,
        linespacing=1.35,
        color="#25313c",
    )


def arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.4,
            color="#64748b",
            connectionstyle="arc3,rad=0",
        )
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.2, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    width = 0.215
    height = 0.48
    y = 0.43
    xs = [0.01, 0.26, 0.51, 0.76]
    boxes = [
        (
            "Benchmark",
            ["coherent 8–14-step\n  attack episodes", "standardised attack\n  displacement", "Modena and L-Town\n  × 3 data seeds"],
            "#e8f1fb",
            "#4c78a8",
        ),
        (
            "Model ladder",
            ["one-step spatial GNN", "six-step GNN–GRU", "six-expert soft MoE"],
            "#fff0e3",
            "#f58518",
        ),
        (
            "Targeted controls",
            ["topology-free\n  temporal model", "matched IID\n  permutation", "label-free baselines"],
            "#e9f6ec",
            "#54a24b",
        ),
        (
            "Paired inference",
            ["shared eligible\n  endpoints", "model seeds nested\n  in data seeds", "AUPRC primary;\n  F1 secondary"],
            "#f1ecf8",
            "#8f63b8",
        ),
    ]
    for x, (title, lines, face, edge) in zip(xs, boxes):
        box(ax, x, y, width, height, title, lines, face, edge)

    for left, right in zip(xs[:-1], xs[1:]):
        arrow(ax, (left + width + 0.005, y + height / 2), (right - 0.006, y + height / 2))

    claim_x, claim_y, claim_w, claim_h = 0.11, 0.07, 0.78, 0.22
    claim = FancyBboxPatch(
        (claim_x, claim_y),
        claim_w,
        claim_h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.6,
        edgecolor="#334155",
        facecolor="#f8fafc",
    )
    ax.add_patch(claim)
    ax.text(
        0.5,
        claim_y + claim_h - 0.055,
        "Defensible conclusions",
        ha="center",
        va="top",
        fontsize=11.5,
        fontweight="bold",
        color="#1f2933",
    )
    ax.text(
        0.5,
        claim_y + 0.075,
        "Topology helps on Modena   •   Temporal recurrence replicates across networks\n"
        "MoE is an accuracy–cost trade-off   •   Severity governs detectability",
        ha="center",
        va="center",
        fontsize=9.0,
        linespacing=1.45,
        color="#25313c",
    )
    arrow(ax, (0.5, y - 0.01), (0.5, claim_y + claim_h + 0.01))

    fig.savefig(OUT / "study_overview.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "study_overview.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT / 'study_overview.png'} and {OUT / 'study_overview.pdf'}")


if __name__ == "__main__":
    main()
