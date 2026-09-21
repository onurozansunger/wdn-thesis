"""Plot paired AUPRC effect sizes from confirmatory JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


V2 = Path(__file__).resolve().parents[1]
SPATIOTEMPORAL = V2 / "outputs" / "runs" / "spatiotemporal_confirmation_summary.json"
MODENA_TOPOLOGY = V2 / "outputs" / "runs" / "modena_topology_confirmation_summary.json"
LTOWN_TOPOLOGY = V2 / "outputs" / "runs" / "ltown_topology_confirmation_summary.json"
MOE = V2 / "outputs" / "runs" / "modena_moe_confirmation_summary.json"
OUT_PNG = V2 / "outputs" / "figures" / "confirmatory_effects.png"
OUT_PDF = V2 / "outputs" / "figures" / "confirmatory_effects.pdf"


def main() -> None:
    st = json.loads(SPATIOTEMPORAL.read_text())
    modena_topology = json.loads(MODENA_TOPOLOGY.read_text())
    ltown_topology = json.loads(LTOWN_TOPOLOGY.read_text())
    moe = json.loads(MOE.read_text()) if MOE.exists() else {"data_seed_summary": []}

    groups = [
        (
            "Temporal − spatial\nModena",
            [
                row["auprc_difference_mean"]
                for row in st["data_seed_summary"]
                if row["network"] == "modena"
            ],
            "#4C78A8",
        ),
        (
            "Temporal − spatial\nL-Town",
            [
                row["auprc_difference_mean"]
                for row in st["data_seed_summary"]
                if row["network"] == "ltown"
            ],
            "#F58518",
        ),
        (
            "Topology − no messages\nModena",
            [
                row["auprc_difference_mean"]
                for row in modena_topology["data_seed_summary"]
            ],
            "#72B7B2",
        ),
        (
            "Topology − no messages\nL-Town",
            [
                row["auprc_difference_mean"]
                for row in ltown_topology["data_seed_summary"]
            ],
            "#B279A2",
        ),
        (
            "MoE − temporal\nModena",
            [row["auprc_difference_mean"] for row in moe.get("data_seed_summary", [])],
            "#54A24B",
        ),
    ]

    fig, ax = plt.subplots(figsize=(10.4, 4.6), constrained_layout=True)
    rng = np.random.default_rng(17)
    for index, (label, values, color) in enumerate(groups):
        if not values:
            continue
        jitter = rng.uniform(-0.075, 0.075, size=len(values))
        ax.scatter(
            np.full(len(values), index) + jitter,
            values,
            s=52,
            color=color,
            alpha=0.85,
            label="Data-seed mean" if index == 0 else None,
            zorder=3,
        )
        mean = float(np.mean(values))
        ax.plot([index - 0.18, index + 0.18], [mean, mean], color="black", linewidth=2.4)
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(0.02, color="#777777", linestyle="--", linewidth=1.2,
               label="MoE practical gate (+0.02)")
    ax.set_xticks(range(len(groups)), [group[0] for group in groups])
    ax.set_ylabel("Paired AUPRC difference")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="best")
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=220)
    fig.savefig(OUT_PDF)
    plt.close(fig)
    print(f"wrote {OUT_PNG.relative_to(V2.parent)} and {OUT_PDF.relative_to(V2.parent)}")


if __name__ == "__main__":
    main()
