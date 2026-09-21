"""Plot the completed three-seed Modena architecture screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


V2 = Path(__file__).resolve().parents[1]
ROOT = V2.parent
SUMMARY = V2 / "outputs" / "runs" / "modena_screening_summary.json"
OUT = V2 / "outputs" / "figures"
LABELS = {"spatial": "Spatial GNN", "temporal": "GNN + GRU", "moe": "GNN + GRU + MoE"}
COLORS = {"spatial": "#4c78a8", "temporal": "#54a24b", "moe": "#e45756"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    args = parser.parse_args()
    payload = json.loads(args.summary.read_text())
    rows = {row["architecture"]: row for row in payload["summary"]}
    order = ["spatial", "temporal", "moe"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ("auprc", "f1"),
        ("Sensor-level AUPRC", "Sensor-level F1"),
    ):
        means = [rows[name][f"{metric}_mean"] for name in order]
        errors = [rows[name][f"{metric}_sd"] for name in order]
        x = np.arange(len(order))
        axis.errorbar(
            x, means, yerr=errors, fmt="none", ecolor="#333333",
            elinewidth=1.5, capsize=4, zorder=1,
        )
        axis.scatter(x, means, s=85, c=[COLORS[name] for name in order], zorder=2)
        axis.set_xticks(x, [LABELS[name] for name in order], rotation=15, ha="right")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_ylabel("Test score (mean ± SD over model seeds)")
        axis.grid(axis="y", color="#dddddd", linewidth=0.8)
        lower = min(value - error for value, error in zip(means, errors))
        upper = max(value + error for value, error in zip(means, errors))
        margin = max(0.02, (upper - lower) * 0.45)
        axis.set_ylim(max(0.0, lower - margin), min(1.0, upper + margin))
        for position, name, mean in zip(x, order, means):
            params = rows[name]["n_params"]
            axis.annotate(
                f"{mean:.3f}\n{params / 1000:.0f}k params",
                (position, mean), xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=8,
            )
    fig.suptitle("Modena architecture screening (data seed 101, three model seeds)")
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = OUT / f"modena_architecture_screening.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        print(f"wrote {path.relative_to(ROOT)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
