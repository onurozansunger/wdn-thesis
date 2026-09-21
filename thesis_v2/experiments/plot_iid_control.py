"""Visualise how the matched IID control removes episode persistence."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


V2 = Path(__file__).resolve().parents[1]
ROOT = V2.parent
OUT = V2 / "outputs" / "figures"
NAMES = ["clean", "random", "replay", "stealthy", "noise", "targeted"]
COLORS = ["#d9d9d9", "#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2"]


def family_matrix(data_dir: Path, scenario_ids: list[int]) -> np.ndarray:
    with (data_dir / "snapshots.pkl").open("rb") as stream:
        snapshots = pickle.load(stream)
    with (data_dir / "corrupted.pkl").open("rb") as stream:
        corrupted = pickle.load(stream)
    rows = []
    for scenario in scenario_ids:
        values = [
            (int(snap.timestep), int(getattr(corr, "attack_type_id", 0)))
            for snap, corr in zip(snapshots, corrupted)
            if int(snap.scenario_id) == scenario
        ]
        rows.append([family for _, family in sorted(values)])
    return np.asarray(rows, dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episode",
        type=Path,
        default=ROOT / "data" / "thesis_v2" / "modena_episode_seed101",
    )
    parser.add_argument(
        "--iid",
        type=Path,
        default=ROOT / "data" / "thesis_v2" / "modena_iid_seed101",
    )
    parser.add_argument("--scenarios", type=int, default=8)
    args = parser.parse_args()
    scenario_ids = list(range(args.scenarios))
    episode = family_matrix(args.episode, scenario_ids)
    iid = family_matrix(args.iid, scenario_ids)

    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(np.arange(-0.5, len(NAMES) + 0.5), cmap.N)
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True, constrained_layout=True)
    for axis, matrix, title in zip(
        axes,
        (episode, iid),
        ("Coherent episodes", "Marginally matched IID control"),
    ):
        axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
        axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
        axis.set_ylabel("Scenario")
        axis.set_yticks(range(len(scenario_ids)), scenario_ids)
        axis.set_xticks(np.arange(0, matrix.shape[1], 4))
        axis.grid(False)
    axes[-1].set_xlabel("Timestep")
    handles = [Patch(facecolor=color, label=name) for name, color in zip(NAMES, COLORS)]
    fig.legend(handles=handles, loc="outside lower center", ncol=6, frameon=False)
    fig.suptitle(
        "Temporal permutation breaks persistence while preserving attack marginals",
        fontsize=12,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = OUT / f"matched_iid_control.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        print(f"wrote {path.relative_to(ROOT)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
