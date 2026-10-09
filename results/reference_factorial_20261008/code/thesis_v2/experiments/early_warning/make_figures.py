"""Figures for EARLY_WARNING_MULTISEED_RESULTS.md.

Every panel is drawn from a frozen campaign artifact; nothing is recomputed or
rounded by hand here. Panels whose input does not exist yet are skipped and
named in the printed summary, so a partial campaign produces a partial figure
set rather than a plausible-looking blank.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from build_feature_cache import ROOT

CAMPAIGN = ROOT / "runs/operational/early_warning_multiseed_v1"
FIGURES = ROOT / "thesis_v2/outputs/early_warning_multiseed"
FAMILIES = ("random", "replay", "drift", "noise", "targeted")

INK = {"baseline": "#868e96", "C1": "#4dabf7", "C2": "#f76707", "C3": "#2b8a3e"}


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def screen_figure(screen):
    arms = {"baseline": screen["baseline"]} | {
        name: entry["report"] for name, entry in screen["candidates"].items()
        if entry.get("passes_gates")}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    names = list(arms)
    axes[0].bar(names, [arms[n]["pooled_f1"] for n in names],
                color=[INK[n] for n in names])
    for i, n in enumerate(names):
        axes[0].text(i, arms[n]["pooled_f1"] + .01, f"{arms[n]['pooled_f1']:.3f}",
                     ha="center", fontsize=9)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("pooled sensor-endpoint F1")
    axes[0].set_title("Calibration pooled F1")

    width = .8 / len(names)
    for offset, n in enumerate(names):
        axes[1].bar(np.arange(len(FAMILIES)) + offset * width,
                    [arms[n][f] for f in FAMILIES], width, label=n, color=INK[n])
    axes[1].set_xticks(np.arange(len(FAMILIES)) + width * (len(names) - 1) / 2)
    axes[1].set_xticklabels(FAMILIES, rotation=20, fontsize=9)
    axes[1].axhline(.80, ls="--", lw=.8, color="#c92a2a")
    axes[1].axhline(.90, ls=":", lw=.8, color="#c92a2a")
    axes[1].set_ylim(.5, 1.02)
    axes[1].set_ylabel("family F1")
    axes[1].set_title("Family F1 against the frozen gates")
    axes[1].legend(fontsize=8)

    axes[2].bar(names, [arms[n]["clean_period_fp"] for n in names],
                color=[INK[n] for n in names])
    for i, n in enumerate(names):
        axes[2].text(i, arms[n]["clean_period_fp"] * 1.02,
                     str(arms[n]["clean_period_fp"]), ha="center", fontsize=9)
    axes[2].set_ylabel("clean-period false alarms")
    axes[2].set_title("Clean-period false alarms")

    fig.suptitle("Stage C bounded screen — Modena calibration, 99 scenarios "
                 "(a selection surface, not a confirmation)")
    fig.tight_layout()
    fig.savefig(FIGURES / "stage_c_candidate_screen.png", dpi=160)
    plt.close(fig)


def early_figure(summary):
    report = summary["train_oof_report"]["by_family"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    deadlines = [0, 1, 2]
    width = .35
    for offset, family in enumerate(("drift", "noise")):
        entry = report[family]
        axes[0].bar(np.array(deadlines) + offset * width,
                    [entry[f"correct_family_by_plus{d}"] for d in deadlines],
                    width, label=f"{family} (n={entry['events']})",
                    color="#2b8a3e" if family == "noise" else "#c92a2a")
    axes[0].set_xticks(np.array(deadlines) + width / 2)
    axes[0].set_xticklabels([f"onset +{d} h" for d in deadlines])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("fraction of all events")
    axes[0].set_title("Correct-family warning by deadline")
    axes[0].legend(fontsize=8)

    for offset, family in enumerate(("drift", "noise")):
        entry = report[family]
        axes[1].bar(np.array(deadlines) + offset * width,
                    [entry[f"warned_by_plus{d}"] for d in deadlines], width,
                    label=family, color="#2b8a3e" if family == "noise" else "#c92a2a")
    axes[1].set_xticks(np.array(deadlines) + width / 2)
    axes[1].set_xticklabels([f"onset +{d} h" for d in deadlines])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("fraction of all events")
    axes[1].set_title("Any warning by deadline (misses included)")
    axes[1].legend(fontsize=8)

    matrix = summary["train_oof_report"]["confusion"]["matrix"]
    columns = summary["train_oof_report"]["confusion"]["columns"]
    rows = list(matrix)
    grid = np.array([[matrix[r][c] for c in columns] for r in rows], dtype=float)
    normalised = grid / np.maximum(grid.sum(1, keepdims=True), 1)
    image = axes[2].imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    axes[2].set_xticks(range(len(columns)))
    axes[2].set_xticklabels(columns, rotation=35, ha="right", fontsize=8)
    axes[2].set_yticks(range(len(rows)))
    axes[2].set_yticklabels(rows, fontsize=8)
    for i in range(len(rows)):
        for j in range(len(columns)):
            axes[2].text(j, i, f"{int(grid[i, j])}", ha="center", va="center",
                         fontsize=7, color="white" if normalised[i, j] > .5 else "black")
    axes[2].set_title("Graph-time confusion (rows: true state)")
    fig.colorbar(image, ax=axes[2], fraction=.046)

    fig.suptitle("Stage B early warning — Modena TRAIN out-of-fold, onset clock")
    fig.tight_layout()
    fig.savefig(FIGURES / "stage_b_early_warning.png", dpi=160)
    plt.close(fig)


def seed_figure(per_seed):
    """Baseline vs candidate across training seeds, once Stage E has results."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    seeds = sorted(per_seed)
    for arm, colour in (("baseline", INK["baseline"]), ("candidate", INK["C3"])):
        values = [per_seed[s][arm] for s in seeds]
        axes[0].plot(range(len(seeds)), values, marker="o", label=arm, color=colour)
    axes[0].set_xticks(range(len(seeds)))
    axes[0].set_xticklabels(seeds)
    axes[0].set_xlabel("training seed")
    axes[0].set_ylabel("pooled F1")
    axes[0].set_title("Fresh-evaluation pooled F1 by training seed")
    axes[0].legend(fontsize=8)

    paired = [per_seed[s]["candidate"] - per_seed[s]["baseline"] for s in seeds]
    axes[1].bar(range(len(seeds)), paired, color=INK["C3"])
    axes[1].axhline(0, color="black", lw=.8)
    axes[1].set_xticks(range(len(seeds)))
    axes[1].set_xticklabels(seeds)
    axes[1].set_xlabel("training seed")
    axes[1].set_ylabel("candidate - baseline")
    axes[1].set_title("Paired difference")
    fig.tight_layout()
    fig.savefig(FIGURES / "stage_e_seed_variability.png", dpi=160)
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    drawn, skipped = [], []

    screen = load(CAMPAIGN / "verifier_screen_v1/selection_frozen.json")
    if screen:
        screen_figure(screen)
        drawn.append("stage_c_candidate_screen.png")
    else:
        skipped.append("stage_c_candidate_screen.png (Stage C screen not frozen)")

    early = load(CAMPAIGN / "early_head_v1/summary.json")
    if early:
        early_figure(early)
        drawn.append("stage_b_early_warning.png")
    else:
        skipped.append("stage_b_early_warning.png (early head not fitted)")

    per_seed = {}
    for path in sorted((CAMPAIGN / "stage_e_modena").glob("seed/*/evaluation_report.json")):
        report = json.loads(path.read_text())
        pooled = {}
        for arm in ("baseline", "candidate"):
            values = [entry["arms"][arm]["pooled_f1"]
                      for entry in report["per_data_seed"].values()
                      if arm in entry["arms"]]
            if values:
                pooled[arm] = float(np.mean(values))
        if len(pooled) == 2:
            per_seed[str(report["training_seed"])] = pooled
    if per_seed:
        seed_figure(per_seed)
        drawn.append("stage_e_seed_variability.png")
    else:
        skipped.append("stage_e_seed_variability.png (no fresh evaluation reports yet)")

    print(json.dumps({"drawn": drawn, "skipped": skipped,
                      "directory": str(FIGURES.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
