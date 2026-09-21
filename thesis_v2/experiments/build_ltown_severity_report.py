"""Aggregate severity-stratified L-Town results and create a thesis figure.

Recall is evaluated at each run's validation-calibrated threshold. AUROC uses
each severity bin's attacked observations against the same clean-test pool.
Model-seed repetitions are nested inside data seeds for summaries.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


V2 = Path(__file__).resolve().parents[1]
RUN_ROOTS = (V2 / "outputs" / "runs" / "spatiotemporal_confirmation",)
OUT_JSON = V2 / "outputs" / "runs" / "ltown_severity_summary.json"
OUT_MD = V2 / "outputs" / "tables" / "ltown_severity.md"
OUT_PNG = V2 / "outputs" / "figures" / "ltown_severity.png"
OUT_PDF = V2 / "outputs" / "figures" / "ltown_severity.pdf"

BINS = ["<0.5", "0.5-1", "1-2", "2-4", ">=4"]
LABELS = [r"$<0.5$", r"$0.5$--$1$", r"$1$--$2$", r"$2$--$4$", r"$\geq4$"]
ARCH_ORDER = ["spatial", "temporal"]
DISPLAY = {"spatial": "Spatial", "temporal": "Temporal"}
COLORS = {"spatial": "#4C78A8", "temporal": "#F58518"}


def architecture(args: dict) -> str:
    return "spatial" if int(args["window_size"]) == 1 else "temporal"


def ltown_data_seed(data_dir: str) -> int | None:
    match = re.search(r"ltown_episode_seed(\d+)$", data_dir)
    return int(match.group(1)) if match else None


def mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main() -> None:
    runs = []
    for root in RUN_ROOTS:
        if not root.exists():
            continue
        for directory in sorted(root.iterdir()):
            args_path = directory / "args.json"
            detail_path = directory / "detailed_analysis.json"
            if not args_path.exists() or not detail_path.exists():
                continue
            args = json.loads(args_path.read_text())
            data_seed = ltown_data_seed(args["data_dir"])
            if (
                data_seed is None
                or int(args["num_experts"]) != 1
                or bool(args.get("no_topology", False))
            ):
                continue
            detail = json.loads(detail_path.read_text())
            severity = detail["severity"]["overall"]
            runs.append(
                {
                    "run_id": directory.name,
                    "architecture": architecture(args),
                    "data_seed": data_seed,
                    "model_seed": int(args["seed"]),
                    "bins": {
                        name: {
                            "positives": int(severity[name]["positives"]),
                            "recall": float(severity[name]["recall"]),
                            "auroc": float(severity[name]["auroc"]),
                        }
                        for name in BINS
                    },
                }
            )

    summary = {}
    for arch in ARCH_ORDER:
        selected = [run for run in runs if run["architecture"] == arch]
        if len(selected) != 9:
            raise SystemExit(f"expected nine L-Town {arch} runs, found {len(selected)}")
        data_seed_means = {
            data_seed: {
                name: {
                    metric: statistics.mean(
                        run["bins"][name][metric]
                        for run in selected
                        if run["data_seed"] == data_seed
                    )
                    for metric in ("recall", "auroc")
                }
                for name in BINS
            }
            for data_seed in (101, 202, 303)
        }
        summary[arch] = {
            name: {
                metric: mean_sd(
                    [data_seed_means[seed][name][metric] for seed in (101, 202, 303)]
                )
                for metric in ("recall", "auroc")
            }
            for name in BINS
        }
        correlations = []
        monotonic_runs = 0
        for run in selected:
            recalls = [run["bins"][name]["recall"] for name in BINS]
            correlations.append(float(spearmanr(range(len(BINS)), recalls).statistic))
            monotonic_runs += int(all(a <= b for a, b in zip(recalls, recalls[1:])))
        per_seed_correlations = [
            statistics.mean(
                correlations[index]
                for index, run in enumerate(selected)
                if run["data_seed"] == data_seed
            )
            for data_seed in (101, 202, 303)
        ]
        summary[arch]["ordered_bin_recall_spearman"] = mean_sd(per_seed_correlations)
        summary[arch]["monotonic_recall_runs"] = monotonic_runs
        summary[arch]["total_runs"] = len(selected)
        summary[arch]["data_seed_means"] = data_seed_means

    payload = {
        "definition": (
            "Absolute reported-minus-clean displacement divided by the training "
            "sensor's robust clean residual scale."
        ),
        "interpretation": (
            "Recall is comparable across bins at a validation-calibrated threshold; "
            "AUROC compares each attacked bin with the shared clean-test pool. "
            "Means and error bars use the three data-seed means; model seeds are "
            "nested repetitions."
        ),
        "runs": runs,
        "summary": summary,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# L-Town severity analysis",
        "",
        payload["definition"],
        "",
        "| Standardised displacement | Spatial recall | Temporal recall |",
        "|---:|---:|---:|",
    ]
    for name in BINS:
        values = [summary[arch][name]["recall"]["mean"] for arch in ARCH_ORDER]
        lines.append(f"| {name} | " + " | ".join(f"{value:.3f}" for value in values) + " |")
    lines.extend(["", "Ordered-bin severity checks:", ""])
    for arch in ARCH_ORDER:
        stat = summary[arch]["ordered_bin_recall_spearman"]
        lines.append(
            f"- **{DISPLAY[arch]}**: Spearman `{stat['mean']:.3f} +/- {stat['sd']:.3f}`; "
            f"monotonic in `{summary[arch]['monotonic_recall_runs']}/{summary[arch]['total_runs']}` runs."
        )
    lines.append("")
    OUT_MD.write_text("\n".join(lines))

    x = np.arange(len(BINS))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), constrained_layout=True)
    for arch in ARCH_ORDER:
        for ax, metric, ylabel in (
            (axes[0], "recall", "Recall at calibrated threshold"),
            (axes[1], "auroc", "AUROC vs shared clean pool"),
        ):
            means = [summary[arch][name][metric]["mean"] for name in BINS]
            errors = [summary[arch][name][metric]["sd"] for name in BINS]
            ax.errorbar(
                x,
                means,
                yerr=errors,
                marker="o",
                linewidth=2,
                capsize=3,
                label=DISPLAY[arch],
                color=COLORS[arch],
            )
            ax.set_ylabel(ylabel)
            ax.set_xticks(x, LABELS)
            ax.set_xlabel(r"Absolute displacement / clean residual scale ($\sigma$)")
            ax.set_ylim(0, 1.02)
            ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, loc="upper left")
    fig.savefig(OUT_PNG, dpi=220)
    fig.savefig(OUT_PDF)
    plt.close(fig)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
