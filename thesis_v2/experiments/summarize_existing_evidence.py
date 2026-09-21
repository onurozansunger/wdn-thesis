"""Summarise the copied Modena episode ablation artifacts.

The script deliberately reads only JSON files stored under thesis_v2/evidence
and writes machine-readable and Markdown summaries under thesis_v2/outputs.
It does not depend on training checkpoints.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path


V2 = Path(__file__).resolve().parents[1]
EVIDENCE = V2 / "evidence" / "episode_ablation"
OUT_RUNS = V2 / "outputs" / "runs"
OUT_TABLES = V2 / "outputs" / "tables"


def configuration(args: dict) -> str:
    if args.get("no_topology", False):
        return "no_topology"
    if args.get("window_size") == 1:
        return "no_temporal"
    if args.get("num_experts") == 1:
        return "no_mixture"
    return "full"


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def main() -> None:
    rows: list[dict] = []
    for run_dir in sorted(EVIDENCE.iterdir()):
        if not run_dir.is_dir():
            continue
        args = json.loads((run_dir / "args.json").read_text())
        result = json.loads((run_dir / "test_results.json").read_text())
        pressure = result["anomaly_detection"]["pressure"]
        rows.append(
            {
                "run_id": run_dir.name,
                "config": configuration(args),
                "seed": int(args["seed"]),
                "f1": float(pressure["f1"]),
                "auroc": float(pressure["auroc"]),
                "router_acc": float(result.get("router_acc", 0.0)),
            }
        )

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["config"]].append(row)

    order = ["full", "no_topology", "no_temporal", "no_mixture"]
    summary = []
    for name in order:
        group = sorted(grouped[name], key=lambda row: row["seed"])
        f1_mean, f1_sd = mean_sd([row["f1"] for row in group])
        auroc_mean, auroc_sd = mean_sd([row["auroc"] for row in group])
        full_by_seed = {row["seed"]: row for row in grouped["full"]}
        differences = [
            full_by_seed[row["seed"]]["f1"] - row["f1"]
            for row in group
            if name != "full" and row["seed"] in full_by_seed
        ]
        summary.append(
            {
                "config": name,
                "n": len(group),
                "f1_mean": f1_mean,
                "f1_sd": f1_sd,
                "auroc_mean": auroc_mean,
                "auroc_sd": auroc_sd,
                "full_minus_ablation_f1_mean": (
                    statistics.mean(differences) if differences else None
                ),
                "full_minus_ablation_f1_sd": (
                    statistics.stdev(differences) if len(differences) > 1 else None
                ),
            }
        )

    OUT_RUNS.mkdir(parents=True, exist_ok=True)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    (OUT_RUNS / "episode_ablation_summary.json").write_text(
        json.dumps({"runs": rows, "summary": summary}, indent=2) + "\n"
    )

    lines = [
        "# Existing Modena episode ablation",
        "",
        "| Configuration | n | F1 mean +/- SD | AUROC mean +/- SD | Full - ablation F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        diff = row["full_minus_ablation_f1_mean"]
        diff_text = "--" if diff is None else f"{diff:+.4f}"
        lines.append(
            f"| {row['config']} | {row['n']} | "
            f"{row['f1_mean']:.4f} +/- {row['f1_sd']:.4f} | "
            f"{row['auroc_mean']:.4f} +/- {row['auroc_sd']:.4f} | "
            f"{diff_text} |"
        )
    lines.extend(
        [
            "",
            "These are provisional results from one generated dataset. They do not",
            "measure data-generation uncertainty.",
            "",
        ]
    )
    (OUT_TABLES / "episode_ablation.md").write_text("\n".join(lines))

    print("\n".join(lines))


if __name__ == "__main__":
    main()

