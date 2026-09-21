"""Aggregate topology-aware versus no-message-passing Modena controls."""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

from scipy.stats import t


V2 = Path(__file__).resolve().parents[1]
TEMPORAL_ROOTS = [
    V2 / "outputs" / "runs" / "modena_screening",
    V2 / "outputs" / "runs" / "spatiotemporal_confirmation",
]
CONTROL_ROOTS = [
    V2 / "outputs" / "runs" / "modena_topology_control",
    V2 / "outputs" / "runs" / "modena_topology_confirmation",
]
OUT_JSON = V2 / "outputs" / "runs" / "modena_topology_confirmation_summary.json"
OUT_MD = V2 / "outputs" / "tables" / "modena_topology_confirmation.md"


def load(roots: list[Path], no_topology: bool) -> dict[tuple[int, int], dict]:
    rows = {}
    for root in roots:
        if not root.exists():
            continue
        for directory in root.iterdir():
            args_path = directory / "args.json"
            detail_path = directory / "detailed_analysis.json"
            if not args_path.exists() or not detail_path.exists():
                continue
            args = json.loads(args_path.read_text())
            match = re.search(r"modena_episode_seed(\d+)$", args["data_dir"])
            if not match:
                continue
            is_temporal_single = (
                int(args["window_size"]) == 6 and int(args["num_experts"]) == 1
            )
            if not is_temporal_single or bool(args.get("no_topology", False)) != no_topology:
                continue
            key = (int(match.group(1)), int(args["seed"]))
            rows[key] = json.loads(detail_path.read_text())["aligned_sensor_metrics"]["test"]
    return rows


def ci95(values: list[float]) -> list[float]:
    mean = statistics.mean(values)
    sem = statistics.stdev(values) / len(values) ** 0.5
    return [float(value) for value in t.interval(0.95, len(values) - 1, loc=mean, scale=sem)]


def main() -> None:
    gnn = load(TEMPORAL_ROOTS, False)
    control = load(CONTROL_ROOTS, True)
    expected = {(d, m) for d in (101, 202, 303) for m in (1, 2, 3)}
    common = sorted(expected & set(gnn) & set(control))
    paired = [
        {
            "data_seed": data_seed,
            "model_seed": model_seed,
            **{
                f"{metric}_difference": gnn[(data_seed, model_seed)][metric]
                - control[(data_seed, model_seed)][metric]
                for metric in ("f1", "auprc", "auroc")
            },
        }
        for data_seed, model_seed in common
    ]
    seed_summary = []
    for data_seed in (101, 202, 303):
        group = [row for row in paired if row["data_seed"] == data_seed]
        if not group:
            continue
        item = {"data_seed": data_seed, "model_seeds": len(group)}
        for metric in ("f1", "auprc", "auroc"):
            values = [row[f"{metric}_difference"] for row in group]
            item[f"{metric}_difference_mean"] = statistics.mean(values)
            item[f"{metric}_difference_sd"] = statistics.stdev(values)
        seed_summary.append(item)
    summary = {"data_seeds": len(seed_summary)}
    if len(seed_summary) == 3:
        for metric in ("f1", "auprc", "auroc"):
            values = [row[f"{metric}_difference_mean"] for row in seed_summary]
            summary[f"{metric}_difference_mean"] = statistics.mean(values)
            summary[f"{metric}_difference_ci95"] = ci95(values)
    missing = sorted(expected - set(common))
    payload = {
        "paired_model_seed_differences": paired,
        "data_seed_summary": seed_summary,
        "summary": summary,
        "missing": [list(key) for key in missing],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Modena topology confirmation",
        "",
        "Temporal GNN minus the matched temporal model with message passing disabled.",
        "",
        "| Data seed | ΔF1 | ΔAUPRC | ΔAUROC |",
        "|---:|---:|---:|---:|",
    ]
    for row in seed_summary:
        lines.append(
            f"| {row['data_seed']} | {row['f1_difference_mean']:+.4f} | "
            f"{row['auprc_difference_mean']:+.4f} | "
            f"{row['auroc_difference_mean']:+.4f} |"
        )
    if len(seed_summary) == 3:
        ci = summary["auprc_difference_ci95"]
        lines.extend([
            "",
            f"Mean ΔAUPRC `{summary['auprc_difference_mean']:+.4f}`, "
            f"95% CI `[{ci[0]:+.4f}, {ci[1]:+.4f}]`.",
        ])
    if missing:
        lines.extend(["", f"Incomplete cells: `{missing}`."])
    lines.append("")
    OUT_MD.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
