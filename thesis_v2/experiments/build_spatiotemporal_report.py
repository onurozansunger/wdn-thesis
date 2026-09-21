"""Aggregate paired spatial-versus-temporal confirmation results.

Model seeds are repeated measurements nested inside a generated data seed.
For Modena, confidence intervals are therefore formed over the three data-seed
mean paired differences rather than pretending all nine runs are independent.
Both networks are summarised over three independently generated data seeds.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from scipy.stats import t


V2 = Path(__file__).resolve().parents[1]
SCREENING = V2 / "outputs" / "runs" / "modena_screening"
CONFIRMATION = V2 / "outputs" / "runs" / "spatiotemporal_confirmation"
OUT_JSON = V2 / "outputs" / "runs" / "spatiotemporal_confirmation_summary.json"
OUT_MD = V2 / "outputs" / "tables" / "spatiotemporal_confirmation.md"


def architecture(args: dict) -> str | None:
    if int(args["num_experts"]) != 1:
        return None
    return "spatial" if int(args["window_size"]) == 1 else "temporal"


def dataset_identity(data_dir: str) -> tuple[str, int]:
    match = re.search(r"/(modena|ltown)_episode_seed(\d+)$", data_dir)
    if not match:
        raise ValueError(f"unrecognised thesis dataset: {data_dir}")
    return match.group(1), int(match.group(2))


def load(root: Path) -> list[dict]:
    rows = []
    if not root.exists():
        return rows
    for directory in sorted(root.iterdir()):
        args_path = directory / "args.json"
        result_path = directory / "test_results.json"
        if not args_path.exists() or not result_path.exists():
            continue
        args = json.loads(args_path.read_text())
        arch = architecture(args)
        if arch is None:
            continue
        network, data_seed = dataset_identity(args["data_dir"])
        pressure = json.loads(result_path.read_text())["anomaly_detection"]["pressure"]
        detail_path = directory / "detailed_analysis.json"
        if detail_path.exists():
            pressure = json.loads(detail_path.read_text())["aligned_sensor_metrics"]["test"]
        rows.append(
            {
                "run_id": directory.name,
                "network": network,
                "data_seed": data_seed,
                "model_seed": int(args["seed"]),
                "architecture": arch,
                "f1": float(pressure["f1"]),
                "auprc": float(pressure["auprc"]),
                "auroc": float(pressure["auroc"]),
            }
        )
    return rows


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def ci95(values: list[float]) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    mean = statistics.mean(values)
    sem = statistics.stdev(values) / len(values) ** 0.5
    low, high = t.interval(0.95, df=len(values) - 1, loc=mean, scale=sem)
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    rows = load(SCREENING) + load(CONFIRMATION)
    index = {
        (row["network"], row["data_seed"], row["model_seed"], row["architecture"]): row
        for row in rows
    }
    expected = [
        ("modena", data_seed, model_seed, architecture)
        for data_seed in (101, 202, 303)
        for model_seed in (1, 2, 3)
        for architecture in ("spatial", "temporal")
    ] + [
        ("ltown", data_seed, model_seed, architecture)
        for data_seed in (101, 202, 303)
        for model_seed in (1, 2, 3)
        for architecture in ("spatial", "temporal")
    ]
    missing = [key for key in expected if key not in index]
    if missing and not args.allow_incomplete:
        raise SystemExit(f"confirmation incomplete; missing {missing}")

    paired = []
    for network, data_seed, model_seed, _ in expected:
        if _ != "spatial":
            continue
        spatial = index.get((network, data_seed, model_seed, "spatial"))
        temporal = index.get((network, data_seed, model_seed, "temporal"))
        if not spatial or not temporal:
            continue
        paired.append(
            {
                "network": network,
                "data_seed": data_seed,
                "model_seed": model_seed,
                **{
                    f"{metric}_difference": temporal[metric] - spatial[metric]
                    for metric in ("f1", "auprc", "auroc")
                },
            }
        )

    seed_groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in paired:
        seed_groups[(row["network"], row["data_seed"])].append(row)
    seed_summary = []
    for (network, data_seed), group in sorted(seed_groups.items()):
        item = {"network": network, "data_seed": data_seed, "model_seeds": len(group)}
        for metric in ("f1", "auprc", "auroc"):
            values = [row[f"{metric}_difference"] for row in group]
            mean, sd = mean_sd(values)
            item[f"{metric}_difference_mean"] = mean
            item[f"{metric}_difference_sd"] = sd
        seed_summary.append(item)

    network_summary = []
    for network in ("modena", "ltown"):
        groups = [row for row in seed_summary if row["network"] == network]
        if not groups:
            continue
        item = {"network": network, "data_seeds": len(groups)}
        for metric in ("f1", "auprc", "auroc"):
            values = [row[f"{metric}_difference_mean"] for row in groups]
            item[f"{metric}_difference_mean"] = statistics.mean(values)
            item[f"{metric}_difference_ci95"] = ci95(values)
        network_summary.append(item)

    architecture_seed_summary = []
    for network in ("modena", "ltown"):
        for data_seed in (101, 202, 303):
            for arch in ("spatial", "temporal"):
                group = [
                    row for row in rows
                    if row["network"] == network
                    and row["data_seed"] == data_seed
                    and row["architecture"] == arch
                ]
                if not group:
                    continue
                item = {
                    "network": network,
                    "data_seed": data_seed,
                    "architecture": arch,
                    "model_seeds": len(group),
                }
                for metric in ("f1", "auprc", "auroc"):
                    item[f"{metric}_mean"] = statistics.mean(
                        row[metric] for row in group
                    )
                architecture_seed_summary.append(item)

    architecture_network_summary = []
    for network in ("modena", "ltown"):
        for arch in ("spatial", "temporal"):
            group = [
                row for row in architecture_seed_summary
                if row["network"] == network and row["architecture"] == arch
            ]
            if not group:
                continue
            item = {
                "network": network,
                "architecture": arch,
                "data_seeds": len(group),
            }
            for metric in ("f1", "auprc", "auroc"):
                values = [row[f"{metric}_mean"] for row in group]
                item[f"{metric}_mean"] = statistics.mean(values)
                item[f"{metric}_ci95"] = ci95(values)
            architecture_network_summary.append(item)

    payload = {
        "runs": rows,
        "paired_model_seed_differences": paired,
        "data_seed_summary": seed_summary,
        "network_summary": network_summary,
        "architecture_data_seed_summary": architecture_seed_summary,
        "architecture_network_summary": architecture_network_summary,
        "missing": [list(key) for key in missing],
        "inference_note": (
            "For each network, 95% t intervals use data-seed mean paired "
            "differences (n=3); model seeds are nested repetitions."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Spatial–temporal confirmation",
        "",
        payload["inference_note"],
        "",
        "| Network | Data seed | ΔF1 temporal-spatial | ΔAUPRC | ΔAUROC |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in seed_summary:
        lines.append(
            f"| {row['network']} | {row['data_seed']} | "
            f"{row['f1_difference_mean']:+.4f} | "
            f"{row['auprc_difference_mean']:+.4f} | "
            f"{row['auroc_difference_mean']:+.4f} |"
        )
    lines.extend(["", "## Network-level summary", ""])
    for row in network_summary:
        ci = row["auprc_difference_ci95"]
        ci_text = "descriptive only" if ci is None else f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]"
        lines.append(
            f"- **{row['network']}**: mean paired ΔAUPRC "
            f"{row['auprc_difference_mean']:+.4f}; {ci_text}."
        )
    lines.extend([
        "",
        "## Absolute performance over data seeds",
        "",
        "| Network | Architecture | F1 | AUPRC | AUPRC 95% CI | AUROC |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in architecture_network_summary:
        ci = row["auprc_ci95"]
        ci_text = "descriptive" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
        lines.append(
            f"| {row['network']} | {row['architecture']} | "
            f"{row['f1_mean']:.4f} | {row['auprc_mean']:.4f} | {ci_text} | "
            f"{row['auroc_mean']:.4f} |"
        )
    if missing:
        lines.extend(["", f"Incomplete cells: `{missing}`", ""])
    else:
        lines.append("")
    OUT_MD.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
