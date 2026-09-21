"""Aggregate paired MoE-versus-single-expert confirmation on Modena."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from scipy.stats import t


V2 = Path(__file__).resolve().parents[1]
ROOTS = [
    V2 / "outputs" / "runs" / "modena_screening",
    V2 / "outputs" / "runs" / "spatiotemporal_confirmation",
    V2 / "outputs" / "runs" / "modena_moe_confirmation",
]
OUT_JSON = V2 / "outputs" / "runs" / "modena_moe_confirmation_summary.json"
OUT_MD = V2 / "outputs" / "tables" / "modena_moe_confirmation.md"


def identity(args: dict) -> tuple[int, int, str] | None:
    match = re.search(r"modena_episode_seed(\d+)$", args["data_dir"])
    if not match:
        return None
    if int(args["num_experts"]) > 1:
        arch = "moe"
    elif int(args["window_size"]) > 1:
        arch = "temporal"
    else:
        return None
    return int(match.group(1)), int(args["seed"]), arch


def load() -> list[dict]:
    rows = []
    for root in ROOTS:
        if not root.exists():
            continue
        for directory in sorted(root.iterdir()):
            args_path = directory / "args.json"
            detail_path = directory / "detailed_analysis.json"
            result_path = directory / "test_results.json"
            if not args_path.exists() or not result_path.exists():
                continue
            args = json.loads(args_path.read_text())
            key = identity(args)
            if key is None:
                continue
            data_seed, model_seed, arch = key
            if detail_path.exists():
                metrics = json.loads(detail_path.read_text())["aligned_sensor_metrics"]["test"]
            else:
                metrics = json.loads(result_path.read_text())["anomaly_detection"]["pressure"]
            rows.append({
                "run_id": directory.name,
                "data_seed": data_seed,
                "model_seed": model_seed,
                "architecture": arch,
                **{name: float(metrics[name]) for name in ("f1", "auprc", "auroc")},
            })
    return rows


def ci95(values: list[float]) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    mean = statistics.mean(values)
    sem = statistics.stdev(values) / len(values) ** 0.5
    low, high = t.interval(0.95, len(values) - 1, loc=mean, scale=sem)
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    rows = load()
    index = {
        (row["data_seed"], row["model_seed"], row["architecture"]): row
        for row in rows
    }
    expected = [
        (data_seed, model_seed, arch)
        for data_seed in (101, 202, 303)
        for model_seed in (1, 2, 3)
        for arch in ("temporal", "moe")
    ]
    missing = [key for key in expected if key not in index]
    if missing and not args.allow_incomplete:
        raise SystemExit(f"MoE confirmation incomplete; missing {missing}")

    pairs = []
    for data_seed in (101, 202, 303):
        for model_seed in (1, 2, 3):
            temporal = index.get((data_seed, model_seed, "temporal"))
            moe = index.get((data_seed, model_seed, "moe"))
            if not temporal or not moe:
                continue
            pairs.append({
                "data_seed": data_seed,
                "model_seed": model_seed,
                **{
                    f"{metric}_difference": moe[metric] - temporal[metric]
                    for metric in ("f1", "auprc", "auroc")
                },
            })

    seed_summary = []
    for data_seed in (101, 202, 303):
        group = [row for row in pairs if row["data_seed"] == data_seed]
        if not group:
            continue
        item = {"data_seed": data_seed, "model_seeds": len(group)}
        for metric in ("f1", "auprc", "auroc"):
            values = [row[f"{metric}_difference"] for row in group]
            item[f"{metric}_difference_mean"] = statistics.mean(values)
            item[f"{metric}_difference_sd"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        seed_summary.append(item)

    summary = {"data_seeds": len(seed_summary)}
    for metric in ("f1", "auprc", "auroc"):
        values = [row[f"{metric}_difference_mean"] for row in seed_summary]
        if values:
            summary[f"{metric}_difference_mean"] = statistics.mean(values)
            summary[f"{metric}_difference_ci95"] = ci95(values)
    auprc_ci = summary.get("auprc_difference_ci95")
    summary["predeclared_performance_gate_pass"] = bool(
        len(seed_summary) == 3
        and summary["auprc_difference_mean"] >= 0.02
        and auprc_ci is not None and auprc_ci[0] > 0
        and all(row["auprc_difference_mean"] > 0 for row in seed_summary)
    )

    payload = {
        "runs": rows,
        "paired_model_seed_differences": pairs,
        "data_seed_summary": seed_summary,
        "summary": summary,
        "missing": [list(key) for key in missing],
        "gate": (
            "Mean paired AUPRC gain >= 0.02, data-seed 95% CI excludes zero, "
            "and all data-seed mean differences are positive."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Modena MoE confirmation",
        "",
        payload["gate"],
        "",
        "| Data seed | ΔF1 MoE-temporal | ΔAUPRC | ΔAUROC |",
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
            f"Mean paired ΔAUPRC: `{summary['auprc_difference_mean']:+.4f}`; "
            f"95% CI `[{ci[0]:+.4f}, {ci[1]:+.4f}]`.",
            "",
            f"Performance gate: `{'PASS' if summary['predeclared_performance_gate_pass'] else 'FAIL'}`.",
        ])
    if missing:
        lines.extend(["", f"Incomplete cells: `{missing}`."])
    lines.append("")
    OUT_MD.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
