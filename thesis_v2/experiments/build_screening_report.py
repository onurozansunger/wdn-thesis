"""Aggregate the three-seed Modena architecture screening."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


V2 = Path(__file__).resolve().parents[1]
ROOT = V2.parent
DEFAULT_RUN_ROOT = V2 / "outputs" / "runs" / "modena_screening"
OUT_JSON = V2 / "outputs" / "runs" / "modena_screening_summary.json"
OUT_TABLE = V2 / "outputs" / "tables" / "modena_screening.md"


def architecture(args: dict) -> str:
    if int(args["window_size"]) == 1 and int(args["num_experts"]) == 1:
        return "spatial"
    if int(args["window_size"]) > 1 and int(args["num_experts"]) == 1:
        return "temporal"
    if int(args["window_size"]) > 1 and int(args["num_experts"]) > 1:
        return "moe"
    return "unknown"


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def load_runs(run_root: Path) -> list[dict]:
    rows = []
    for directory in sorted(run_root.iterdir()):
        args_path = directory / "args.json"
        result_path = directory / "test_results.json"
        if not args_path.exists() or not result_path.exists():
            continue
        args = json.loads(args_path.read_text())
        if not str(args.get("run_tag", "")).startswith("screen_"):
            continue
        result = json.loads(result_path.read_text())
        pressure = result["anomaly_detection"]["pressure"]
        row = {
            "run_id": directory.name,
            "architecture": architecture(args),
            "model_seed": int(args["seed"]),
            "n_params": int(result["n_params"]),
            "f1": float(pressure["f1"]),
            "auprc": float(pressure["auprc"]),
            "auroc": float(pressure["auroc"]),
            "precision": float(pressure["precision"]),
            "recall": float(pressure["recall"]),
            "per_attack": result["per_attack_pressure"],
        }
        detail_path = directory / "detailed_analysis.json"
        if detail_path.exists():
            detail = json.loads(detail_path.read_text())
            aligned = detail["aligned_sensor_metrics"]["test"]
            row["raw_unaligned"] = {
                metric: row[metric]
                for metric in ("f1", "auprc", "auroc", "precision", "recall")
            }
            for metric in ("f1", "auprc", "auroc", "precision", "recall"):
                row[metric] = float(aligned[metric])
            event = detail["events"]["overall"]
            row["event_detection_rate"] = float(event["detection_rate"])
            row["event_delay"] = event["median_delay_from_episode_start"]
            row["event_false_alarm_rate"] = float(
                detail["events"]["clean_endpoints"]["false_alarm_rate"]
            )
            row["severity_recall"] = {
                name: values.get("recall")
                for name, values in detail["severity"]["overall"].items()
            }
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    args = parser.parse_args()
    rows = load_runs(args.run_root)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["architecture"]].append(row)

    expected = {"spatial", "temporal", "moe"}
    incomplete = {
        name: sorted({1, 2, 3} - {row["model_seed"] for row in grouped[name]})
        for name in expected
        if {row["model_seed"] for row in grouped[name]} != {1, 2, 3}
    }
    if incomplete:
        raise SystemExit(f"screening incomplete: {incomplete}")

    summary = []
    for name in ("spatial", "temporal", "moe"):
        group = sorted(grouped[name], key=lambda row: row["model_seed"])
        item = {"architecture": name, "n": len(group), "n_params": group[0]["n_params"]}
        for metric in ("f1", "auprc", "auroc", "precision", "recall"):
            mean, sd = mean_sd([row[metric] for row in group])
            item[f"{metric}_mean"] = mean
            item[f"{metric}_sd"] = sd
        if all("event_detection_rate" in row for row in group):
            mean, sd = mean_sd([row["event_detection_rate"] for row in group])
            item["event_detection_rate_mean"] = mean
            item["event_detection_rate_sd"] = sd
            delays = [row["event_delay"] for row in group if row["event_delay"] is not None]
            item["event_delay_mean"] = statistics.mean(delays) if delays else None
            mean, sd = mean_sd([row["event_false_alarm_rate"] for row in group])
            item["event_false_alarm_rate_mean"] = mean
            item["event_false_alarm_rate_sd"] = sd
        summary.append(item)

    by_arch_seed = {
        (row["architecture"], row["model_seed"]): row for row in rows
    }
    paired = {}
    for left, right, label in (
        ("temporal", "spatial", "temporal_minus_spatial"),
        ("moe", "temporal", "moe_minus_temporal"),
    ):
        paired[label] = {}
        for metric in ("f1", "auprc", "auroc"):
            differences = [
                by_arch_seed[(left, seed)][metric] - by_arch_seed[(right, seed)][metric]
                for seed in (1, 2, 3)
            ]
            paired[label][metric] = {
                "values": differences,
                "mean": statistics.mean(differences),
                "sd": statistics.stdev(differences),
            }

    moe_auprc = paired["moe_minus_temporal"]["auprc"]
    gate = {
        "required_mean_auprc_gain": 0.02,
        "observed_mean_auprc_gain": moe_auprc["mean"],
        "all_model_seed_differences_positive": all(
            value > 0 for value in moe_auprc["values"]
        ),
    }
    gate["performance_pass"] = (
        gate["observed_mean_auprc_gain"] >= gate["required_mean_auprc_gain"]
        and gate["all_model_seed_differences_positive"]
    )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(
            {"runs": rows, "summary": summary, "paired": paired, "moe_gate": gate},
            indent=2,
        )
        + "\n"
    )

    lines = [
        "# Modena architecture screening",
        "",
        "One generated dataset (data seed 101), three model seeds, 30 epochs.",
        "All architectures are re-scored on shared endpoints (timestep >= 5); "
        "thresholds are recalibrated on the corresponding validation endpoints.",
        "These are screening results, not confirmatory confidence intervals.",
        "",
        "| Architecture | Parameters | F1 mean +/- SD | AUPRC mean +/- SD | AUROC mean +/- SD |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['architecture']} | {row['n_params']:,} | "
            f"{row['f1_mean']:.4f} +/- {row['f1_sd']:.4f} | "
            f"{row['auprc_mean']:.4f} +/- {row['auprc_sd']:.4f} | "
            f"{row['auroc_mean']:.4f} +/- {row['auroc_sd']:.4f} |"
        )
    lines.extend(["", "## Paired differences", ""])
    for label, values in paired.items():
        lines.append(
            f"- `{label}`: F1 {values['f1']['mean']:+.4f}, "
            f"AUPRC {values['auprc']['mean']:+.4f}, "
            f"AUROC {values['auroc']['mean']:+.4f}."
        )
    lines.extend(
        [
            "",
            "## MoE performance gate",
            "",
            f"- Required mean paired AUPRC gain: `+{gate['required_mean_auprc_gain']:.2f}`.",
            f"- Observed gain: `{gate['observed_mean_auprc_gain']:+.4f}`.",
            f"- Every model-seed difference positive: `{gate['all_model_seed_differences_positive']}`.",
            f"- Performance gate: `{'PASS' if gate['performance_pass'] else 'FAIL'}`. "
            "Expert-specialisation stability is assessed separately before retention.",
        ]
    )
    if all("event_detection_rate_mean" in row for row in summary):
        lines.extend(
            [
                "",
                "## Coherent-episode detection",
                "",
                "| Architecture | Event detection rate mean +/- SD | Clean-endpoint FPR | Mean median delay |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in summary:
            delay = row["event_delay_mean"]
            lines.append(
                f"| {row['architecture']} | "
                f"{row['event_detection_rate_mean']:.4f} +/- "
                f"{row['event_detection_rate_sd']:.4f} | "
                f"{row['event_false_alarm_rate_mean']:.4f} +/- "
                f"{row['event_false_alarm_rate_sd']:.4f} | "
                f"{('--' if delay is None else f'{delay:.2f}')} |"
            )

        lines.extend(
            [
                "",
                "## Recall by standardised displacement",
                "",
                "Each severity slice uses the same clean-test observations as controls.",
                "",
                "| Architecture | <0.5σ | 0.5–1σ | 1–2σ | 2–4σ | >=4σ |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for name in ("spatial", "temporal", "moe"):
            group = grouped[name]
            values = []
            for severity in ("<0.5", "0.5-1", "1-2", "2-4", ">=4"):
                observed = [
                    row["severity_recall"].get(severity)
                    for row in group
                    if row["severity_recall"].get(severity) is not None
                ]
                values.append("--" if not observed else f"{statistics.mean(observed):.3f}")
            lines.append(f"| {name} | " + " | ".join(values) + " |")
    lines.append("")
    OUT_TABLE.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
