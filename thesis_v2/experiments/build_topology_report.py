"""Compare the temporal GNN with its no-message-passing control."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


V2 = Path(__file__).resolve().parents[1]
TEMPORAL_ROOT = V2 / "outputs" / "runs" / "modena_screening"
CONTROL_ROOT = V2 / "outputs" / "runs" / "modena_topology_control"
OUT_JSON = V2 / "outputs" / "runs" / "modena_topology_summary.json"
OUT_MD = V2 / "outputs" / "tables" / "modena_topology.md"


def load(root: Path, tag_fragment: str) -> dict[int, dict]:
    rows = {}
    for directory in root.iterdir():
        args_path = directory / "args.json"
        detail_path = directory / "detailed_analysis.json"
        if not args_path.exists() or not detail_path.exists():
            continue
        args = json.loads(args_path.read_text())
        if tag_fragment not in args.get("run_tag", ""):
            continue
        rows[int(args["seed"])] = json.loads(detail_path.read_text())[
            "aligned_sensor_metrics"
        ]["test"]
    return rows


def main() -> None:
    temporal = load(TEMPORAL_ROOT, "screen_temporal")
    no_topology = load(CONTROL_ROOT, "topology_none")
    if set(temporal) != {1, 2, 3} or set(no_topology) != {1, 2, 3}:
        raise SystemExit(
            f"incomplete topology comparison: temporal={sorted(temporal)}, "
            f"no_topology={sorted(no_topology)}"
        )
    paired = []
    for seed in (1, 2, 3):
        paired.append(
            {
                "model_seed": seed,
                **{
                    f"{metric}_difference": temporal[seed][metric] - no_topology[seed][metric]
                    for metric in ("f1", "auprc", "auroc")
                },
            }
        )
    summary = {}
    for metric in ("f1", "auprc", "auroc"):
        values = [row[f"{metric}_difference"] for row in paired]
        summary[f"{metric}_difference_mean"] = statistics.mean(values)
        summary[f"{metric}_difference_sd"] = statistics.stdev(values)
    OUT_JSON.write_text(json.dumps({"paired": paired, "summary": summary}, indent=2) + "\n")
    lines = [
        "# Modena topology control",
        "",
        "Temporal GNN minus the matched temporal model with message passing disabled.",
        "",
        "| Model seed | ΔF1 | ΔAUPRC | ΔAUROC |",
        "|---:|---:|---:|---:|",
    ]
    for row in paired:
        lines.append(
            f"| {row['model_seed']} | {row['f1_difference']:+.4f} | "
            f"{row['auprc_difference']:+.4f} | {row['auroc_difference']:+.4f} |"
        )
    lines.extend(
        [
            "",
            f"Mean paired gain: F1 `{summary['f1_difference_mean']:+.4f}`, "
            f"AUPRC `{summary['auprc_difference_mean']:+.4f}`, "
            f"AUROC `{summary['auroc_difference_mean']:+.4f}`.",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
