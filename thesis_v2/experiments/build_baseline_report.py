"""Aggregate full-dataset label-free baselines across data seeds."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path


V2 = Path(__file__).resolve().parents[1]
SOURCE = V2 / "outputs" / "baselines"
OUT_JSON = V2 / "outputs" / "runs" / "baseline_summary.json"
OUT_TABLE = V2 / "outputs" / "tables" / "baselines.md"


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def main() -> None:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for path in sorted(SOURCE.glob("*.json")):
        result = json.loads(path.read_text())
        dataset = result["dataset"]
        if "_smoke" in dataset or "_episode_seed" not in dataset:
            continue
        network = "ltown" if "ltown" in dataset else "modena"
        for method, report in result["methods"].items():
            grouped[(network, method)].append(report["test"])

    summary = []
    for (network, method), rows in sorted(grouped.items()):
        if len(rows) != 3:
            raise SystemExit(f"expected 3 data seeds for {network}/{method}, got {len(rows)}")
        item = {"network": network, "method": method, "n_data_seeds": len(rows)}
        for metric in ("f1", "auprc", "auroc", "precision", "recall"):
            mean, sd = mean_sd([row[metric] for row in rows])
            item[f"{metric}_mean"] = mean
            item[f"{metric}_sd"] = sd
        summary.append(item)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Label-free baseline summary",
        "",
        "Three independently generated datasets per network.",
        "Evaluation is restricted to the shared timestep >= 5 endpoints.",
        "",
        "| Network | Method | F1 mean +/- SD | AUPRC mean +/- SD | AUROC mean +/- SD |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['network']} | {row['method']} | "
            f"{row['f1_mean']:.4f} +/- {row['f1_sd']:.4f} | "
            f"{row['auprc_mean']:.4f} +/- {row['auprc_sd']:.4f} | "
            f"{row['auroc_mean']:.4f} +/- {row['auroc_sd']:.4f} |"
        )
    lines.append("")
    OUT_TABLE.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
