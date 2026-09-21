"""Test whether the temporal gain shrinks after episode order is destroyed."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


V2 = Path(__file__).resolve().parents[1]
IID_ROOT = V2 / "outputs" / "runs" / "modena_iid_control"
EPISODE_ROOT = V2 / "outputs" / "runs" / "modena_screening"
OUT_JSON = V2 / "outputs" / "runs" / "modena_iid_control_summary.json"
OUT_MD = V2 / "outputs" / "tables" / "modena_iid_control.md"


def architecture(args: dict) -> str | None:
    if int(args.get("num_experts", 1)) != 1:
        return None
    return "spatial" if int(args["window_size"]) == 1 else "temporal"


def load(root: Path, tag_prefix: str) -> dict[tuple[str, int], dict]:
    rows = {}
    for directory in root.iterdir():
        args_path = directory / "args.json"
        detail_path = directory / "detailed_analysis.json"
        if not args_path.exists() or not detail_path.exists():
            continue
        args = json.loads(args_path.read_text())
        if not args.get("run_tag", "").startswith(tag_prefix):
            continue
        arch = architecture(args)
        if arch is None:
            continue
        rows[(arch, int(args["seed"]))] = json.loads(detail_path.read_text())[
            "aligned_sensor_metrics"
        ]["test"]
    return rows


def main() -> None:
    iid = load(IID_ROOT, "iid_")
    episode = load(EPISODE_ROOT, "screen_")
    expected = {(arch, seed) for arch in ("spatial", "temporal") for seed in (1, 2, 3)}
    if set(iid) != expected or not expected.issubset(episode):
        raise SystemExit(
            f"incomplete IID comparison: iid={sorted(iid)}, episode={sorted(episode)}"
        )

    paired = []
    for seed in (1, 2, 3):
        row = {"model_seed": seed}
        for metric in ("f1", "auprc", "auroc"):
            iid_gain = iid[("temporal", seed)][metric] - iid[("spatial", seed)][metric]
            episode_gain = (
                episode[("temporal", seed)][metric] - episode[("spatial", seed)][metric]
            )
            row[f"iid_{metric}_gain"] = iid_gain
            row[f"episode_{metric}_gain"] = episode_gain
            row[f"interaction_{metric}"] = episode_gain - iid_gain
        paired.append(row)

    summary = {}
    for prefix in ("iid", "episode", "interaction"):
        for metric in ("f1", "auprc", "auroc"):
            values = [row[f"{prefix}_{metric}" if prefix == "interaction" else f"{prefix}_{metric}_gain"] for row in paired]
            summary[f"{prefix}_{metric}_mean"] = statistics.mean(values)
            summary[f"{prefix}_{metric}_sd"] = statistics.stdev(values)
    OUT_JSON.write_text(json.dumps({"paired": paired, "summary": summary}, indent=2) + "\n")

    lines = [
        "# Matched IID temporal control",
        "",
        "Positive interaction means the temporal-minus-spatial gain is larger "
        "with coherent episodes than after within-scenario temporal permutation.",
        "",
        "| Model seed | Episode ΔAUPRC | IID ΔAUPRC | Interaction |",
        "|---:|---:|---:|---:|",
    ]
    for row in paired:
        lines.append(
            f"| {row['model_seed']} | {row['episode_auprc_gain']:+.4f} | "
            f"{row['iid_auprc_gain']:+.4f} | {row['interaction_auprc']:+.4f} |"
        )
    lines.extend(
        [
            "",
            f"Mean temporal gain: episode `{summary['episode_auprc_mean']:+.4f}` "
            f"versus IID `{summary['iid_auprc_mean']:+.4f}`; "
            f"interaction `{summary['interaction_auprc_mean']:+.4f}`.",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
