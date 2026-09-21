"""Render the MoE specialisation JSON as an auditable Markdown table."""

from __future__ import annotations

import json
from pathlib import Path


V2 = Path(__file__).resolve().parents[1]
SOURCE = V2 / "outputs" / "runs" / "modena_moe_specialisation.json"
OUTPUT = V2 / "outputs" / "tables" / "modena_moe_specialisation.md"


def main() -> None:
    payload = json.loads(SOURCE.read_text())
    lines = [
        "# Modena MoE specialisation screen",
        "",
        "True-family (oracle) routing is a diagnostic ceiling, not a deployable method.",
        "",
        "| Data seed | Model seed | Soft AUPRC | Top-1 AUPRC | Oracle AUPRC | Router accuracy | Owner best |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(payload["runs"], key=lambda item: (item["data_seed"], item["model_seed"])):
        lines.append(
            f"| {row['data_seed']} | {row['model_seed']} | "
            f"{row['variants']['soft']['auprc']:.4f} | "
            f"{row['variants']['top1']['auprc']:.4f} | "
            f"{row['variants']['oracle']['auprc']:.4f} | "
            f"{row['router_accuracy']:.4f} | "
            f"{row['families_with_owner_best']}/{row['families_evaluated']} |"
        )
    summary = payload["summary"]
    lines.extend(
        [
            "",
            f"- Mean soft-mixture AUPRC: `{summary['soft_auprc_mean']:.4f}`.",
            f"- Mean oracle-minus-soft AUPRC: `{summary['oracle_minus_soft_auprc_mean']:+.4f}`.",
            f"- Mean router accuracy: `{summary['router_accuracy_mean']:.4f}`.",
            f"- Family-owning expert ranks first in `{summary['owner_best_fraction']:.0%}` "
            "of family-by-seed cases.",
            "- Owner-win counts: " + ", ".join(
                f"{family} {counts['wins']}/{counts['runs']}"
                for family, counts in summary["family_owner_best"].items()
            ) + ".",
            "",
            "Interpretation: the experts exhibit stable but incomplete specialisation. "
            "Soft mixing is materially better than router top-1, so the current gain "
            "should be described as a soft expert ensemble rather than reliable hard routing.",
            "",
        ]
    )
    OUTPUT.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
