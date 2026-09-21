"""Generate LaTeX result-table fragments from recorded JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path


V2 = Path(__file__).resolve().parents[1]
RUNS = V2 / "outputs" / "runs"
OUT = V2 / "outputs" / "tables" / "latex"


def read(name: str):
    return json.loads((RUNS / name).read_text())


def write(name: str, lines: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path.relative_to(V2.parent)}")


def screening() -> None:
    payload = read("modena_screening_summary.json")
    labels = {"spatial": "Spatial GNN", "temporal": "GNN+GRU", "moe": "GNN+GRU+MoE"}
    rows = {row["architecture"]: row for row in payload["summary"]}
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Development-screen performance on Modena data seed 101. Values are mean $\pm$ standard deviation over three model seeds on shared endpoints.}",
        r"  \label{tab:modena-screen}",
        r"  \begin{tabular}{lrrrr}",
        r"    \toprule",
        r"    Model & Parameters & F1 & AUPRC & AUROC \\",
        r"    \midrule",
    ]
    for arch in ("spatial", "temporal", "moe"):
        row = rows[arch]
        lines.append(
            f"    {labels[arch]} & {row['n_params']:,} & "
            f"${row['f1_mean']:.3f} \\pm {row['f1_sd']:.3f}$ & "
            f"${row['auprc_mean']:.3f} \\pm {row['auprc_sd']:.3f}$ & "
            f"${row['auroc_mean']:.3f} \\pm {row['auroc_sd']:.3f}$ \\\\"
        )
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}"])
    write("screening.tex", lines)


def baselines() -> None:
    payload = read("baseline_summary.json")
    method_labels = {
        "historical_profile": "Historical profile",
        "persistence": "Persistence",
    }
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Label-free baseline performance over three data seeds. Values are mean $\pm$ standard deviation across generated datasets.}",
        r"  \label{tab:baselines}",
        r"  \begin{tabular}{llrrr}",
        r"    \toprule",
        r"    Network & Baseline & F1 & AUPRC & AUROC \\",
        r"    \midrule",
    ]
    for row in sorted(payload, key=lambda r: (r["network"], r["method"])):
        network = "L-Town" if row["network"] == "ltown" else "Modena"
        lines.append(
            f"    {network} & {method_labels[row['method']]} & "
            f"${row['f1_mean']:.3f} \\pm {row['f1_sd']:.3f}$ & "
            f"${row['auprc_mean']:.3f} \\pm {row['auprc_sd']:.3f}$ & "
            f"${row['auroc_mean']:.3f} \\pm {row['auroc_sd']:.3f}$ \\\\"
        )
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}"])
    write("baselines.tex", lines)


def controls() -> None:
    topology = read("modena_topology_summary.json")["summary"]
    iid = read("modena_iid_control_summary.json")["summary"]
    rows = [
        ("GNN $-$ topology-free", topology["f1_difference_mean"], topology["auprc_difference_mean"], topology["auroc_difference_mean"]),
        ("Temporal $-$ spatial, episodes", iid["episode_f1_mean"], iid["episode_auprc_mean"], iid["episode_auroc_mean"]),
        ("Temporal $-$ spatial, matched IID", iid["iid_f1_mean"], iid["iid_auprc_mean"], iid["iid_auroc_mean"]),
        ("Episode $-$ IID interaction", iid["interaction_f1_mean"], iid["interaction_auprc_mean"], iid["interaction_auroc_mean"]),
    ]
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Paired component controls on Modena data seed 101, averaged over three matched model seeds. Positive values favour the first condition.}",
        r"  \label{tab:component-controls}",
        r"  \begin{tabular}{lrrr}",
        r"    \toprule",
        r"    Comparison & $\Delta$F1 & $\Delta$AUPRC & $\Delta$AUROC \\",
        r"    \midrule",
    ]
    lines.extend(
        f"    {label} & {f1:+.3f} & {auprc:+.3f} & {auroc:+.3f} \\\\"
        for label, f1, auprc, auroc in rows
    )
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}"])
    write("controls.tex", lines)


def topology_confirmation() -> None:
    payloads = {
        "Modena": read("modena_topology_confirmation_summary.json"),
        "L-Town": read("ltown_topology_confirmation_summary.json"),
    }
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Topology-aware temporal minus topology-free temporal paired effects on both networks. Each row averages three matched model seeds.}",
        r"  \label{tab:topology-confirmation}",
        r"  \setlength{\tabcolsep}{9pt}",
        r"  \begin{tabular}{lrrrr}",
        r"    \toprule",
        r"    Network & Data seed & $\Delta$F1 & $\Delta$AUPRC & $\Delta$AUROC \\",
        r"    \midrule",
    ]
    for network, payload in payloads.items():
        for row in sorted(payload["data_seed_summary"], key=lambda r: r["data_seed"]):
            lines.append(
                f"    {network} & {row['data_seed']} & "
                f"{row['f1_difference_mean']:+.3f} & "
                f"{row['auprc_difference_mean']:+.3f} & "
                f"{row['auroc_difference_mean']:+.3f} \\\\"
            )
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}"])
    write("topology_confirmation.tex", lines)


def confirmation() -> None:
    payload = read("spatiotemporal_confirmation_summary.json")
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Temporal-minus-spatial paired effects. Each row averages three matched model seeds within a generated dataset.}",
        r"  \label{tab:temporal-confirmation}",
        r"  \begin{tabular}{lrrrr}",
        r"    \toprule",
        r"    Network & Data seed & $\Delta$F1 & $\Delta$AUPRC & $\Delta$AUROC \\",
        r"    \midrule",
    ]
    for row in sorted(payload["data_seed_summary"], key=lambda r: (r["network"], r["data_seed"])):
        network = "L-Town" if row["network"] == "ltown" else "Modena"
        lines.append(
            f"    {network} & {row['data_seed']} & "
            f"{row['f1_difference_mean']:+.3f} & "
            f"{row['auprc_difference_mean']:+.3f} & "
            f"{row['auroc_difference_mean']:+.3f} \\\\"
        )
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}"])
    write("temporal_confirmation.tex", lines)

    labels = {"spatial": "Spatial GNN", "temporal": "GNN+GRU"}
    absolute = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Absolute confirmation performance. Each value is the mean of three data-seed means; model seeds are nested repetitions.}",
        r"  \label{tab:absolute-confirmation}",
        r"  \begin{tabular}{llrrr}",
        r"    \toprule",
        r"    Network & Model & F1 & AUPRC & AUROC \\",
        r"    \midrule",
    ]
    order = {"ltown": 0, "modena": 1}
    for row in sorted(
        payload["architecture_network_summary"],
        key=lambda r: (order[r["network"]], r["architecture"]),
    ):
        network = "L-Town" if row["network"] == "ltown" else "Modena"
        absolute.append(
            f"    {network} & {labels[row['architecture']]} & "
            f"{row['f1_mean']:.3f} & {row['auprc_mean']:.3f} & "
            f"{row['auroc_mean']:.3f} \\\\"
        )
    absolute.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}"])
    write("absolute_confirmation.tex", absolute)


def severity() -> None:
    payload = read("modena_severity_summary.json")["summary"]
    bins = ["<0.5", "0.5-1", "1-2", "2-4", ">=4"]
    labels = {"<0.5": "$<0.5$", "0.5-1": "$0.5$--$1$", "1-2": "$1$--$2$", "2-4": "$2$--$4$", ">=4": r"$\geq4$"}
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Recall by absolute displacement in units of the training sensor's clean residual scale, averaged over three Modena data seeds.}",
        r"  \label{tab:severity-recall}",
        r"  \begin{tabular}{rrrr}",
        r"    \toprule",
        r"    Displacement & Spatial & Temporal & MoE \\",
        r"    \midrule",
    ]
    for name in bins:
        lines.append(
            f"    {labels[name]} & "
            f"{payload['spatial'][name]['recall']['mean']:.3f} & "
            f"{payload['temporal'][name]['recall']['mean']:.3f} & "
            f"{payload['moe'][name]['recall']['mean']:.3f} \\\\"
        )
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}"])
    write("severity.tex", lines)


def ltown_severity() -> None:
    payload = read("ltown_severity_summary.json")["summary"]
    bins = ["<0.5", "0.5-1", "1-2", "2-4", ">=4"]
    labels = {"<0.5": "$<0.5$", "0.5-1": "$0.5$--$1$", "1-2": "$1$--$2$", "2-4": "$2$--$4$", ">=4": r"$\geq4$"}
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Recall by standardised absolute displacement on L-Town, averaged over three data seeds.}",
        r"  \label{tab:ltown-severity-recall}",
        r"  \begin{tabular}{rrr}",
        r"    \toprule",
        r"    Displacement & Spatial & Temporal \\",
        r"    \midrule",
    ]
    for name in bins:
        lines.append(
            f"    {labels[name]} & "
            f"{payload['spatial'][name]['recall']['mean']:.3f} & "
            f"{payload['temporal'][name]['recall']['mean']:.3f} \\\\"
        )
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}"])
    write("ltown_severity.tex", lines)


def moe_confirmation() -> None:
    payload = read("modena_moe_confirmation_summary.json")
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{MoE-minus-single-expert paired effects on Modena. Each row averages three matched model seeds.}",
        r"  \label{tab:moe-confirmation}",
        r"  \begin{tabular}{rrrr}",
        r"    \toprule",
        r"    Data seed & $\Delta$F1 & $\Delta$AUPRC & $\Delta$AUROC \\",
        r"    \midrule",
    ]
    for row in sorted(payload["data_seed_summary"], key=lambda r: r["data_seed"]):
        lines.append(
            f"    {row['data_seed']} & {row['f1_difference_mean']:+.3f} & "
            f"{row['auprc_difference_mean']:+.3f} & "
            f"{row['auroc_difference_mean']:+.3f} \\\\"
        )
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}"])
    write("moe_confirmation.tex", lines)


def main() -> None:
    baselines()
    screening()
    controls()
    topology_confirmation()
    confirmation()
    severity()
    ltown_severity()
    moe_confirmation()


if __name__ == "__main__":
    main()
