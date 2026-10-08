"""Render the completed, frozen comparison; never creates placeholder results.

Before first actual PDF authoring, run the PDF skill's artifact-operation marker
once for two PDF outputs. Run this script only when all 120 cells are complete.
It writes only analysis/report. Scientific plots are Matplotlib vector PDFs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
OPERATING = HERE / "operating_points"
REPORT = HERE / "report"
NETWORKS = ("modena", "ltown")
NAMES = {"modena": "Modena", "ltown": "L-Town"}
SEEDS = tuple(range(701, 711))
METRICS = ("pooled_f1", "random", "replay", "drift", "noise", "targeted",
           "family_macro_f1", "worst_family_f1", "clean_fpr", "all_negative_fpr")
METRIC_LABELS = {"pooled_f1": "Pooled F1", "random": "Random F1", "replay": "Replay F1",
                 "drift": "Drift F1", "noise": "Noise F1", "targeted": "Targeted F1",
                 "family_macro_f1": "Mean family F1", "worst_family_f1": "Mean cellwise worst-family F1",
                 "clean_fpr": "Clean false-positive rate", "all_negative_fpr": "All-negative false-positive rate"}
PANEL_METRICS = ("pooled_f1", "replay", "drift", "noise", "family_macro_f1")
DIAGNOSTICS = ("original", "general_only", "general_plus_drift", "general_plus_noise", "specialists_only",
               "without_verifier", "without_router_feedback", "without_both_guards")
DIAG_LABELS = {"original": "Full frozen rule", "general_only": "General alarms only",
               "general_plus_drift": "General + Drift alarms", "general_plus_noise": "General + Noise alarms",
               "specialists_only": "Drift + Noise alarms", "without_verifier": "Verifier bypassed",
               "without_router_feedback": "External replay veto disabled", "without_both_guards": "Verifier and external replay veto disabled"}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_complete():
    summary_path = OPERATING / "complete_summary.json"
    if not summary_path.is_file():
        raise RuntimeError("complete_summary.json is absent; no report can be rendered from partial outcomes")
    summary = read(summary_path)
    protocol = read(OPERATING / "protocol.json")
    policy = protocol["policy"]
    if summary["status"] != "complete" or summary["policy_sha256"] != protocol["policy_sha256"]:
        raise RuntimeError("Summary is incomplete or its frozen policy identity differs")
    selections = {}
    inputs = {str(summary_path.name): sha(summary_path), "protocol.json": sha(OPERATING / "protocol.json")}
    import math
    for network in NETWORKS:
        result = summary["networks"][network]
        if result["status"] != "complete" or result["completed_cells"] != 60 or result["expected_cells"] != 60 or result["missing_cells"]:
            raise RuntimeError(f"{network} does not contain all 60 cells")
        sources = policy["evaluation_sources"][network]
        if len(sources) != 6 or len(set(sources)) != 6:
            raise RuntimeError("Expected six distinct shared source datasets per network")
        for seed in SEEDS:
            path = OPERATING / network / f"seed{seed}" / "calibration_selection.json"
            cal = read(path)
            if cal["policy_sha256"] != protocol["policy_sha256"]:
                raise RuntimeError(f"Calibration policy mismatch: {path}")
            selections[(network, seed)] = cal
            for source in sources:
                cell_path = path.parent / f"evaluation_source{source}.json"
                cell = read(cell_path)
                if cell["policy_sha256"] != protocol["policy_sha256"] or cell["calibration_selection_sha256"] != sha(path):
                    raise RuntimeError(f"Evaluation policy or calibration identity mismatch: {cell_path}")
                if not cell["original_counts_reproduced"] or cell["model_seed"] != seed or cell["source_seed"] != source:
                    raise RuntimeError(f"Evaluation identity/reproduction check failed: {cell_path}")
                inputs[str(cell_path.relative_to(OPERATING))] = sha(cell_path)
        for name, group in result["groups"].items():
            if len(group["by_source"]) != 6 or len(group["by_model"]) != 10:
                raise RuntimeError(f"Descriptive aggregation is incomplete: {network}/{name}")
            for metric in METRICS:
                value = group["mean"][metric]
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise RuntimeError(f"Invalid mean metric: {network}/{name}/{metric}")
    return summary, protocol, selections, inputs


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"Refusing empty final table: {path}")
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise RuntimeError("Table records have inconsistent schemas")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def group_values(row, group):
    for metric in METRICS:
        row[f"mean_{metric}"] = group["mean"][metric]
        row[f"model_mean_sd_{metric}"] = group["model_mean_sd"][metric]
        row[f"source_mean_sd_{metric}"] = group["source_mean_sd"][metric]
    row["mean_evaluation_clean_fpr_percent"] = 100*group["mean"]["clean_fpr"]
    return row


def tables(summary, protocol, selections):
    caps = protocol["policy"]["calibration_clean_fpr_caps"]
    objectives = tuple(protocol["policy"]["objectives"])
    selected_rows = []; paired_rows = []; diagnostics_rows = []; original_rows = []; curve_rows = []
    text_table = ["MATCHED CALIBRATION-CAP COMPARISON: PRIMARY 3-HOUR LIGHTGBM", "",
                  "Both systems use the same calibration cap and objective. Evaluation FPR is observed, not forced to match.",
                  "All values are equal-weight means of 60 model/source cells per network. FPR columns are percentages.",
                  "F1 differences are hybrid minus LightGBM; a positive FPR difference means more false alarms.", ""]
    for network in NETWORKS:
        groups = summary["networks"][network]["groups"]
        for method, key in [("hybrid", "hybrid_original"),
                            ("lightgbm_3h_pooled", "lightgbm_delta3_original_pooled"),
                            ("lightgbm_3h_balanced", "lightgbm_delta3_original_balanced"),
                            ("lightgbm_0h_pooled", "lightgbm_delta0_original_pooled"),
                            ("lightgbm_0h_balanced", "lightgbm_delta0_original_balanced")]:
            original_rows.append(group_values({"network": network, "original_operating_point": method}, groups[key]))
        for objective in objectives:
            text_table += [f"{NAMES[network]} / {objective} calibration objective", 
                " Cal cap% | Eval FPR% H/LGB | Pooled F1 H/LGB (diff) | Replay H/LGB (diff) | Drift H/LGB (diff) | Noise H/LGB (diff)"]
            for cap in caps:
                hg = groups[f"hybrid_{objective}_cap{cap:g}"]
                for method, key, delta in [("hybrid", f"hybrid_{objective}_cap{cap:g}", 3),
                                           ("lightgbm", f"lightgbm_delta3_{objective}_cap{cap:g}", 3),
                                           ("lightgbm", f"lightgbm_delta0_{objective}_cap{cap:g}", 0)]:
                    cal_fprs = []
                    for seed in SEEDS:
                        fit = selections[(network, seed)]
                        choices = fit["hybrid"]["selected"] if method == "hybrid" else fit["baseline"][f"delta{delta}"]["selected"]
                        point = next(p for p in choices if p["objective"] == objective and p["cap"] == cap)
                        cal_fprs.append(point["calibration"]["_overall"]["clean_fpr"])
                    row = {"network": network, "method": method, "maximum_decision_delay_hours": delta,
                           "calibration_objective": objective, "calibration_clean_fpr_cap": cap,
                           "calibration_clean_fpr_cap_percent": cap*100,
                           "mean_achieved_calibration_clean_fpr": sum(cal_fprs)/len(cal_fprs),
                           "mean_achieved_calibration_clean_fpr_percent": 100*sum(cal_fprs)/len(cal_fprs)}
                    selected_rows.append(group_values(row, groups[key]))
                for delta in (0, 3):
                    paired = summary["networks"][network]["paired_comparisons"][f"hybrid_minus_delta{delta}_{objective}_cap{cap:g}"]
                    for metric in METRICS:
                        row = {"network": network, "lightgbm_delay_hours": delta, "objective": objective,
                               "calibration_clean_fpr_cap": cap, "metric": metric,
                               "mean_hybrid_minus_lightgbm": paired["mean_difference"][metric],
                               "higher_cells": paired["higher_cell_count"][metric],
                               "equal_cells": paired["equal_cell_count"][metric],
                               "lower_cells": 60-paired["higher_cell_count"][metric]-paired["equal_cell_count"][metric]}
                        for source in protocol["policy"]["evaluation_sources"][network]:
                            row[f"source_{source}_mean_difference"] = paired["by_source"][str(source)][metric]
                        # Network-specific source column names are normalized below.
                        row["source_mean_differences_json"] = json.dumps({s: v[metric] for s, v in paired["by_source"].items()}, sort_keys=True)
                        for k in list(row):
                            if k.startswith("source_") and k.endswith("_mean_difference"):
                                del row[k]
                        row["model_mean_differences_json"] = json.dumps({s: v[metric] for s, v in paired["by_model"].items()}, sort_keys=True)
                        paired_rows.append(row)
                h = hg["mean"]; b = groups[f"lightgbm_delta3_{objective}_cap{cap:g}"]["mean"]
                cells = [f"{cap*100:8.4f}", f"{h['clean_fpr']*100:.4f}/{b['clean_fpr']*100:.4f}"]
                cells += [f"{h[k]:.4f}/{b[k]:.4f} ({h[k]-b[k]:+.4f})" for k in ("pooled_f1", "replay", "drift", "noise")]
                text_table.append(" | ".join(cells))
            text_table.append("")
        full = groups["diagnostic_original"]["mean"]
        for diagnostic in DIAGNOSTICS:
            group = groups[f"diagnostic_{diagnostic}"]
            row = group_values({"network": network, "diagnostic": diagnostic, "description": DIAG_LABELS[diagnostic]}, group)
            for metric in METRICS: row[f"difference_from_full_{metric}"] = group["mean"][metric]-full[metric]
            diagnostics_rows.append(row)
        for method, prefix in [("hybrid", "hybrid"), ("lightgbm_3h", "lightgbm_delta3"), ("lightgbm_0h", "lightgbm_delta0")]:
            for i, multiplier in enumerate(protocol["policy"]["lambda_grid"]):
                row = {"network": network, "method": method, "lambda": multiplier,
                       "path_index": i, "thresholds_selected_on": "calibration"}
                curve_rows.append(group_values(row, groups[f"{prefix}_path_{i}"]))
    write_csv(REPORT / "selected_operating_points.csv", selected_rows)
    write_csv(REPORT / "paired_differences.csv", paired_rows)
    write_csv(REPORT / "frozen_component_diagnostics.csv", diagnostics_rows)
    write_csv(REPORT / "original_operating_points.csv", original_rows)
    write_csv(REPORT / "operating_point_curves.csv", curve_rows)
    (REPORT / "matched_cap_tables.txt").write_text("\n".join(text_table)+"\n")


def render_figures(summary, protocol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5,
                         "axes.titlesize": 10.5, "axes.labelsize": 9.5, "xtick.labelsize": 9.5,
                         "ytick.labelsize": 9.5, "legend.fontsize": 9.5, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white"})
    colors = {"hybrid": "#204A87", "delta3": "#087E72", "delta0": "#808080"}
    paths = [("Hybrid: fixed gates/allocation", "hybrid", colors["hybrid"], "-", 1.8),
             ("LightGBM: 3-hour allowance", "lightgbm_delta3", colors["delta3"], "--", 1.5),
             ("LightGBM: zero hours", "lightgbm_delta0", colors["delta0"], ":", 1.35)]
    markers = [("Original hybrid", "hybrid_original", "D", colors["hybrid"], True),
               ("Original LGB3h: pooled", "lightgbm_delta3_original_pooled", "o", colors["delta3"], True),
               ("Original LGB3h: balanced", "lightgbm_delta3_original_balanced", "s", colors["delta3"], True),
               ("Original LGB0h: pooled", "lightgbm_delta0_original_pooled", "o", colors["delta0"], False),
               ("Original LGB0h: balanced", "lightgbm_delta0_original_balanced", "s", colors["delta0"], False)]
    length = len(protocol["policy"]["lambda_grid"])
    for network in NETWORKS:
        groups = summary["networks"][network]["groups"]
        # At the thesis's 396pt text width this scales by about .81, retaining
        # roughly 7.7pt labels and ticks. Five panels plus one legend cell.
        fig, axes = plt.subplots(3, 2, figsize=(6.8, 8.8))
        fig.subplots_adjust(left=.09, right=.98, bottom=.165, top=.90, wspace=.32, hspace=.58)
        maximum = max(100*groups[f"{prefix}_path_{i}"]["mean"]["clean_fpr"] for _, prefix, *_ in paths for i in range(length))
        maximum = max(maximum, max(100*groups[key]["mean"]["clean_fpr"] for _, key, *_ in markers), .01)
        for ax, metric in zip(axes.flat, PANEL_METRICS):
            for label, prefix, color, style, width in paths:
                x = np.array([100*groups[f"{prefix}_path_{i}"]["mean"]["clean_fpr"] for i in range(length)])
                y = np.array([groups[f"{prefix}_path_{i}"]["mean"][metric] for i in range(length)])
                if np.any(np.diff(x) < -1e-12):
                    raise RuntimeError(f"Nonmonotone evaluation FPR along nested path: {network}/{prefix}")
                ax.plot(x, y, color=color, linestyle=style, linewidth=width, label=label)
            for label, key, marker, color, filled in markers:
                mean = groups[key]["mean"]
                ax.plot(mean["clean_fpr"]*100, mean[metric], marker=marker, linestyle="none", markersize=5.5,
                        markerfacecolor=color if filled else "white", markeredgecolor=color,
                        markeredgewidth=.95, zorder=5, label=label)
            ax.set_xscale("symlog", linthresh=.01, linscale=.7)
            ax.set_xlim(0, maximum*1.09); ax.set_ylim(0, 1.025)
            ticks = [t for t in (0., .01, .1, 1., 10., 100.) if t <= maximum*1.09]
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.set_xlabel("Evaluation clean FPR (%)")
            ax.set_ylabel("F1"); ax.set_title(METRIC_LABELS[metric], loc="left", fontweight="bold")
            ax.grid(True, color="#D8DDE3", linewidth=.55, alpha=.75)
            ax.set_axisbelow(True)
        legend = axes[2, 1]; legend.axis("off")
        handles = [Line2D([0], [0], color=c, linestyle=s, linewidth=w, label=l) for l, _p, c, s, w in paths]
        handles += [Line2D([0], [0], linestyle="none", marker=m, markerfacecolor=c if filled else "white",
                           markeredgecolor=c, markersize=5.5, label=l) for l, _k, m, c, filled in markers]
        legend.legend(handles=handles, loc="upper left", frameon=False, borderaxespad=0., labelspacing=.23,
                      handlelength=1.9, handletextpad=.55)
        fig.suptitle(f"{NAMES[network]}: operating-point sensitivity", x=.09, y=.978, ha="left", fontsize=13, fontweight="bold")
        fig.text(.09, .935, "Calibration selects thresholds. FPR: false-positive rate; LGB: LightGBM.",
                 ha="left", fontsize=9.5, color="#404852")
        footer = ["Means: 10 models × 6 shared sources. The 60 cells are not independent.",
                  "Hybrid gates/allocation fixed. Three-hour control primary; zero-hour contextual.",
                  "FPR axis: linear to 0.01%, logarithmic above. No exhaustive frontier."]
        for y, line in zip((.105, .080, .055), footer):
            fig.text(.09, y, line, ha="left", fontsize=9.5)
        stem = REPORT / f"{network}_operating_point_sensitivity"
        fig.savefig(stem.with_suffix(".pdf"), metadata={"Title": f"{NAMES[network]} operating-point sensitivity", "Author": "Thesis supplementary analysis"})
        fig.savefig(stem.with_suffix(".png"), dpi=220)
        plt.close(fig)


def write_report(summary, protocol):
    caps = protocol["policy"]["calibration_clean_fpr_caps"]
    lines = ["SUPPLEMENTARY OPERATING-POINT ANALYSIS", "",
             "Complete: 10 existing model fits x 6 existing evaluation sources x 2 networks = 120 cells.",
             "All 20 calibration selections were frozen before evaluation analysis. No models were retrained and no data were generated.",
             "Retained original counts and reported metrics were reproduced before comparisons. These are reused simulation sources, not a new independent confirmation.",
             "For Modena, reproduction checks match the exact saved pooled confusion counts and saved rounded family F1 values; no historical per-endpoint decision vector was retained for a direct vector-equality check.",
             "For L-Town, checks match exact per-family and pooled confusion counts and the retained specialist union. Inference uses the unchanged fitted models and frozen rules.", "",
             "Interpretation: thresholds vary along a prespecified path. Hybrid branch allocation, General's internal routing, external replay veto and verifier behavior remain fixed.",
             "Clean-FPR caps constrain calibration only; achieved evaluation FPR is reported separately. They are not matched evaluation-FPR guarantees.",
             "Three-hour LightGBM is the primary timing comparison. Its zero-hour counterpart is included for context.",
             "There is no intrinsic hybrid probability/AUPRC claim or claim to search every possible hybrid threshold/gate configuration.",
             "Evaluation summaries are equal-weight averages of 60 model/source cells per network. F1 and nonlinear summaries are computed within each cell before averaging.",
             "Mean worst-family F1 is the average of the 60 cellwise minimum family F1 values, not the minimum of the displayed family means.",
             "Model-mean and source-mean standard deviations are descriptive. Shared cells are not used as independent evidence of significance.", ""]
    for network in NETWORKS:
        groups = summary["networks"][network]["groups"]
        lines += [NAMES[network].upper(), "Original operating points: pooled F1 / mean family F1 / mean cellwise worst-family F1 / clean FPR (%)."]
        for name, key in [("Hybrid", "hybrid_original"), ("LightGBM 3h pooled", "lightgbm_delta3_original_pooled"),
                          ("LightGBM 3h balanced", "lightgbm_delta3_original_balanced"),
                          ("LightGBM 0h pooled", "lightgbm_delta0_original_pooled"),
                          ("LightGBM 0h balanced", "lightgbm_delta0_original_balanced")]:
            m = groups[key]["mean"]
            lines.append(f"  {name}: {m['pooled_f1']:.4f} / {m['family_macro_f1']:.4f} / {m['worst_family_f1']:.4f} / {100*m['clean_fpr']:.4f}%")
        lines += ["", "Paired differences at all seven calibration caps (hybrid minus 3h LightGBM).",
                  "Higher/lower counts below refer to operating-point choices, not independent experiments. FPR differences favor lower values."]
        for objective in protocol["policy"]["objectives"]:
            lines.append(f"  {objective.upper()} calibration objective:")
            for metric in METRICS:
                ds = [summary["networks"][network]["paired_comparisons"][f"hybrid_minus_delta3_{objective}_cap{cap:g}"]["mean_difference"][metric] for cap in caps]
                higher = [c for c, d in zip(caps, ds) if d > 1e-12]
                lower = [c for c, d in zip(caps, ds) if d < -1e-12]
                equal = len(caps)-len(higher)-len(lower)
                unit = " percentage points" if metric.endswith("fpr") else ""
                scale = 100 if metric.endswith("fpr") else 1
                line = f"    {METRIC_LABELS[metric]}: difference range {min(ds)*scale:+.4f} to {max(ds)*scale:+.4f}{unit}; higher {len(higher)}, lower {len(lower)}, equal {equal}."
                if higher and lower:
                    hi = ", ".join(f"{100*c:g}%" for c in higher); lo = ", ".join(f"{100*c:g}%" for c in lower)
                    line += f" Direction reverses: higher at caps [{hi}], lower at [{lo}]."
                lines.append(line)
        lines += ["", "Frozen-rule component diagnostics: pooled / replay / drift / noise / clean FPR (%); changes from full rule in parentheses.",
                  "These remove alarm contributions or bypass guards while retaining fitted upstream features, scores and original thresholds.",
                  "Disabling the external replay veto does not change General's internal mixture routing. All model scores were computed for the diagnostics; the specialists-only alarm rule does not consume General's output.",
                  "Omitting a specialist's alarm contribution leaves that specialist's score channels in the shared router, feedback and verifier inputs.",
                  "They are conditional diagnostics, not retrained or recalibrated ablations; they do not isolate early-warning features from the verifier."]
        original = groups["diagnostic_original"]["mean"]
        for name in DIAGNOSTICS:
            m = groups[f"diagnostic_{name}"]["mean"]
            parts = [f"{m[k]:.4f} ({m[k]-original[k]:+.4f})" for k in ("pooled_f1", "replay", "drift", "noise")]
            parts.append(f"{100*m['clean_fpr']:.4f}% ({100*(m['clean_fpr']-original['clean_fpr']):+.4f} pp)")
            lines.append(f"  {DIAG_LABELS[name]}: " + " / ".join(parts))
        lines.append("")
    lines += ["FILES", "  Two network PDF/PNG figures: full prespecified paths with original operating-point markers.",
              "  selected_operating_points.csv: all caps, both objectives, both baseline delays, actual calibration/evaluation FPR and descriptive SDs.",
              "  paired_differences.csv: all metrics, cap/objective/delay comparisons, paired source/model means and cell direction counts.",
              "  frozen_component_diagnostics.csv: every diagnostic and every reported metric, including differences from the full rule.",
              "  original_operating_points.csv and operating_point_curves.csv: complete plotting values.",
              "  matched_cap_tables.txt: primary three-hour comparison table, with distinct calibration cap and evaluation FPR columns.", "",
              "No conclusion about target-group exclusion versus robust fitting follows from this analysis.",
              "No thesis files, model files, benchmark data or historical results were changed."]
    (REPORT / "report.txt").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Verify completeness and hashes; do not create any report files")
    args = parser.parse_args()
    summary, protocol, selections, inputs = require_complete()
    if args.check_only:
        print("READY: all 120 cells, all 20 calibration selections and complete summary verified; no files created")
        return
    REPORT.mkdir(parents=True, exist_ok=True)
    tables(summary, protocol, selections)
    write_report(summary, protocol)
    render_figures(summary, protocol)
    outputs = sorted(p for p in REPORT.iterdir() if p.is_file() and p.name != "render_manifest.json")
    manifest = {"status": "generated_requires_visual_review", "policy_sha256": protocol["policy_sha256"],
                "renderer_sha256": sha(__file__), "input_sha256": inputs,
                "output_sha256": {p.name: sha(p) for p in outputs},
                "complete_cells": 120, "new_inference": False, "training": False,
                "visual_review_completed": False}
    (REPORT / "render_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"Generated {len(outputs)} files under {REPORT}. PDF/PNG visual inspection is still required.")


if __name__ == "__main__":
    main()
