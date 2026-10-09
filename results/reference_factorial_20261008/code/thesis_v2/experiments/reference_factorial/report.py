"""Validate and summarize the complete prespecified reference factorial.

Run after all evaluations finish:
  python thesis_v2/experiments/reference_factorial/report.py

Writes only RUN/report: compact summary JSON and five CSV tables. Refuses any
missing cell or changed frozen dependency. No models are loaded or fitted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = ROOT / "runs/operational/reference_factorial_20261008"
sys.path.insert(0, str(HERE))
import metrics

NETWORKS = ("modena", "ltown")
SEEDS = (701, 702, 703)
CONDITIONS = metrics.CONDITIONS
POLICIES = tuple((cap, objective) for cap in metrics.CAPS for objective in metrics.OBJECTIVES)
METRIC_NAMES = (*metrics.FAMILIES.values(), "pooled_f1", "clean_fpr", "all_negative_fpr",
                "family_macro_f1", "worst_family_f1")
EFFECT_NAMES = tuple(metrics.factorial_contrasts(dict.fromkeys(CONDITIONS, 0.)))


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def located(root, name):
    name = Path(name)
    path = root / name
    if name.is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise RuntimeError(f"Artifact path escapes the project root: {name}")
    return path


def relative(root, path):
    return str(Path(path).relative_to(root))


def expected_sources(design):
    if (tuple(design["networks"]) != NETWORKS or tuple(design["model_seeds"]) != SEEDS
            or set(design["conditions"]) != set(CONDITIONS)):
        raise RuntimeError("Frozen experiment population differs from the report policy")
    policies = [design["calibration"]["primary"], *design["calibration"]["secondary"]]
    if len(policies) != 4 or {(p["clean_fpr_cap"], p["objective"]) for p in policies} != set(POLICIES):
        raise RuntimeError("Frozen calibration policies differ")
    if design["calibration"]["primary"] != {"objective": "pooled", "clean_fpr_cap": .0005}:
        raise RuntimeError("Primary outcome differs from the prespecified policy")
    sources = design["data"]["evaluation_sources"]
    if set(sources) != set(NETWORKS) or any(len(sources[n]) != 6 or len(set(sources[n])) != 6 for n in NETWORKS):
        raise RuntimeError("Exactly six evaluation sources per network are required")
    return sources


def validate_report(value):
    """Recompute every stored ratio from counts before accepting any result."""
    required = {"_overall", *metrics.FAMILIES.values()}
    if set(value) != required:
        raise RuntimeError("Metric report family scope differs")
    for name, row in value.items():
        counts = [row.get(k) for k in ("tp", "fp", "fn", "tn")]
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in counts):
            raise RuntimeError("Invalid confusion counts")
        tp, fp, fn, _ = counts
        expected = 2 * tp / max(1, 2 * tp + fp + fn)
        if row.get("f1") != expected:
            raise RuntimeError(f"Stored F1 differs from counts: {name}")
    o = value["_overall"]
    clean_n, clean_fp = o.get("clean_rows"), o.get("clean_period_fp")
    if (not isinstance(clean_n, int) or not isinstance(clean_fp, int)
            or clean_n <= 0 or not 0 <= clean_fp <= clean_n):
        raise RuntimeError("Invalid clean-period counts")
    if o.get("clean_fpr") != clean_fp / clean_n:
        raise RuntimeError("Stored clean FPR differs from counts")
    for key in ("tp", "fn"):
        if sum(value[n][key] for n in metrics.FAMILIES.values()) != o[key]:
            raise RuntimeError("Positive family counts do not partition pooled counts")
    if sum(value[n]["fp"] for n in metrics.FAMILIES.values()) + clean_fp != o["fp"]:
        raise RuntimeError("False-positive scopes do not partition pooled counts")
    if sum(value[n]["tn"] for n in metrics.FAMILIES.values()) + clean_n - clean_fp != o["tn"]:
        raise RuntimeError("True-negative scopes do not partition pooled counts")
    f = metrics.flat(value)
    for key in ("all_negative_fpr", "family_macro_f1", "worst_family_f1"):
        if o.get(key) != f[key]:
            raise RuntimeError(f"Stored nonlinear summary differs: {key}")
    if not np.isfinite(list(f.values())).all():
        raise RuntimeError("Nonfinite reported metric")
    return f


def validate_calibration(value, identity):
    if tuple(value.get(k) for k in ("condition", "network", "model_seed")) != identity:
        raise RuntimeError("Calibration identity differs")
    if value.get("reserved_test_scenarios_scored") is not False or value.get("evaluation_outcomes_read_for_selection") is not False:
        raise RuntimeError("Calibration scope declaration differs")
    thresholds = value["path_thresholds"]
    if set(thresholds) != set(metrics.BRANCHES):
        raise RuntimeError("Calibration branches differ")
    for path in thresholds.values():
        if len(path) != len(metrics.LAMBDAS) or not np.isfinite(path).all() or np.any(np.diff(path) > 0):
            raise RuntimeError("Invalid calibration threshold path")
    reports = value["path_calibration"]
    if len(reports) != len(metrics.LAMBDAS):
        raise RuntimeError("Calibration path length differs")
    for report in reports:
        validate_report(report)
    if value["selected"] != metrics.select(reports, thresholds):
        raise RuntimeError("Stored calibration selections do not implement the frozen policy")


def collect(root=ROOT, run=RUN):
    """Read only a complete, internally consistent and provenance-checked run."""
    root, run = Path(root), Path(run)
    protocol_path, freeze_path = run / "protocol_frozen.json", run / "evaluation_frozen.json"
    protocol, freeze = read(protocol_path), read(freeze_path)
    design = protocol["design"]
    sources = expected_sources(design)
    expected_cal = {(c, n, s): run / "results" / c / n / f"seed{s}" / "calibration_selection.json"
                    for c in CONDITIONS for n in NETWORKS for s in SEEDS}
    expected_eval = {(c, n, s, source): run / "results" / c / n / f"seed{s}" / f"evaluation_source{source}.json"
                     for c in CONDITIONS for n in NETWORKS for s in SEEDS for source in sources[n]}
    actual_eval = set((run / "results").glob("*/*/seed*/evaluation_source*.json"))
    if actual_eval != set(expected_eval.values()) or any(not p.is_file() for p in expected_cal.values()):
        missing = sum(not p.is_file() for p in expected_eval.values())
        raise RuntimeError(f"Incomplete or unexpected factorial cells: require 24 calibrations and 144 evaluation cells; missing {missing} evaluations")
    protocol_hash, freeze_hash = sha(protocol_path), sha(freeze_path)
    if (protocol["design_sha256"] != hashlib.sha256(canonical(design)).hexdigest()
            or read(run / "protocol_design.json") != design
            or freeze["protocol_sha256"] != protocol_hash):
        raise RuntimeError("Frozen protocol or design identity differs")
    if (freeze.get("expected_condition_model_fits") != 24 or freeze.get("expected_evaluation_cells") != 144
            or set(freeze["calibration_selection_sha256"]) != {relative(root, p) for p in expected_cal.values()}):
        raise RuntimeError("Evaluation freeze does not cover the complete factorial")
    checked = {}
    def check(name, digest):
        if name not in checked:
            checked[name] = sha(located(root, name))
        if checked[name] != digest:
            raise RuntimeError(f"Frozen artifact hash differs: {name}")
    for name, digest in protocol["code_sha256"].items():
        check(name, digest)
    for name, digest in freeze["model_sha256"].items():
        check(name, digest)
    check(relative(root, run / "control_parity_approved.json"), protocol["control_parity_sha256"])
    check(relative(root, run / "control_score_parity.json"), protocol["control_score_parity_sha256"])
    score_hashes = {}
    def check_score(record, calibration, role, source=None):
        name, digest = record["path"], record["sha256"]
        check(name, digest)
        if record.get("retained_control"):
            if protocol["input_sha256"].get(name) != digest:
                raise RuntimeError("Retained control score differs from frozen input identity")
        else:
            if (record.get("model_sha256") != calibration["model_sha256"]
                    or record.get("protocol_sha256") != protocol_hash
                    or record.get("role") != role
                    or any(record.get(k) != calibration[k] for k in ("condition", "network", "model_seed"))
                    or (source is not None and record.get("source_seed") != source)):
                raise RuntimeError("Score input is not associated with the frozen model/cell")
        score_hashes[name] = digest
    calibrations, model_union = {}, {}
    for identity, path in expected_cal.items():
        check(relative(root, path), freeze["calibration_selection_sha256"][relative(root, path)])
        value = read(path)
        validate_calibration(value, identity)
        if value["protocol_sha256"] != protocol_hash:
            raise RuntimeError("Calibration protocol hash differs")
        for name, digest in value["model_sha256"].items():
            if freeze["model_sha256"].get(name) != digest:
                raise RuntimeError("Calibration model differs from evaluation freeze")
            model_union[name] = digest
        for name, digest in value["original_rule_sha256"].items():
            if protocol["input_sha256"].get(name) != digest:
                raise RuntimeError("Original calibration allocation/cutoffs differ")
            check(name, digest)
        for record in value["score_inputs"]:
            check_score(record, value, "calibration")
        calibrations[identity] = value
    if model_union != freeze["model_sha256"]:
        raise RuntimeError("Frozen model set and calibration model set differ")
    cells, eval_hashes = [], {}
    for identity, path in expected_eval.items():
        condition, network, seed, source = identity
        value = read(path)
        if tuple(value.get(k) for k in ("condition", "network", "model_seed", "source_seed")) != identity:
            raise RuntimeError("Evaluation identity differs")
        calibration = calibrations[identity[:3]]
        if (value["protocol_sha256"] != protocol_hash or value["evaluation_freeze_sha256"] != freeze_hash
                or value["calibration_selection_sha256"] != sha(expected_cal[identity[:3]])
                or value.get("reserved_test_scenarios_scored") is not False):
            raise RuntimeError("Evaluation provenance or scope differs")
        check_score(value["score_input"], calibration, "evaluation", source)
        reports = value["path_evaluation"]
        if len(reports) != len(metrics.LAMBDAS):
            raise RuntimeError("Evaluation path length differs")
        for item in reports:
            validate_report(item)
        expected_selected = [{**{k: p[k] for k in ("cap", "objective", "lambda_index", "lambda")},
                              "metrics": reports[p["lambda_index"]]} for p in calibration["selected"]]
        if value["selected"] != expected_selected:
            raise RuntimeError("Evaluation selection differs from the frozen calibration path")
        for point in value["selected"]:
            cells.append({"condition": condition, "network": network, "model_seed": seed,
                          "source_seed": source, "cap": point["cap"], "objective": point["objective"],
                          "lambda": point["lambda"], "lambda_index": point["lambda_index"],
                          "metrics": metrics.flat(point["metrics"]), "counts": point["metrics"]})
        eval_hashes[relative(root, path)] = sha(path)
    provenance = {"protocol_path": relative(root, protocol_path), "protocol_sha256": protocol_hash,
                  "evaluation_freeze_path": relative(root, freeze_path), "evaluation_freeze_sha256": freeze_hash,
                  "calibration_selection_sha256": freeze["calibration_selection_sha256"],
                  "model_sha256": freeze["model_sha256"], "evaluation_result_sha256": eval_hashes,
                  "score_input_sha256": score_hashes,
                  "validation": "Code, design, parity marker, models, original rules, all calibration selections, score-input files, counts and selected path indices checked. Original raw corpus files were frozen upstream and are not rehashed by this reporting step."}
    return cells, design, provenance


def descriptive(rows):
    values = np.asarray([[row[k] for k in METRIC_NAMES] for row in rows], dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise RuntimeError("Missing or nonfinite aggregation inputs")
    return {name: dict(zip(METRIC_NAMES, map(float, result))) for name, result in (
        ("mean", values.mean(0)), ("sd", values.std(0, ddof=1) if len(values) > 1 else np.zeros(values.shape[1])),
        ("minimum", values.min(0)), ("maximum", values.max(0)))}


def aggregate(cells, design):
    """Pair before averaging; never treat the 18 seed/source cells as independent."""
    sources = expected_sources(design)
    indexed = {}
    for cell in cells:
        key = tuple(cell[k] for k in ("condition", "network", "model_seed", "source_seed", "cap", "objective"))
        if key in indexed:
            raise RuntimeError("Duplicate factorial cell/policy")
        if set(cell["metrics"]) != set(METRIC_NAMES):
            raise RuntimeError("Metric schema differs")
        indexed[key] = cell["metrics"]
    expected = {(c, n, s, source, cap, objective) for c in CONDITIONS for n in NETWORKS for s in SEEDS
                for source in sources[n] for cap, objective in POLICIES}
    if set(indexed) != expected:
        raise RuntimeError("Aggregation requires all 576 condition/cell/policy records")
    groups, effects, source_rows, source_effect_rows = [], [], [], []
    for network in NETWORKS:
        for cap, objective in POLICIES:
            policy = {"network": network, "calibration_cap_rate": cap, "objective": objective,
                      "primary": cap == .0005 and objective == "pooled"}
            for condition in CONDITIONS:
                by_source = []
                for source in sources[network]:
                    rows = [indexed[condition, network, seed, source, cap, objective] for seed in SEEDS]
                    spread = descriptive(rows)
                    by_source.append({"source_seed": source, "mean": spread["mean"], "model_seed_sd": spread["sd"]})
                    source_rows.append({**policy, "condition": condition, "source_seed": source, **spread["mean"]})
                summary = descriptive([r["mean"] for r in by_source])
                groups.append({**policy, "condition": condition, "model_seeds": 3, "sources": 6, "paired_cells": 18,
                               "mean": summary["mean"], "source_sd": summary["sd"],
                               "source_minimum": summary["minimum"], "source_maximum": summary["maximum"],
                               "per_source": by_source})
            for effect in EFFECT_NAMES:
                by_source = []
                for source in sources[network]:
                    paired = []
                    for seed in SEEDS:
                        paired.append({metric: metrics.factorial_contrasts({condition: indexed[
                            condition, network, seed, source, cap, objective][metric]
                            for condition in CONDITIONS})[effect] for metric in METRIC_NAMES})
                    spread = descriptive(paired)
                    by_source.append({"source_seed": source, "mean": spread["mean"], "model_seed_sd": spread["sd"]})
                    source_effect_rows.append({**policy, "effect": effect, "source_seed": source, **spread["mean"]})
                summary = descriptive([r["mean"] for r in by_source])
                effects.append({**policy, "effect": effect, "model_seeds": 3, "sources": 6, "paired_cells": 18,
                                "mean": summary["mean"], "source_sd": summary["sd"],
                                "source_minimum": summary["minimum"], "source_maximum": summary["maximum"],
                                "per_source": by_source})
    return groups, effects, source_rows, source_effect_rows


def csv_bytes(rows):
    if not rows:
        raise RuntimeError("Refusing empty result table")
    columns = list(rows[0])
    if any(set(row) != set(columns) for row in rows):
        raise RuntimeError("Inconsistent CSV schema")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    return buffer.getvalue().encode()


def immutable_bytes(path, content):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"Existing report differs; refusing overwrite: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(content)
    temporary.replace(path)


def generate(root=ROOT, run=RUN, output=None):
    root, run = Path(root), Path(run)
    output = Path(output) if output is not None else run / "report"
    cells, design, provenance = collect(root, run)
    groups, effects, sources, source_effects = aggregate(cells, design)
    cell_rows = []
    for cell in cells:
        row = {k: cell[k] for k in ("condition", "network", "model_seed", "source_seed", "objective", "lambda", "lambda_index")}
        row.update(calibration_cap_rate=cell["cap"], primary=cell["cap"] == .0005 and cell["objective"] == "pooled")
        row.update(cell["metrics"])
        for family, counts in cell["counts"].items():
            for key in ("tp", "fp", "fn", "tn"):
                row[f"{family}_{key}"] = counts[key]
        row.update(clean_rows=cell["counts"]["_overall"]["clean_rows"],
                   clean_period_fp=cell["counts"]["_overall"]["clean_period_fp"])
        cell_rows.append(row)
    def summary_rows(items, identity):
        out = []
        for item in items:
            row = {k: item[k] for k in ("network", identity, "calibration_cap_rate", "objective", "primary")}
            for name in ("mean", "source_sd", "source_minimum", "source_maximum"):
                row.update({f"{name}_{metric}": item[name][metric] for metric in METRIC_NAMES})
            out.append(row)
        return out
    tables = {"cells.csv": cell_rows, "source_metrics.csv": sources,
              "condition_summary.csv": summary_rows(groups, "condition"),
              "source_effects.csv": source_effects, "effect_summary.csv": summary_rows(effects, "effect")}
    table_manifest = {}
    for name, rows in tables.items():
        content = csv_bytes(rows)
        immutable_bytes(output / name, content)
        table_manifest[name] = {"rows": len(rows), "sha256": hashlib.sha256(content).hexdigest()}
    summary = {"status": "complete", "study": design["study"],
               "counts": {"conditions": 4, "networks": 2, "model_seeds": 3, "sources_per_network": 6,
                          "calibration_records": 24, "evaluation_cells": 144, "selected_policy_records": 576},
               "primary_policy": design["calibration"]["primary"], "secondary_policies": design["calibration"]["secondary"],
               "units": {"f1": "unitless; effect is an absolute F1 difference",
                         "fpr": "rates in [0,1]; multiply condition values by 100 for percent, and effects by 100 for percentage points",
                         "calibration_cap_rate": "calibration constraint, not achieved evaluation FPR",
                         "source_sd": "sample SD (ddof=1) of six source-level means; descriptive, not a standard error"},
               "aggregation": "Compute every metric and paired contrast within a model-seed/source cell, average three model seeds within source, then average six sources equally. Worst-family F1 is averaged after taking each cell's minimum. No claim of 18 independent datasets or p-values.",
               "contrast_definitions": design["metrics"]["contrasts"],
               "limits": design["limits"], "groups": groups, "effects": effects,
               "tables": table_manifest, "provenance": provenance}
    # The complete marker is published only after every CSV has been written.
    immutable_bytes(output / "summary.json", json.dumps(summary, indent=2, allow_nan=False).encode() + b"\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RUN / "report")
    args = parser.parse_args()
    summary = generate(output=args.output)
    print(f"COMPLETE: {summary['counts']['evaluation_cells']} evaluation cells; report at {args.output}")


if __name__ == "__main__":
    main()
