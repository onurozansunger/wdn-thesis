#!/usr/bin/env python3
"""Verify the compact E11 reference-factorial archive using the standard library.

Checks manifest bytes and provenance links, recomputes metrics from all published
confusion counts, repeats calibration selection on the published 82-point path,
and reconstructs all 576 selected cell records, source means and paired effects.
Models, score arrays and telemetry are omitted: their recorded fingerprints are
checked for consistency, not against unavailable bytes. This cannot reproduce
model training/inference, empirical score quantiles, or endpoint alignment.

Installed invocation: python results/reference_factorial_20261008/verify.py
Preparation/testing: python verify_reference_factorial.py --self-test
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import sys

BASE = "results/reference_factorial_20261008"
RUN = "runs/operational/reference_factorial_20261008"
OLD = "runs/operational/early_warning_multiseed_v1"
PROTOCOL_SHA = "474141759c5d624e56e9be85c9378ed7db0e42d94fdfdd90b145367753de56fd"
NETWORKS = ("modena", "ltown")
SEEDS = (701, 702, 703)
CONDITIONS = ("e0r0", "e0r1", "e1r0", "e1r1")
FAMILIES = ("random", "replay", "drift", "noise", "targeted")
BRANCHES = ("general", "drift", "noise")
CAPS = (.0005, .005)
OBJECTIVES = ("pooled", "balanced")
POLICIES = tuple((cap, objective) for cap in CAPS for objective in OBJECTIVES)
LAMBDAS = (0., *(10 ** (-2 + i / 20) for i in range(81)))
METRICS = (*FAMILIES, "pooled_f1", "clean_fpr", "all_negative_fpr", "family_macro_f1", "worst_family_f1")
EFFECTS = ("exclusion_when_irls_on", "exclusion_when_irls_off", "irls_when_exclusion_on",
           "irls_when_exclusion_off", "interaction", "exclusion_main_effect", "irls_main_effect")
TOL = 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def close(actual, expected, where):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f"{where}: keys differ")
        for key in expected:
            close(actual[key], expected[key], f"{where}/{key}")
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), f"{where}: lengths differ")
        for i, (a, b) in enumerate(zip(actual, expected)):
            close(a, b, f"{where}/{i}")
    elif isinstance(expected, (bool, str)) or expected is None:
        require(type(actual) is type(expected) and actual == expected, f"{where}: value differs")
    elif isinstance(expected, int):
        require(isinstance(actual, int) and not isinstance(actual, bool) and actual == expected, f"{where}: count differs")
    else:
        require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and math.isfinite(actual) and math.isfinite(expected)
                and abs(actual - expected) <= TOL, f"{where}: number differs ({actual}, {expected})")


def safe_relative(value):
    require(isinstance(value, str) and value, f"Invalid relative path: {value!r}")
    path = PurePosixPath(value)
    require(not path.is_absolute() and ".." not in path.parts and "\\" not in value,
            f"Unsafe relative path: {value!r}")
    return str(path)


def evaluation_cell_path(name):
    """Count result records, excluding identically named nested score sidecars."""
    return re.fullmatch(re.escape(RUN) +
                        r"/results/[^/]+/[^/]+/seed\d+/evaluation_source\d+\.json", name) is not None


def verify_manifest(repo):
    manifest = read(repo / BASE / "manifest.json")
    require(manifest["evidence_identifier"] == "E11", "Wrong evidence identifier")
    close(manifest["protocol_sha256"], PROTOCOL_SHA, "manifest protocol")
    require(manifest["raw_models_or_prediction_arrays_included"] is False
            and manifest["full_model_reproduction_from_this_repository_alone"] is False,
            "Archive must disclose omitted model/array bytes")
    entries = manifest["files"] + manifest.get("generated_files", [])
    require(entries, "Empty publication manifest")
    originals, published = {}, {}
    for item in entries:
        name = safe_relative(item["published_path"])
        generated_guide = item.get("original_path") is None and name == "docs/reference-factorial.md"
        require((name.startswith(BASE + "/") or generated_guide) and name not in published,
                "Duplicate or out-of-namespace publication path")
        path = repo / name
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(repo.resolve()),
                f"Missing, symbolic or escaped archive file: {name}")
        require(path.suffix.lower() not in (".npz", ".npy", ".pkl", ".pickle", ".joblib"), "Unexpected raw model/array artifact")
        require(isinstance(item["bytes"], int) and path.stat().st_size == item["bytes"], f"File size differs: {name}")
        require(re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is not None and sha(path) == item["sha256"], f"File hash differs: {name}")
        original = item.get("original_path")
        if original is not None:
            original = safe_relative(original)
            require(original not in originals, f"Duplicate original path: {original}")
            expected = BASE + ("/code/" if original.endswith(".py") else "/records/") + original
            require(name == expected, f"Original-path mapping differs: {original}")
            originals[original] = path
        published[name] = item
    return manifest, originals, published


def metric_report(value, where):
    require(set(value) == {"_overall", *FAMILIES}, f"{where}: family populations differ")
    for family, row in value.items():
        require(all(isinstance(row[k], int) and not isinstance(row[k], bool) and row[k] >= 0
                    for k in ("tp", "fp", "fn", "tn")), f"{where}: invalid confusion counts")
        close(row["f1"], 2 * row["tp"] / max(1, 2 * row["tp"] + row["fp"] + row["fn"]), f"{where}/{family}/F1")
    o = value["_overall"]
    clean_n, clean_fp = o["clean_rows"], o["clean_period_fp"]
    require(isinstance(clean_n, int) and isinstance(clean_fp, int) and not isinstance(clean_n, bool)
            and not isinstance(clean_fp, bool) and 0 <= clean_fp <= clean_n and clean_n > 0, f"{where}: invalid clean scope")
    close(o["clean_fpr"], clean_fp / clean_n, f"{where}/cleanFPR")
    for key in ("tp", "fn"):
        close(sum(value[f][key] for f in FAMILIES), o[key], f"{where}/family partition/{key}")
    close(sum(value[f]["fp"] for f in FAMILIES) + clean_fp, o["fp"], f"{where}/FP partition")
    close(sum(value[f]["tn"] for f in FAMILIES) + clean_n - clean_fp, o["tn"], f"{where}/TN partition")
    result = {**{f: value[f]["f1"] for f in FAMILIES}, "pooled_f1": o["f1"], "clean_fpr": clean_fp / clean_n,
              "all_negative_fpr": o["fp"] / max(1, o["fp"] + o["tn"]),
              "family_macro_f1": statistics.mean(value[f]["f1"] for f in FAMILIES),
              "worst_family_f1": min(value[f]["f1"] for f in FAMILIES)}
    for name in ("all_negative_fpr", "family_macro_f1", "worst_family_f1"):
        close(o[name], result[name], f"{where}/{name}")
    return result


def population(report):
    return {f: (report[f]["tp"] + report[f]["fn"], report[f]["fp"] + report[f]["tn"])
            for f in ("_overall", *FAMILIES)} | {"clean_rows": report["_overall"]["clean_rows"]}


def check_path(reports, where):
    require(len(reports) == 82, f"{where}: expected 82 path points")
    for i, report in enumerate(reports):
        metric_report(report, f"{where}/{i}")
        close(population(report), population(reports[0]), f"{where}/constant populations")
        if i:
            require(all(report[f][k] >= reports[i-1][f][k] for f in ("_overall", *FAMILIES)
                        for k in ("tp", "fp")), f"{where}: nonnested decisions")
            require(report["_overall"]["clean_period_fp"] >= reports[i-1]["_overall"]["clean_period_fp"], f"{where}: nonnested clean alarms")


def selected_calibration(cal):
    reports, thresholds = cal["path_calibration"], cal["path_thresholds"]
    result = []
    for cap, objective in POLICIES:
        eligible = [i for i, r in enumerate(reports) if r["_overall"]["clean_fpr"] <= cap]
        require(eligible, "No feasible calibration point")
        def rank(i):
            o = reports[i]["_overall"]
            a, b = (o["f1"], o["worst_family_f1"]) if objective == "pooled" else (o["worst_family_f1"], o["f1"])
            return a, b, -o["clean_fpr"], -i
        index = max(eligible, key=rank)
        result.append({"cap": cap, "objective": objective, "lambda_index": index, "lambda": LAMBDAS[index],
                       "thresholds": {b: thresholds[b][index] for b in BRANCHES}, "calibration": reports[index]})
    return result


def effects(values):
    a, b, c, d = (values[k] for k in CONDITIONS)
    return {"exclusion_when_irls_on": d-b, "exclusion_when_irls_off": c-a,
            "irls_when_exclusion_on": d-c, "irls_when_exclusion_off": b-a,
            "interaction": d-c-b+a, "exclusion_main_effect": .5*((d-b)+(c-a)),
            "irls_main_effect": .5*((d-c)+(b-a))}


def describe(rows):
    return {name: {metric: function([row[metric] for row in rows]) for metric in METRICS}
            for name, function in (("mean", statistics.mean), ("sd", statistics.stdev),
                                   ("minimum", min), ("maximum", max))}


def aggregates(cells, sources):
    indexed = {}
    for cell in cells:
        key = tuple(cell[k] for k in ("condition", "network", "model_seed", "source_seed", "cap", "objective"))
        require(key not in indexed, "Duplicate selected cell")
        indexed[key] = cell["metrics"]
    expected = {(c, n, s, source, cap, objective) for c in CONDITIONS for n in NETWORKS for s in SEEDS
                for source in sources[n] for cap, objective in POLICIES}
    require(set(indexed) == expected and len(indexed) == 576, "Missing or unexpected selected cell")
    groups, effect_groups, source_rows, effect_rows = [], [], [], []
    for network in NETWORKS:
        for cap, objective in POLICIES:
            policy = {"network": network, "calibration_cap_rate": cap, "objective": objective,
                      "primary": (cap, objective) == (.0005, "pooled")}
            for condition in CONDITIONS:
                per_source = []
                for source in sources[network]:
                    summary = describe([indexed[condition, network, s, source, cap, objective] for s in SEEDS])
                    per_source.append({"source_seed": source, "mean": summary["mean"], "model_seed_sd": summary["sd"]})
                    source_rows.append({**policy, "condition": condition, "source_seed": source, **summary["mean"]})
                summary = describe([r["mean"] for r in per_source])
                groups.append({**policy, "condition": condition, "model_seeds": 3, "sources": 6, "paired_cells": 18,
                               "mean": summary["mean"], "source_sd": summary["sd"], "source_minimum": summary["minimum"],
                               "source_maximum": summary["maximum"], "per_source": per_source})
            for effect in EFFECTS:
                per_source = []
                for source in sources[network]:
                    paired = [{m: effects({c: indexed[c, network, s, source, cap, objective][m]
                                           for c in CONDITIONS})[effect] for m in METRICS} for s in SEEDS]
                    summary = describe(paired)
                    per_source.append({"source_seed": source, "mean": summary["mean"], "model_seed_sd": summary["sd"]})
                    effect_rows.append({**policy, "effect": effect, "source_seed": source, **summary["mean"]})
                summary = describe([r["mean"] for r in per_source])
                effect_groups.append({**policy, "effect": effect, "model_seeds": 3, "sources": 6, "paired_cells": 18,
                                      "mean": summary["mean"], "source_sd": summary["sd"], "source_minimum": summary["minimum"],
                                      "source_maximum": summary["maximum"], "per_source": per_source})
    return groups, effect_groups, source_rows, effect_rows


def csv_compare(path, expected):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(set(reader.fieldnames or []) == set(expected[0]), f"{path.name}: columns differ")
        actual = list(reader)
    require(len(actual) == len(expected), f"{path.name}: row count differs")
    for index, (row, wanted) in enumerate(zip(actual, expected)):
        for key, value in wanted.items():
            item = row[key]
            if isinstance(value, bool):
                require(item == str(value), f"{path.name}/{index}/{key}: boolean differs")
            elif isinstance(value, int):
                close(int(item), value, f"{path.name}/{index}/{key}")
            elif isinstance(value, float):
                close(float(item), value, f"{path.name}/{index}/{key}")
            else:
                close(item, value, f"{path.name}/{index}/{key}")


def table_rows(cells, groups, effect_groups, source_rows, effect_rows):
    cell_rows = []
    for cell in cells:
        row = {k: cell[k] for k in ("condition", "network", "model_seed", "source_seed", "objective", "lambda", "lambda_index")}
        row.update(calibration_cap_rate=cell["cap"], primary=(cell["cap"], cell["objective"]) == (.0005, "pooled"))
        row.update(cell["metrics"])
        for family, counts in cell["counts"].items():
            for key in ("tp", "fp", "fn", "tn"):
                row[f"{family}_{key}"] = counts[key]
        row.update(clean_rows=cell["counts"]["_overall"]["clean_rows"], clean_period_fp=cell["counts"]["_overall"]["clean_period_fp"])
        cell_rows.append(row)
    def summaries(items, identity):
        rows = []
        for item in items:
            row = {k: item[k] for k in ("network", identity, "calibration_cap_rate", "objective", "primary")}
            for name in ("mean", "source_sd", "source_minimum", "source_maximum"):
                row.update({f"{name}_{m}": item[name][m] for m in METRICS})
            rows.append(row)
        return rows
    return {"cells.csv": cell_rows, "source_metrics.csv": source_rows,
            "condition_summary.csv": summaries(groups, "condition"), "source_effects.csv": effect_rows,
            "effect_summary.csv": summaries(effect_groups, "effect")}


def verify_supplemental_schema(original, protocol, fingerprint):
    record = read(original(RUN + "/supplemental_schema_inputs.json"))
    close(record["protocol_path"], RUN + "/protocol_frozen.json", "supplemental schema protocol path")
    close(record["protocol_sha256"], PROTOCOL_SHA, "supplemental schema protocol")
    require(record["status"] == "verified" and record["new_training_signatures_present"] == 0,
            "Supplemental schema verification timing differs")
    expected = {"runs/operational/blind_reference_probe_rank16/splits.json",
                "runs/operational/seasonal_family_deployment_v2/feature_names.json",
                "runs/operational/seasonal_family_deployment_v2/seasonal_feature_names.json"}
    require({v["path"] for v in record["supplemental_inputs"]} == expected
            and len(record["supplemental_inputs"]) == 3, "Supplemental schema dependency set differs")
    for item in record["supplemental_inputs"]:
        require(item["separately_listed_in_protocol_input_sha256"] is False
                and item["path"] not in protocol["input_sha256"], "Supplement must not retroactively claim a frozen input")
        fingerprint(item["path"], item["sha256"], True)
        close(original(item["path"]).stat().st_size, item["bytes"], "supplemental schema bytes")
    base = read(original("runs/operational/seasonal_family_deployment_v2/feature_names.json"))
    seasonal = read(original("runs/operational/seasonal_family_deployment_v2/seasonal_feature_names.json"))
    close([len(base), len(seasonal)], [109, 7], "base and seasonal feature counts")
    checks = record["frozen_manifest_checks"]
    require(len(checks) == 4 and {(v["network"], v["role"]) for v in checks}
            == {(n, r) for n in NETWORKS for r in ("train", "calibration")}, "Frozen manifest check set differs")
    for item in checks:
        path = f"{OLD}/features/{item['network']}_{item['role']}/manifest.json"
        close(item["path"], path, "supplemental manifest identity")
        close(item["sha256"], protocol["input_sha256"].get(path), "already-frozen manifest hash")
        fingerprint(path, item["sha256"], True)
        require(item["already_explicitly_frozen_input"] is True
                and item["ordered_base_plus_seasonal_names_equal_manifest"] is True, "Frozen schema check declaration differs")
        close([item["feature_count"], item["base_feature_count"], item["seasonal_feature_count"]], [116, 109, 7], "declared schema counts")
        close(read(original(path))["feature_names"], base + seasonal, "ordered 109+7 feature schema")
    splits = read(original("runs/operational/blind_reference_probe_rank16/splits.json"))
    checks = record["used_split_allowlist_checks"]
    require(len(checks) == 2 and {v["role"] for v in checks} == {"train", "calibration"}, "Used split check set differs")
    for item in checks:
        path = f"{OLD}/features/modena_{item['role']}/manifest.json"
        close(item["source_seed"], 811, "split source")
        close(item["manifest"], path, "split manifest")
        close(item["manifest_sha256"], protocol["input_sha256"].get(path), "split frozen manifest hash")
        pieces = [p for p in read(original(path))["pieces"] if p["seed"] == 811]
        require(len(pieces) == 1 and item["sorted_split_ids_equal_frozen_manifest_piece"] is True, "Split allowlist declaration differs")
        close(item["scenario_ids"], pieces[0]["scenarios"], "recorded scenario allowlist")
        close(sorted(map(int, splits[item["role"]])), pieces[0]["scenarios"], "actual TRAIN/calibration allowlist")
    train, cal = set(map(int, splits["train"])), set(map(int, splits["calibration"]))
    reserved = set(map(int, splits["validation"])) | set(map(int, splits["test"]))
    require(not train & cal and not (train | cal) & reserved, "TRAIN/calibration/reserved scenario overlap")
    close(record["split_separation"], {"train_calibration_disjoint": True,
          "used_train_calibration_disjoint_from_validation_and_reserved_test": True}, "split separation declaration")


def verify(repo):
    repo = Path(repo).resolve()
    manifest, originals, published = verify_manifest(repo)
    def original(name):
        require(name in originals, f"Required record is absent from the manifest: {name}")
        return originals[name]
    fingerprints, omitted = {}, {}
    def fingerprint(name, digest, required=False):
        safe_relative(name)
        require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None, f"Invalid provenance SHA: {name}")
        if name in fingerprints:
            close(digest, fingerprints[name], f"consistent fingerprint/{name}")
        fingerprints[name] = digest
        if required or name in originals:
            close(sha(original(name)), digest, f"published provenance bytes/{name}")
        else:
            omitted[name] = digest
    protocol = read(original(RUN + "/protocol_frozen.json"))
    close(sha(original(RUN + "/protocol_frozen.json")), PROTOCOL_SHA, "frozen protocol identity")
    design = protocol["design"]
    close(read(original(RUN + "/protocol_design.json")), design, "published design")
    close(hashlib.sha256(canonical(design)).hexdigest(), protocol["design_sha256"], "canonical design hash")
    close(design["networks"], list(NETWORKS), "networks")
    close(design["model_seeds"], list(SEEDS), "model seeds")
    require(set(design["conditions"]) == set(CONDITIONS), "Four factorial conditions required")
    for condition in CONDITIONS:
        close(design["conditions"][condition], {"target_group_exclusion": condition[1] == "1",
              "irls_reweighting": condition[3] == "1"}, f"factors/{condition}")
    close(design["calibration"]["primary"], {"objective": "pooled", "clean_fpr_cap": .0005}, "primary policy")
    policies = [design["calibration"]["primary"], *design["calibration"]["secondary"]]
    require(len(policies) == 4 and {(p["clean_fpr_cap"], p["objective"]) for p in policies} == set(POLICIES), "Four calibration policies required")
    require(design["training"]["new_data_generation"] is False and design["training"]["reserved_test_scenarios_scored"] is False,
            "Data-use scope differs")
    close(design["training"]["reference_feature_blas_threads"], 2, "feature BLAS threads")
    close(design["calibration"]["lambda_grid"]["unique_count"], 82, "lambda count")
    require(protocol["models_altered_conditions_fitted_before_freeze"] is False
            and protocol["reserved_test_scenarios_scored"] is False, "Protocol timing/scope differs")
    sources = design["data"]["evaluation_sources"]
    require(set(sources) == set(NETWORKS) and all(len(sources[n]) == len(set(sources[n])) == 6 for n in NETWORKS), "Six shared sources per network required")
    for name, digest in protocol["code_sha256"].items():
        require(original(name) == repo / BASE / "code" / name, f"Code snapshot mapping differs: {name}")
        fingerprint(name, digest, True)
    for name, digest in protocol["input_sha256"].items():
        fingerprint(name, digest)
    verify_supplemental_schema(original, protocol, fingerprint)
    for filename, key, expected_cases in (("control_parity_approved.json", "control_parity_sha256", 13),
                                          ("control_score_parity.json", "control_score_parity_sha256", 2)):
        name = RUN + "/" + filename
        fingerprint(name, protocol[key], True)
        marker = read(original(name))
        require(marker["status"] == "approved" and marker["comparison"] == "bitwise"
                and len(marker["cases"]) == expected_cases, f"Parity approval differs: {filename}")
        for code, digest in marker["code_sha256"].items():
            fingerprint(code, digest, True)
    freeze_name = RUN + "/evaluation_frozen.json"
    freeze = read(original(freeze_name)); freeze_hash = sha(original(freeze_name))
    close(freeze["protocol_sha256"], PROTOCOL_SHA, "evaluation freeze protocol")
    close(freeze["expected_condition_model_fits"], 24, "frozen calibration count")
    close(freeze["expected_evaluation_cells"], 144, "frozen evaluation count")
    require(freeze["new_evaluation_phase_started"] is False, "Calibration freeze timing declaration differs")
    expected_cal = {f"{RUN}/results/{c}/{n}/seed{s}/calibration_selection.json" for c in CONDITIONS for n in NETWORKS for s in SEEDS}
    expected_eval = {f"{RUN}/results/{c}/{n}/seed{s}/evaluation_source{source}.json" for c in CONDITIONS for n in NETWORKS for s in SEEDS for source in sources[n]}
    require(set(freeze["calibration_selection_sha256"]) == expected_cal, "Incomplete evaluation freeze")
    require({p for p in originals if evaluation_cell_path(p)} == expected_eval,
            "Missing or extra evaluation cells")
    for name, digest in freeze["model_sha256"].items():
        fingerprint(name, digest)
    score_hashes, model_union, evaluation_hashes, populations, cells = {}, {}, {}, {}, []
    def score_input(record, cal, role, source=None):
        name, digest = record["path"], record["sha256"]
        fingerprint(name, digest)
        score_hashes[name] = digest
        if record.get("retained_control"):
            close(protocol["input_sha256"].get(name), digest, "retained control score fingerprint")
        else:
            close(record["model_sha256"], cal["model_sha256"], "score-associated models")
            close(record["protocol_sha256"], PROTOCOL_SHA, "score-associated protocol")
            for key in ("condition", "network", "model_seed"):
                close(record[key], cal[key], "score-associated " + key)
            close(record["role"], role, "score role")
            if source is not None:
                close(record["source_seed"], source, "score source")
        sidecar_name = str(PurePosixPath(name).with_suffix(".json"))
        if sidecar_name in originals:
            sidecar = read(original(sidecar_name))
            if record.get("retained_control"):
                close(sidecar["npz_sha256"], digest, "retained score sidecar fingerprint")
                close(sidecar["network"], cal["network"], "retained score network")
                close(sidecar["model_seed"], cal["model_seed"], "retained model seed")
                close(sidecar["role"], role, "retained score role")
                if source is not None:
                    close(sidecar["source_seed"], source, "retained score source")
            else:
                close(sidecar, record, "published score sidecar")
    for condition in CONDITIONS:
        for network in NETWORKS:
            for seed in SEEDS:
                folder = f"{RUN}/results/{condition}/{network}/seed{seed}"
                cal_name = folder + "/calibration_selection.json"
                fingerprint(cal_name, freeze["calibration_selection_sha256"][cal_name], True)
                cal = read(original(cal_name))
                close([cal["condition"], cal["network"], cal["model_seed"]], [condition, network, seed], "calibration identity")
                close(cal["protocol_sha256"], PROTOCOL_SHA, "calibration protocol")
                require(cal["reserved_test_scenarios_scored"] is False and cal["evaluation_outcomes_read_for_selection"] is False, "Calibration scope differs")
                for name, digest in cal["model_sha256"].items():
                    close(freeze["model_sha256"].get(name), digest, "frozen calibration model")
                    fingerprint(name, digest)
                    model_union[name] = digest
                stage_name = f"{OLD}/stage_e_{network}/seed/{seed}/operating_points.json"
                rule_names = {stage_name}
                if network == "ltown":
                    history_name = f"{OLD}/protected_history_system_v1/seed/{seed}/full_history_selection.json"
                    rule_names.add(history_name)
                require(set(cal["original_rule_sha256"]) == rule_names, "Original rule provenance incomplete")
                for name, digest in cal["original_rule_sha256"].items():
                    close(protocol["input_sha256"].get(name), digest, "frozen original rule")
                    fingerprint(name, digest, True)
                stage = read(original(stage_name))["arms"]["candidate"]["rule"]
                budgets = dict(zip(BRANCHES, [.005 * share for share in stage["budget_shares"]]))
                rule = stage
                thresholds = {"general": stage["thresholds"]["mixture"], **{b: stage["thresholds"][b] for b in BRANCHES[1:]}}
                if network == "ltown":
                    rule = read(original(history_name))["rule"]
                    budgets["general"] = rule["mixture_clean_budget"]
                    thresholds = {"general": rule["mixture_threshold"], **rule["specialist_thresholds"]}
                close(cal["raw_budgets"], budgets, "shared raw branch budgets")
                close(cal["fixed_verifier_cutoffs"], rule["verifier_cutoffs"], "fixed verifier cutoffs")
                for record in cal["score_inputs"]:
                    score_input(record, cal, "calibration")
                require(len(cal["score_inputs"]) == (5 if network == "modena" else 4), "Calibration source count differs")
                require(set(cal["path_thresholds"]) == set(BRANCHES), "Calibration branch set differs")
                for branch, path in cal["path_thresholds"].items():
                    require(len(path) == 82 and all(math.isfinite(v) for v in path)
                            and all(a >= b for a, b in zip(path, path[1:])), "Invalid branch threshold path")
                    if condition == "e1r1":
                        close(path[41], thresholds[branch], "lambda-one retained control threshold")
                check_path(cal["path_calibration"], "calibration path")
                pop = population(cal["path_calibration"][0]); pop_key = (network, "calibration")
                if pop_key in populations:
                    close(pop, populations[pop_key], "paired calibration population")
                populations[pop_key] = pop
                close(cal["selected"], selected_calibration(cal), "calibration objective and ties")
                for source in sources[network]:
                    name = folder + f"/evaluation_source{source}.json"
                    value = read(original(name))
                    evaluation_hashes[name] = sha(original(name))
                    close([value[k] for k in ("condition", "network", "model_seed", "source_seed")], [condition, network, seed, source], "evaluation identity")
                    close(value["protocol_sha256"], PROTOCOL_SHA, "evaluation protocol")
                    close(value["evaluation_freeze_sha256"], freeze_hash, "evaluation freeze")
                    close(value["calibration_selection_sha256"], sha(original(cal_name)), "evaluation calibration link")
                    require(value["reserved_test_scenarios_scored"] is False, "Evaluation scope differs")
                    score_input(value["score_input"], cal, "evaluation", source)
                    check_path(value["path_evaluation"], "evaluation path")
                    pop = population(value["path_evaluation"][0]); pop_key = (network, source)
                    if pop_key in populations:
                        close(pop, populations[pop_key], "paired evaluation population")
                    populations[pop_key] = pop
                    expected = [{**{k: p[k] for k in ("cap", "objective", "lambda_index", "lambda")},
                                 "metrics": value["path_evaluation"][p["lambda_index"]]} for p in cal["selected"]]
                    close(value["selected"], expected, "evaluation frozen-path selection")
                    for point in value["selected"]:
                        cells.append({"condition": condition, "network": network, "model_seed": seed, "source_seed": source,
                                      **{k: point[k] for k in ("cap", "objective", "lambda", "lambda_index")},
                                      "metrics": metric_report(point["metrics"], "selected evaluation"), "counts": point["metrics"]})
    close(model_union, freeze["model_sha256"], "complete model fingerprint union")
    groups, effect_groups, source_rows, effect_rows = aggregates(cells, sources)
    summary_name = RUN + "/report/summary.json"
    summary = read(original(summary_name))
    require(summary["status"] == "complete", "Report is not complete")
    close(summary["counts"], {"conditions": 4, "networks": 2, "model_seeds": 3, "sources_per_network": 6,
          "calibration_records": 24, "evaluation_cells": 144, "selected_policy_records": 576}, "report completeness")
    close(summary["study"], design["study"], "report study")
    close(summary["primary_policy"], design["calibration"]["primary"], "report primary")
    close(summary["secondary_policies"], design["calibration"]["secondary"], "report secondaries")
    close(summary["contrast_definitions"], design["metrics"]["contrasts"], "contrast definitions")
    close(summary["limits"], design["limits"], "reported limitations")
    close(summary["groups"], groups, "source-weighted condition results")
    close(summary["effects"], effect_groups, "paired simple effects and interactions")
    provenance = summary["provenance"]
    for key, expected in (("protocol_path", RUN + "/protocol_frozen.json"), ("protocol_sha256", PROTOCOL_SHA),
                          ("evaluation_freeze_path", freeze_name), ("evaluation_freeze_sha256", freeze_hash),
                          ("calibration_selection_sha256", freeze["calibration_selection_sha256"]),
                          ("model_sha256", freeze["model_sha256"]), ("evaluation_result_sha256", evaluation_hashes),
                          ("score_input_sha256", score_hashes)):
        close(provenance[key], expected, "report provenance/" + key)
    tables = table_rows(cells, groups, effect_groups, source_rows, effect_rows)
    require(set(summary["tables"]) == set(tables), "Report table set differs")
    for name, expected in tables.items():
        path = original(RUN + "/report/" + name)
        close(summary["tables"][name], {"rows": len(expected), "sha256": sha(path)}, "table manifest/" + name)
        csv_compare(path, expected)
    return {"verified_publication_files": len(published), "calibration_records": 24, "evaluation_cells": 144,
            "selected_cell_policy_records": len(cells), "condition_summaries": len(groups), "effect_summaries": len(effect_groups),
            "omitted_artifact_fingerprints_consistent_but_bytes_not_verified": len(omitted),
            "model_inference_or_endpoint_alignment_reproduced": False}


def self_test():
    sample = RUN + "/results/e1r1/modena/seed701/evaluation_source40811.json"
    require(evaluation_cell_path(sample) and not evaluation_cell_path(
        sample.replace("/evaluation_source", "/scores/evaluation_source")),
        "Evaluation result inventory must exclude score sidecars")
    close(effects(dict(zip(CONDITIONS, (1., 3., 4., 10.)))), dict(zip(EFFECTS, (7., 3., 6., 2., 4., 5., 4.))), "hand-computed interaction")
    require(len(LAMBDAS) == 82 and LAMBDAS[41] == 1., "Lambda grid self-test")
    sources = {"modena": list(range(101, 107)), "ltown": list(range(201, 207))}
    cells = []
    for condition, base in zip(CONDITIONS, (.3, .4, .45, .6)):
        for network in NETWORKS:
            for j, source in enumerate(sources[network]):
                for k, seed in enumerate(SEEDS):
                    value = base + .01*j + .001*k + (.001*j + .002*k if condition == "e1r1" else 0.)
                    for cap, objective in POLICIES:
                        cells.append({"condition": condition, "network": network, "model_seed": seed, "source_seed": source,
                                      "cap": cap, "objective": objective, "metrics": dict.fromkeys(METRICS, value)})
    groups, effect_groups, source_rows, effect_rows = aggregates(cells, sources)
    close([len(groups), len(effect_groups), len(source_rows), len(effect_rows)], [32, 56, 192, 336], "aggregate dimensions")
    result = next(e for e in effect_groups if e["network"] == "modena" and e["primary"] and e["effect"] == "interaction")
    close(result["mean"]["pooled_f1"], .0545, "paired source-weighted interaction")
    close(result["source_sd"]["pooled_f1"], statistics.stdev([.001*j for j in range(6)]), "descriptive source SD")
    for mutation in (cells[:-1], cells + [cells[0]]):
        try:
            aggregates(mutation, sources)
        except ValueError:
            pass
        else:
            raise AssertionError("Missing/duplicate cell accepted")
    for path in ("../outside", "/absolute", "a\\b"):
        try:
            safe_relative(path)
        except ValueError:
            pass
        else:
            raise AssertionError("Unsafe publication path accepted")
    # Identical F1/worst-F1 paths must prefer lower FPR, then lower lambda.
    reports = [{"_overall": {"f1": .7, "worst_family_f1": .5, "clean_fpr": .0001}} for _ in LAMBDAS]
    reports[0]["_overall"]["clean_fpr"] = .0002
    cal = {"path_calibration": reports, "path_thresholds": {b: [1.] * 82 for b in BRANCHES}}
    require(all(p["lambda_index"] == 1 for p in selected_calibration(cal)), "Calibration tie ordering differs")
    print("PASS: paired aggregation, source SD, interaction, complete populations, lambda grid, calibration ties and safe paths")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        result = verify(args.repo)
        print(json.dumps(result, indent=2))
        print("Verified published E11 counts, frozen path choices and paired summaries. Omitted model/array bytes and inference were not reproduced.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(f"VERIFICATION FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
