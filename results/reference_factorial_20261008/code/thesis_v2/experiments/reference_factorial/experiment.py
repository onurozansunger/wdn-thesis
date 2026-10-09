"""Paired reference-mechanism factorial with refitted downstream detectors.

All calibration selections are frozen before the evaluation phase. Existing
datasets and anchor references are immutable. Only this campaign's disposable
derived caches may be removed after all three corresponding fits are complete.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

for _name, _value in {"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                     "MKL_NUM_THREADS": "2", "VECLIB_MAXIMUM_THREADS": "2"}.items():
    os.environ[_name] = _value
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = ROOT / "runs/operational/reference_factorial_20261008"
ORIGINAL = ROOT / "runs/operational/early_warning_multiseed_v1"
E10 = ROOT / "output/supervisor_revision_20261008/analysis/hybrid_scores"
sys.path[:0] = [str(ROOT / "src"), str(HERE), str(HERE.parent / "early_warning")]

import numpy as np
import pipeline
import metrics

CONDITIONS = ("e1r1", "e1r0", "e0r0", "e0r1")
NETWORKS = ("modena", "ltown")
SEEDS = (701, 702, 703)
META = ("labels", "families", "source", "scenario", "event", "timestep", "node")
BRANCHES = ("general", "drift", "noise")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def relative(path):
    return str(Path(path).relative_to(ROOT))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write(path, value, *, frozen=True):
    path = Path(path)
    if frozen and path.exists():
        if read(path) != value:
            raise RuntimeError(f"Existing frozen record differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def sources(network, role):
    if role == "calibration":
        return [(ORIGINAL / "features" / f"{network}_calibration/manifest.json")]
    final = read(ORIGINAL / "final_ten_model_seeds_v1/final_results.json")
    prefix = "modena_eval_seed" if network == "modena" else "ltown_protected_history_confirmation_seed"
    return [ORIGINAL / "features" / f"{prefix}{source}/manifest.json"
            for source in final["evaluation_sources"][network]]


def seed_root(condition, network, seed):
    return RUN / "results" / condition / network / f"seed{seed}"


def original_rule(network, seed):
    stage = ORIGINAL / f"stage_e_{network}/seed/{seed}/operating_points.json"
    stage_rule = read(stage)["arms"]["candidate"]["rule"]
    budgets = dict(zip(BRANCHES, [.005 * x for x in stage_rule["budget_shares"]]))
    rule = stage_rule
    files = [stage]
    thresholds = {"general": rule["thresholds"]["mixture"],
                  **{b: rule["thresholds"][b] for b in ("drift", "noise")}}
    if network == "ltown":
        path = ORIGINAL / f"protected_history_system_v1/seed/{seed}/full_history_selection.json"
        rule = read(path)["rule"]
        budgets["general"] = rule["mixture_clean_budget"]
        thresholds = {"general": rule["mixture_threshold"], **rule["specialist_thresholds"]}
        files.append(path)
    return rule, budgets, thresholds, {relative(p): sha(p) for p in files}


def freeze_protocol():
    design = read(RUN / "protocol_design.json")
    if (tuple(design["networks"]) != NETWORKS or tuple(design["model_seeds"]) != SEEDS
            or set(design["conditions"]) != set(CONDITIONS)
            or design["training"]["reference_feature_blas_threads"] != 2
            or design["calibration"]["lambda_grid"]["unique_count"] != len(metrics.LAMBDAS)):
        raise RuntimeError("Protocol constants differ from implementation")
    policies = [design["calibration"]["primary"], *design["calibration"]["secondary"]]
    if {(p["clean_fpr_cap"], p["objective"]) for p in policies} != {
            (float(cap), objective) for cap in metrics.CAPS for objective in metrics.OBJECTIVES}:
        raise RuntimeError("Protocol policies differ from implementation")
    parity = read(RUN / "control_parity_approved.json")
    if parity.get("status") != "approved":
        raise RuntimeError("The control parity gate must pass first")
    pipeline._require_control_parity()
    score_parity_path = RUN / "control_score_parity.json"
    score_parity = read(score_parity_path)
    if (score_parity.get("status") != "approved" or score_parity.get("comparison") != "bitwise"
            or len(score_parity.get("cases", [])) != 2):
        raise RuntimeError("Both networks must pass the control scoring adapter gate")
    if "thesis_v2/experiments/reference_factorial/pipeline.py" not in score_parity.get("code_sha256", {}):
        raise RuntimeError("Control score parity lacks the scoring adapter hash")
    for name, digest in score_parity["code_sha256"].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError(f"Scoring adapter code changed since parity: {name}")
    code_files = sorted(set((ROOT / "src/wdn").rglob("*.py")) |
                        set(HERE.glob("*.py")) | set((HERE.parent / "early_warning").glob("*.py")))
    inputs = set()
    for network in NETWORKS:
        manifests = [ORIGINAL / "features" / f"{network}_train/manifest.json"]
        manifests += sources(network, "calibration") + sources(network, "evaluation")
        for manifest in manifests:
            inputs.add(manifest)
            for piece in read(manifest)["pieces"]:
                directory = ROOT / "data/thesis_v2" / piece["directory"]
                inputs.update(directory / name for name in ("corrupted.pkl", "snapshots.pkl", "events.json", "generate_config.yaml"))
        anchor = (ROOT / "runs/operational/seasonal_family_deployment_v2/full/reference.joblib"
                  if network == "modena" else ORIGINAL / "stage_e_ltown/reference.joblib")
        inputs.add(anchor)
        inputs.update((ORIGINAL / f"stage_e_{network}/folds").glob("fold_*/reference.joblib"))
        for seed in SEEDS:
            _, _, _, rules = original_rule(network, seed)
            inputs.update(ROOT / p for p in rules)
            inputs.update(ORIGINAL / f"stage_e_{network}/seed/{seed}" / name
                          for name in ("base_bundle.joblib", "delayed_bundle.joblib", "heads_bundle.joblib"))
            if network == "ltown":
                inputs.add(ORIGINAL / f"protected_history_system_v1/seed/{seed}/history.joblib")
            for role in ("calibration", "evaluation"):
                for manifest in sources(network, role):
                    for piece in read(manifest)["pieces"]:
                        score = E10 / network / f"seed{seed}/{role}_source{piece['seed']}.npz"
                        meta = read(score.with_suffix(".json"))
                        if sha(score) != meta["npz_sha256"]:
                            raise RuntimeError("Retained E10 control score identity differs")
                        inputs.update((score, score.with_suffix(".json")))
    from importlib.metadata import version
    record = {"design": design, "design_sha256": hashlib.sha256(canonical(design)).hexdigest(),
              "runtime": {"python": sys.version, "libraries": {name: version(name) for name in
                          ("numpy", "scipy", "scikit-learn", "lightgbm", "joblib")},
                          "reference_feature_blas_threads": 2},
              "code_sha256": {relative(p): sha(p) for p in code_files},
              "input_sha256": {relative(p): sha(p) for p in sorted(inputs)},
              "control_parity_sha256": sha(RUN / "control_parity_approved.json"),
              "control_score_parity_sha256": sha(score_parity_path),
              "reserved_test_scenarios_scored": False,
              "models_altered_conditions_fitted_before_freeze": False}
    write(RUN / "protocol_frozen.json", record)
    print("PROTOCOL FROZEN", sha(RUN / "protocol_frozen.json"), flush=True)


def require_protocol(*, verify_inputs=False):
    record = read(RUN / "protocol_frozen.json")
    if hashlib.sha256(canonical(record["design"])).hexdigest() != record["design_sha256"]:
        raise RuntimeError("Frozen design hash differs")
    if read(RUN / "protocol_design.json") != record["design"]:
        raise RuntimeError("Current design differs from the frozen protocol")
    for name, digest in record["code_sha256"].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError(f"Runtime code changed after protocol freeze: {name}")
    if verify_inputs:
        for name, digest in record["input_sha256"].items():
            if sha(ROOT / name) != digest:
                raise RuntimeError(f"Original input changed after protocol freeze: {name}")
    return sha(RUN / "protocol_frozen.json")


def require_original_input(path):
    record = read(RUN / "protocol_frozen.json")
    name = relative(path)
    if name not in record["input_sha256"] or sha(path) != record["input_sha256"][name]:
        raise RuntimeError(f"Original artifact changed or was not frozen: {name}")


def model_files(condition, network, seed):
    if condition == "e1r1":
        stage = ORIGINAL / f"stage_e_{network}/seed/{seed}"
        history = ORIGINAL / f"protected_history_system_v1/seed/{seed}/history.joblib"
    else:
        stage, history_directory = pipeline.model_paths(condition, network, seed)
        history = history_directory / "history.joblib"
    files = [stage / name for name in ("base_bundle.joblib", "delayed_bundle.joblib", "heads_bundle.joblib")]
    if network == "ltown":
        files.append(history)
    if condition == "e1r1" and (RUN / "protocol_frozen.json").exists():
        for path in files:
            require_original_input(path)
    return {relative(p): sha(p) for p in files}


def load_scores(condition, network, seed, role, manifest, piece):
    source = int(piece["seed"])
    if condition == "e1r1":
        path = E10 / network / f"seed{seed}/{role}_source{source}.npz"
        record = read(path.with_suffix(".json"))
        require_original_input(path)
        require_original_input(path.with_suffix(".json"))
        if sha(path) != record["npz_sha256"]:
            raise RuntimeError("Retained control score file changed")
        with np.load(path, allow_pickle=False) as z:
            arrays = {key: z[key] for key in z.files}
        return arrays, {"path": relative(path), "sha256": record["npz_sha256"], "retained_control": True}
    target = seed_root(condition, network, seed) / "scores" / f"{role}_source{source}.npz"
    if target.exists():
        record = read(target.with_suffix(".json"))
        if sha(target) != record["sha256"] or record["model_sha256"] != model_files(condition, network, seed):
            raise RuntimeError("New score cache or fitted models changed")
        identity = (record.get("condition"), record.get("network"), record.get("model_seed"),
                    record.get("role"), record.get("source_seed"), record.get("protocol_sha256"))
        if identity != (condition, network, seed, role, source, sha(RUN / "protocol_frozen.json")):
            raise RuntimeError("New score cache provenance differs")
        with np.load(target, allow_pickle=False) as z:
            return {key: z[key] for key in z.files}, record
    arrays = pipeline.score_piece(condition, network, seed, manifest, piece)
    # Match identical endpoints to the already-verified control before aggregation.
    control_path = E10 / network / f"seed{seed}/{role}_source{source}.npz"
    require_original_input(control_path)
    with np.load(control_path, allow_pickle=False) as z:
        for key in META:
            if not np.array_equal(arrays[key], z[key]):
                raise RuntimeError(f"Factorial/control endpoint alignment differs: {key}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".partial")
    with temp.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(target)
    record = {"path": relative(target), "sha256": sha(target), "rows": len(arrays["labels"]),
              "condition": condition, "network": network, "model_seed": seed, "role": role,
              "source_seed": source, "model_sha256": model_files(condition, network, seed),
              "control_endpoint_alignment": True, "feature_sha256": piece["cache_sha256"],
              "protocol_sha256": sha(RUN / "protocol_frozen.json"),
              "fitted_rule_metadata": pipeline.model_metadata(condition, network, seed)}
    write(target.with_suffix(".json"), record)
    return arrays, record


def condition_manifest(condition, network, original_path):
    if condition == "e1r1":
        require_original_input(original_path)
        return read(original_path)
    return read(pipeline.feature_manifest_path(condition, network, original_path.parent.name))


def validate_selection(value, condition, network, seed):
    if (value.get("condition"), value.get("network"), value.get("model_seed")) != (condition, network, seed):
        raise RuntimeError("Calibration identity differs")
    expected = {(float(cap), objective) for cap in metrics.CAPS for objective in metrics.OBJECTIVES}
    points = value.get("selected", [])
    if len(points) != len(expected) or {(p.get("cap"), p.get("objective")) for p in points} != expected:
        raise RuntimeError("Calibration must contain all four prespecified cap/objective pairs")
    paths = value.get("path_thresholds", {})
    if set(paths) != set(BRANCHES) or any(len(paths[b]) != len(metrics.LAMBDAS) for b in BRANCHES):
        raise RuntimeError("Calibration threshold path differs from the protocol")
    reports = value.get("path_calibration", [])
    if len(reports) != len(metrics.LAMBDAS):
        raise RuntimeError("Calibration report path differs from the protocol")
    for point in points:
        index = point.get("lambda_index")
        if not isinstance(index, int) or not 0 <= index < len(metrics.LAMBDAS):
            raise RuntimeError("Invalid selected lambda index")
        if point.get("lambda") != float(metrics.LAMBDAS[index]):
            raise RuntimeError("Selected lambda value differs from its index")
        if point.get("thresholds") != {b: paths[b][index] for b in BRANCHES}:
            raise RuntimeError("Selected thresholds differ from the retained path")
        if point.get("calibration") != reports[index]:
            raise RuntimeError("Selected calibration counts differ from the retained path")
        if point["calibration"]["_overall"]["clean_fpr"] > point["cap"]:
            raise RuntimeError("Selected calibration rule violates its FPR cap")


def calibrate(condition, network, seed):
    protocol_hash = require_protocol()
    destination = seed_root(condition, network, seed) / "calibration_selection.json"
    hashes = model_files(condition, network, seed)
    if destination.exists():
        old = read(destination)
        validate_selection(old, condition, network, seed)
        if old["protocol_sha256"] != protocol_hash or old["model_sha256"] != hashes:
            raise RuntimeError("Frozen calibration provenance differs")
        return
    manifest = condition_manifest(condition, network, sources(network, "calibration")[0])
    parts, inputs = [], []
    for piece in manifest["pieces"]:
        arrays, meta = load_scores(condition, network, seed, "calibration", manifest, piece)
        parts.append({k: arrays[k] for k in ("labels", "families", *BRANCHES, "allow_drift", "allow_noise")})
        inputs.append(meta)
    arrays = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    del parts
    gc.collect()
    rule, budgets, original, rule_hashes = original_rule(network, seed)
    for name in rule_hashes:
        require_original_input(ROOT / name)
    clean = arrays["families"] == 0
    thresholds = {b: metrics.quantiles(arrays[b][clean], metrics.LAMBDAS * budgets[b]) for b in BRANCHES}
    anchor = int(np.flatnonzero(metrics.LAMBDAS == 1.)[0])
    if condition == "e1r1" and any(thresholds[b][anchor] != original[b] for b in BRANCHES):
        raise RuntimeError("The control's original thresholds were not reproduced")
    reports = metrics.hybrid_path_reports(arrays, thresholds)
    if condition == "e1r1":
        retained = [read((ROOT / item["path"]).with_suffix(".json"))["original_counts"]
                    for item in inputs]
        for family in ("_overall", *metrics.FAMILIES.values()):
            keys = ("tp", "fp", "fn", "tn")
            if family == "_overall":
                keys += ("clean_rows", "clean_period_fp")
            for key in keys:
                if reports[anchor][family][key] != sum(p[family][key] for p in retained):
                    raise RuntimeError(f"Control alarm count parity failed: {family}/{key}")
    selected = metrics.select(reports, thresholds)
    record = {"condition": condition, "network": network, "model_seed": seed,
          "protocol_sha256": protocol_hash, "model_sha256": hashes, "score_inputs": inputs,
          "original_rule_sha256": rule_hashes, "raw_budgets": budgets,
          "fixed_verifier_cutoffs": rule["verifier_cutoffs"],
          "path_thresholds": {k: v.tolist() for k, v in thresholds.items()},
          "path_calibration": reports, "selected": selected,
          "reserved_test_scenarios_scored": False, "evaluation_outcomes_read_for_selection": False}
    validate_selection(record, condition, network, seed)
    write(destination, record)
    print("CALIBRATION FROZEN", condition, network, seed, flush=True)


def freeze_evaluation():
    require_protocol()
    selections, models = {}, {}
    for condition in CONDITIONS:
        for network in NETWORKS:
            for seed in SEEDS:
                path = seed_root(condition, network, seed) / "calibration_selection.json"
                value = read(path)
                validate_selection(value, condition, network, seed)
                if value["protocol_sha256"] != sha(RUN / "protocol_frozen.json"):
                    raise RuntimeError("Calibration protocol identity mismatch")
                current = model_files(condition, network, seed)
                if current != value["model_sha256"]:
                    raise RuntimeError("A model changed after calibration")
                models.update(current)
                selections[relative(path)] = sha(path)
    write(RUN / "evaluation_frozen.json", {"protocol_sha256": sha(RUN / "protocol_frozen.json"),
          "calibration_selection_sha256": selections, "model_sha256": models,
          "expected_condition_model_fits": 24, "expected_evaluation_cells": 144,
          "new_evaluation_phase_started": False})
    print("ALL 24 CALIBRATIONS FROZEN BEFORE EVALUATION", flush=True)


def evaluate(condition, network, seed):
    require_protocol()
    freeze = read(RUN / "evaluation_frozen.json")
    selection_path = seed_root(condition, network, seed) / "calibration_selection.json"
    if sha(selection_path) != freeze["calibration_selection_sha256"][relative(selection_path)]:
        raise RuntimeError("Calibration changed after evaluation freeze")
    selection = read(selection_path)
    validate_selection(selection, condition, network, seed)
    if model_files(condition, network, seed) != selection["model_sha256"]:
        raise RuntimeError("Fitted model changed after evaluation freeze")
    for path in sources(network, "evaluation"):
        manifest = condition_manifest(condition, network, path)
        for piece in manifest["pieces"]:
            target = seed_root(condition, network, seed) / f"evaluation_source{piece['seed']}.json"
            if target.exists():
                if read(target)["calibration_selection_sha256"] != sha(selection_path):
                    raise RuntimeError("Existing evaluation provenance differs")
                continue
            arrays, input_record = load_scores(condition, network, seed, "evaluation", manifest, piece)
            reports = metrics.hybrid_path_reports(arrays, selection["path_thresholds"])
            selected = [{**{k: p[k] for k in ("cap", "objective", "lambda_index", "lambda")},
                         "metrics": reports[p["lambda_index"]]} for p in selection["selected"]]
            write(target, {"condition": condition, "network": network, "model_seed": seed,
                  "source_seed": int(piece["seed"]), "protocol_sha256": sha(RUN / "protocol_frozen.json"),
                  "evaluation_freeze_sha256": sha(RUN / "evaluation_frozen.json"),
                  "calibration_selection_sha256": sha(selection_path), "score_input": input_record,
                  "selected": selected, "path_evaluation": reports,
                  "reserved_test_scenarios_scored": False})
            print("EVALUATED", condition, network, seed, piece["seed"], flush=True)
            del arrays
            gc.collect()


def cleanup_training(condition, network):
    """Remove only this campaign's generated TRAIN/fold caches, never symlinks."""
    if condition == "e1r1":
        return
    for seed in SEEDS:
        record = read(seed_root(condition, network, seed) / "calibration_selection.json")
        if record["model_sha256"] != model_files(condition, network, seed):
            raise RuntimeError("Models must be verified before disposable cache removal")
    paths = pipeline.disposable_training_caches(condition, network)
    removed = []
    for path in paths:
        path = Path(path)
        if path.is_symlink() or not path.resolve().is_relative_to((RUN / condition).resolve()) or path.suffix != ".npz":
            raise RuntimeError(f"Refusing unsafe cache removal: {path}")
        if path.exists():
            removed.append({"path": relative(path), "bytes": path.stat().st_size, "sha256": sha(path)})
            path.unlink()
    write(RUN / "cleanup" / f"{condition}_{network}.json", {"removed_disposable_training_caches": removed})


def run_child(stage, condition, network, seed=None):
    job = f"{stage}_{condition}_{network}" + (f"_{seed}" if seed is not None else "")
    path = RUN / "logs" / f"{job}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__)), "--stage", stage,
               "--condition", condition, "--network", network]
    if seed is not None:
        command += ["--seed", str(seed)]
    print("START", job, flush=True)
    with path.open("a") as log:
        subprocess.run(command, cwd=ROOT, env=os.environ, stdout=log, stderr=subprocess.STDOUT, check=True)
    print("DONE", job, flush=True)


def orchestrate():
    require_protocol()
    for condition in CONDITIONS:
        for network in NETWORKS:
            write(RUN / "status.json", {"phase": "training_and_calibration", "condition": condition,
                  "network": network, "unix_time": time.time()}, frozen=False)
            if all((seed_root(condition, network, s) / "calibration_selection.json").exists() for s in SEEDS):
                for seed in SEEDS:
                    calibrate(condition, network, seed)
                if not (RUN / "cleanup" / f"{condition}_{network}.json").exists():
                    cleanup_training(condition, network)
                continue
            if condition != "e1r1":
                run_child("prepare", condition, network)
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(run_child, "traincal", condition, network, s) for s in SEEDS]
                for future in futures:
                    future.result()
            cleanup_training(condition, network)
    freeze_evaluation()
    for condition in CONDITIONS:
        for network in NETWORKS:
            write(RUN / "status.json", {"phase": "evaluation", "condition": condition,
                  "network": network, "unix_time": time.time()}, frozen=False)
            if condition != "e1r1":
                run_child("evalfeatures", condition, network)
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(run_child, "evaluate", condition, network, s) for s in SEEDS]
                for future in futures:
                    future.result()
    subprocess.run([sys.executable, str(HERE / "report.py")], cwd=ROOT, env=os.environ, check=True)
    write(RUN / "status.json", {"phase": "evaluation_and_report_complete", "unix_time": time.time()}, frozen=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("freeze", "prepare", "traincal", "freeze-evaluation",
                        "evalfeatures", "evaluate", "all"))
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--network", choices=NETWORKS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze_protocol()
    elif args.stage == "all":
        orchestrate()
    elif args.stage == "freeze-evaluation":
        freeze_evaluation()
    else:
        require_protocol()
        pipeline.configure(args.condition, args.network)
        if args.stage == "prepare":
            require_protocol(verify_inputs=True)
            pipeline.build_full_and_calibration_features(args.condition, args.network)
            pipeline.build_folds(args.condition, args.network)
        elif args.stage == "traincal":
            if args.condition != "e1r1":
                pipeline.train(args.condition, args.network, args.seed)
            calibrate(args.condition, args.network, args.seed)
        elif args.stage == "evalfeatures":
            if not (RUN / "evaluation_frozen.json").exists():
                raise RuntimeError("All selections must be frozen before evaluation features")
            require_protocol(verify_inputs=True)
            pipeline.build_evaluation_features(args.condition, args.network)
        elif args.stage == "evaluate":
            evaluate(args.condition, args.network, args.seed)


if __name__ == "__main__":
    main()
