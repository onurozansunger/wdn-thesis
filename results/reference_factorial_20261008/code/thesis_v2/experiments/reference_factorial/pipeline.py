"""Isolated feature, fit and scoring stages for the reference factorial.

One condition/network per process; module configuration is not thread safe.
Existing research inputs are read only. No generator, calibration search or
reserved-test access is part of this adapter.
"""
from __future__ import annotations

import fcntl
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs/operational/reference_factorial_20261008"
ORIGINAL = ROOT / "runs/operational/early_warning_multiseed_v1"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "thesis_v2/experiments/early_warning")]

import lightgbm  # noqa: F401: load before sklearn's OpenMP runtime
import joblib
import numpy as np
from threadpoolctl import threadpool_limits

import build_feature_cache as fc
import run_campaign as rc
import protected_history_system as ph
from screen_observed_history import build_bank
from screen_router_temperature import transformed_mixture
from wdn.run_expert_redesign import CampaignData

CONDITIONS = {"e1r1": (True, True), "e1r0": (True, False),
              "e0r1": (False, True), "e0r0": (False, False)}
META = ("labels", "families", "source", "scenario", "event", "timestep", "node")
EVAL_SOURCES = {"modena": (40811, 41811, 42811, 43811, 44811, 45811),
                "ltown": (110811, 111811, 112811, 113811, 114811, 115811)}


def read(path):
    return json.loads(Path(path).read_text())


def relative(path):
    return str(Path(path).relative_to(ROOT))


def condition_root(condition):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    return RUN / condition


def stage_path(condition, network, seed=None):
    if network not in EVAL_SOURCES:
        raise ValueError(network)
    path = condition_root(condition) / f"stage_e_{network}"
    return path if seed is None else path / "seed" / str(seed)


def history_path(condition, seed):
    return condition_root(condition) / "protected_history_system_v1/seed" / str(seed)


def model_paths(condition, network, seed):
    """Control models are retained; altered conditions have isolated new fits."""
    if condition == "e1r1":
        base = ORIGINAL if seed <= 705 else ORIGINAL / "final_ten_model_seeds_v1"
        return base / f"stage_e_{network}/seed/{seed}", base / f"protected_history_system_v1/seed/{seed}"
    return stage_path(condition, network, seed), history_path(condition, seed)


def feature_manifest_path(condition, network, corpus_name):
    allowed = sum((corpus_names(network, role) for role in ("train", "calibration", "evaluation")), [])
    if corpus_name not in allowed:
        raise ValueError("Corpus is outside the declared network allowlist")
    base = ORIGINAL if condition == "e1r1" else condition_root(condition)
    return base / "features" / corpus_name / "manifest.json"


def disposable_training_caches(condition, network):
    """List only newly generated full/fold feature NPZs; retain OOF scores/models."""
    if condition == "e1r1":
        return []
    base = condition_root(condition)
    manifest = read(feature_manifest_path(condition, network, f"{network}_train"))
    paths = [ROOT / piece["cache"] for piece in manifest["pieces"]]
    paths += list((stage_path(condition, network) / "folds").glob("fold_*/features_*.npz"))
    for path in paths:
        if path.is_symlink() or not path.resolve().is_relative_to(base.resolve()) or path.suffix != ".npz":
            raise RuntimeError(f"Disposable cache is outside its new condition: {path}")
    return sorted(set(paths))


def full_anchor_path(network):
    return (ROOT / "runs/operational/seasonal_family_deployment_v2/full/reference.joblib"
            if network == "modena" else ORIGINAL / "stage_e_ltown/reference.joblib")


def corpus_names(network, role):
    if role in ("train", "calibration"):
        return [f"{network}_{role}"]
    if role == "evaluation":
        prefix = "modena_eval" if network == "modena" else "ltown_protected_history_confirmation"
        return [f"{prefix}_seed{s}" for s in EVAL_SOURCES[network]]
    raise ValueError(role)


def _require_protocol():
    path = RUN / "protocol_frozen.json"
    if not path.is_file():
        raise RuntimeError("Factorial protocol must be frozen before feature construction or fitting")
    read(path)
    return path


def _immutable_json(path, record):
    path = Path(path)
    record = json.loads(json.dumps(record))
    if path.exists():
        if read(path) != record:
            raise RuntimeError(f"Existing factorial signature differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        fc.write_json(path, record)


def _require_control_parity():
    marker = RUN / "control_parity_approved.json"
    if not marker.is_file():
        raise RuntimeError("e1r1 cache reuse requires control_parity_approved.json")
    record = read(marker)
    if (record.get("status") != "approved" or record.get("comparison") != "bitwise"
            or record.get("expected_case_count") != 13 or record.get("completed_case_count") != 13
            or len(record.get("cases", [])) != 13):
        raise RuntimeError("Control parity must approve both full and all11 fold reference cases")
    hashes = record.get("code_sha256", {})
    required = "src/wdn/models/reference_factorial.py"
    if required not in hashes:
        raise RuntimeError("Control parity lacks the reference implementation hash")
    for name, digest in hashes.items():
        path = ROOT / name
        if not path.resolve().is_relative_to(ROOT) or fc.sha(path) != digest:
            raise RuntimeError(f"Control parity code changed: {name}")
    return marker


def _require_evaluation_freeze():
    path = RUN / "evaluation_frozen.json"
    if not path.is_file():
        raise RuntimeError("All24 calibration selections must be frozen before evaluation")
    read(path)
    return path


def configure(condition, network):
    """Redirect imported fit helpers to one isolated condition; perform no writes."""
    target = condition_root(condition)
    stage_path(condition, network)
    fc.CAMPAIGN = target
    rc.CAMPAIGN = target
    rc.out = lambda name, *parts: stage_path(condition, name).joinpath(*(str(part) for part in parts))
    ph.CAMPAIGN = target
    ph.OUT = target / "protected_history_system_v1"
    ph.STAGE = target / "stage_e_ltown/seed"
    ph.SELECTION = target / "unused_original_selection"

    def status(name, phase, **details):
        folder = stage_path(condition, name, details.get("seed"))
        folder.mkdir(parents=True, exist_ok=True)
        record = {"condition": condition, "network": name, "phase": phase,
                  "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **details}
        fc.write_json(folder / "status.json", record)
        print(json.dumps(record), flush=True)

    rc.status = status
    # Build definitions exclusively from the existing allowlisted manifests.
    for role in ("train", "calibration", "evaluation"):
        for name in corpus_names(network, role):
            old = read(ORIGINAL / "features" / name / "manifest.json")
            pieces = []
            for piece in old["pieces"]:
                ids = list(piece["scenarios"])
                if piece["directory"] == "operational_modena_seed811":
                    spec = f"split:{role}"
                    if ids != sorted(map(int, read(fc.SPLITS)[role])):
                        raise RuntimeError("Original split allowlist differs")
                else:
                    if ids != list(range(len(ids))):
                        raise RuntimeError("Unexpected non-contiguous corpus allowlist")
                    spec = f"all:{len(ids)}"
                pieces.append((piece["directory"], piece["seed"], spec))
            fc.CORPORA[name] = {"reference": stage_path(condition, network) / "reference.joblib",
                                "network": network, "pieces": pieces}
    return target


def _make_reference(condition, network, anchor_path, destination):
    from wdn.models.reference_factorial import ReferenceFactorial
    destination = Path(destination)
    signature = {"condition": condition, "network": network,
                 "anchor_path": relative(anchor_path), "anchor_sha256": fc.sha(anchor_path),
                 "protocol_sha256": fc.sha(_require_protocol())}
    _immutable_json(destination.with_suffix(".signature.json"), signature)
    if not destination.exists():
        anchor = joblib.load(anchor_path)
        exclude, robust = CONDITIONS[condition]
        reference = ReferenceFactorial(anchor, exclude_target_group=exclude,
                                       robust_reweighting=robust)
        temporary = destination.with_suffix(".joblib.tmp")
        joblib.dump(reference, temporary)
        temporary.replace(destination)
    return destination


def _reuse_control_corpus(condition, network, name):
    marker = _require_control_parity()
    old_path = ORIGINAL / "features" / name / "manifest.json"
    old = read(old_path)
    for piece in old["pieces"]:
        path = ROOT / piece["cache"]
        if fc.sha(path) != piece["cache_sha256"]:
            raise RuntimeError(f"Original cache changed: {path}")
    result = {**old, "reference_sha256": fc.sha(stage_path(condition, network) / "reference.joblib"),
              "cache_reuse": "original e1r1 cache after explicit parity approval",
              "original_manifest": relative(old_path), "original_manifest_sha256": fc.sha(old_path),
              "control_parity_sha256": fc.sha(marker)}
    result.pop("elapsed_seconds", None)
    destination = condition_root(condition) / "features" / name / "manifest.json"
    _immutable_json(destination, result)
    return result


def build_features(condition, network, role):
    """Build the allowlisted role; orchestration gates evaluation separately."""
    _require_protocol()
    configure(condition, network)
    if role not in ("train", "calibration", "evaluation"):
        raise ValueError(role)
    if role == "evaluation":
        _require_evaluation_freeze()
    _make_reference(condition, network, full_anchor_path(network),
                    stage_path(condition, network) / "reference.joblib")
    results = []
    for name in corpus_names(network, role):
        if condition == "e1r1":
            results.append(_reuse_control_corpus(condition, network, name))
        else:
            # Match the numerical kernel configuration that passed control parity.
            with threadpool_limits(limits=2, user_api="blas"):
                results.append(fc.build(name, output_root=condition_root(condition) / "features"))
    return results


def build_full_and_calibration_features(condition, network):
    return {role: build_features(condition, network, role) for role in ("train", "calibration")}


def build_evaluation_features(condition, network):
    return build_features(condition, network, "evaluation")


def _fold_payload(arrays, source):
    rows = len(arrays["labels"])
    return {"X": np.asarray(arrays["X"], np.float32),
            "labels": np.asarray(arrays["labels"], np.int8),
            "families": np.asarray(arrays["families"], np.int8),
            "event": np.asarray(arrays["event"], np.int32),
            "scenario": np.asarray(arrays["scenario"], np.int64) + source * 1000,
            "source": np.full(rows, source, np.int64),
            "timestep": np.asarray(arrays["timestep"], np.int32),
            "node": np.asarray(arrays["node"], np.int32)}


def build_folds(condition, network):
    """Keep each original fold's fitted geometry and scale; rebuild its features."""
    _require_protocol()
    configure(condition, network)
    manifest_path = ORIGINAL / "features" / f"{network}_train/manifest.json"
    manifest = read(manifest_path)
    sources = [piece["seed"] for piece in manifest["pieces"]]
    base = stage_path(condition, network) / "folds"
    _immutable_json(base / "signature.json", {"condition": condition, "network": network,
        "sources": sources, "original_train_manifest_sha256": fc.sha(manifest_path),
        "protocol_sha256": fc.sha(_require_protocol()), "anchor_refit": False})
    for fold, held in enumerate(sources):
        target = base / f"fold_{fold}"
        original_fold = ORIGINAL / f"stage_e_{network}/folds/fold_{fold}"
        reference_path = _make_reference(condition, network, original_fold / "reference.joblib",
                                          target / "reference.joblib")
        if condition == "e1r1":
            marker = _require_control_parity()
            hashes = {}
            for split in ("train", "held_out"):
                source = original_fold / f"features_{split}.npz"
                destination = target / source.name
                if destination.is_symlink():
                    if destination.resolve() != source.resolve():
                        raise RuntimeError(f"Unexpected fold symlink: {destination}")
                elif destination.exists():
                    raise RuntimeError(f"Control fold destination is not a provenance symlink: {destination}")
                else:
                    destination.symlink_to(source)
                hashes[source.name] = fc.sha(source)
            _immutable_json(target / "feature_completion.json", {"cache_reuse": True,
                "original_fold": relative(original_fold), "hashes": hashes,
                "control_parity_sha256": fc.sha(marker)})
            continue
        reference = joblib.load(reference_path)
        for split, wanted in (("train", [s for s in sources if s != held]), ("held_out", [held])):
            destination = target / f"features_{split}.npz"
            sidecar = destination.with_suffix(".json")
            if destination.exists():
                if not sidecar.exists() or read(sidecar)["sha256"] != fc.sha(destination):
                    raise RuntimeError(f"Incomplete or changed factorial fold: {destination}")
                continue
            rc.status(network, "building factorial fold features", fold=fold, split=split)
            parts = []
            for piece in manifest["pieces"]:
                if piece["seed"] not in wanted:
                    continue
                data = CampaignData(ROOT / "data/thesis_v2" / piece["directory"])
                with threadpool_limits(limits=2, user_api="blas"):
                    arrays, names = rc.specialist_bank(data, piece["scenarios"], reference)
                if names != manifest["feature_names"]:
                    raise RuntimeError("Fold feature schema differs")
                parts.append(_fold_payload(arrays, piece["seed"]))
                del data, arrays
                gc.collect()
            merged = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
            fc.atomic_npz(destination, **merged)
            fc.write_json(sidecar, {"sha256": fc.sha(destination), "rows": len(merged["labels"]),
                "reference_sha256": fc.sha(reference_path), "sources": wanted})
            del parts, merged
            gc.collect()
        del reference
        gc.collect()


def _oof_checkpointed(condition, network, seed):
    folder = stage_path(condition, network, seed)
    target = folder / "oof_scores.npz"
    if target.exists():
        return
    folds = sorted((stage_path(condition, network) / "folds").glob("fold_*"))
    if len(folds) != (6 if network == "modena" else 5):
        raise RuntimeError("All original source-held folds are required")
    manifest = read(condition_root(condition) / "features" / f"{network}_train/manifest.json")
    bank = manifest["feature_names"]
    events = rc.events_for(network, f"{network}_train")
    paths = []
    for fold, directory in enumerate(folds):
        checkpoint = folder / f"oof_fold_{fold}.npz"
        paths.append(checkpoint)
        if checkpoint.exists():
            continue
        rc.status(network, "fitting source-held experts", seed=seed, fold=fold)
        with np.load(directory / "features_train.npz") as z:
            train = {key: z[key] for key in z.files}
        train["event"] = fc.globalise_events(train["event"], train["source"])
        train = rc.add_early_flag(train, events)
        experts = rc.fit_experts(train, bank, bank[:109], seed)
        head = rc.fit_delayed(train, bank, seed)
        del train
        gc.collect()
        with np.load(directory / "features_held_out.npz") as z:
            held = {key: z[key] for key in z.files}
        held["event"] = fc.globalise_events(held["event"], held["source"])
        causal, delayed = rc.apply_experts(experts, head, held)
        mixture = experts["mixture"].predict(held["X"])["mixture"]
        fc.atomic_npz(checkpoint, **{key: held[key] for key in META},
                      causal=causal, delayed=delayed, mixture=mixture)
        del held, experts, head, causal, delayed, mixture
        gc.collect()
        rc.status(network, "OOF fold checkpoint complete", seed=seed, fold=fold)
    parts = []
    for path in paths:
        with np.load(path) as z:
            parts.append({key: z[key] for key in z.files})
    fc.atomic_npz(target, **{key: np.concatenate([part[key] for part in parts]) for key in parts[0]})
    rc.status(network, "OOF complete", seed=seed)


def train(condition, network, seed):
    """Refit all downstream TRAIN components; perform no calibration or evaluation."""
    protocol = _require_protocol()
    configure(condition, network)
    if seed not in (701, 702, 703):
        raise ValueError("Only prespecified seeds 701–703 are supported")
    if condition == "e1r1":
        _require_control_parity()
        return model_metadata(condition, network, seed)
    folder = stage_path(condition, network, seed)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "worker.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _immutable_json(folder / "training_signature.json", {"condition": condition, "network": network,
            "seed": seed, "protocol_sha256": fc.sha(protocol), "adapter_sha256": fc.sha(__file__)})
        completion = folder / "training_complete.json"
        if completion.exists():
            record = read(completion)
            if record["model_hashes"] != model_metadata(condition, network, seed)["model_hashes"]:
                raise RuntimeError("Completed factorial models changed")
            return record
        started = time.monotonic()
        _oof_checkpointed(condition, network, seed)
        # Do not silently treat interrupted multi-file stages as complete.
        for output, completion in (("base_bundle.joblib", "fit_summary.json"),
                                   ("heads_bundle.joblib", "heads_summary.json")):
            if (folder / output).exists() and not (folder / completion).exists():
                raise RuntimeError(f"Interrupted fit needs explicit recovery: {folder / output}")
        rc.stage_fit(network, seed)
        rc.stage_heads(network, seed)
        if network == "ltown":
            _immutable_json(ph.OUT / "protocol_frozen.json", {"factorial_protocol_sha256": fc.sha(protocol),
                "condition": condition, "variant": "full_history", "calibration_search": False})
            ph.fit(seed)
        fc.write_json(folder / "training_complete.json", {"condition": condition, "network": network,
            "seed": seed, "seconds": time.monotonic() - started, "calibration_evaluated": False,
            "evaluation_evaluated": False, "model_hashes": model_metadata(condition, network, seed)["model_hashes"]})
        rc.status(network, "all TRAIN fits complete", seed=seed)


def original_rule(network, seed):
    base = ORIGINAL if seed <= 705 else ORIGINAL / "final_ten_model_seeds_v1"
    path = (base / f"stage_e_modena/seed/{seed}/operating_points.json" if network == "modena"
            else base / f"protected_history_system_v1/seed/{seed}/full_history_selection.json")
    record = read(path)
    rule = record["arms"]["candidate"]["rule"] if network == "modena" else record["rule"]
    return path, rule


def model_metadata(condition, network, seed):
    folder, history = model_paths(condition, network, seed)
    files = [folder / name for name in ("base_bundle.joblib", "delayed_bundle.joblib", "heads_bundle.joblib")]
    if network == "ltown":
        files.append(history / "history.joblib")
    heads = joblib.load(folder / "heads_bundle.joblib")
    path, rule = original_rule(network, seed)
    return {"model_hashes": {relative(path): fc.sha(path) for path in files},
            "original_rule_path": relative(path), "original_rule_sha256": fc.sha(path),
            "verifier_prethresholds": dict(heads["verifier"].pre_thresholds),
            "verifier_cutoffs": rule["verifier_cutoffs"],
            "general_temperature": rule.get("router_temperature", 1.),
            "general_shrinkage": rule.get("uniform_shrinkage", 0.),
            "veto_thresholds": [.10, .10], "veto_true_means": "block specialist alarm"}


def score_piece(condition, network, seed, manifest, piece):
    """Return per-source float64 scores and bool gates; no thresholds selected."""
    _require_protocol()
    name = manifest.get("corpus")
    if name not in corpus_names(network, "calibration") + corpus_names(network, "evaluation"):
        raise ValueError("Scoring accepts only declared calibration/evaluation corpora")
    if name in corpus_names(network, "evaluation"):
        _require_evaluation_freeze()
    configure(condition, network)
    cache = ROOT / piece["cache"]
    if fc.sha(cache) != piece["cache_sha256"]:
        raise RuntimeError(f"Feature cache changed: {cache}")
    with np.load(cache) as z:
        arrays = {key: z[key] for key in (*META, "X")}
    arrays["event"] = fc.globalise_events(arrays["event"], arrays["source"])
    if len(arrays["labels"]) != piece["rows"] or not np.all(arrays["source"] == piece["seed"]):
        raise RuntimeError("Source metadata mismatch")
    names = manifest["feature_names"]
    folder, history_folder = model_paths(condition, network, seed)
    experts = joblib.load(folder / "base_bundle.joblib")
    head = joblib.load(folder / "delayed_bundle.joblib")
    heads = joblib.load(folder / "heads_bundle.joblib")
    _, rule = original_rule(network, seed)
    causal, delayed = rc.apply_experts(experts, head, arrays)
    scores = rc.specialist_score_parts(causal, delayed, arrays)
    local = rc.early_local_evidence(arrays["X"], names, causal)
    early = heads["early"].predict_groups(local, arrays)
    del local, causal, delayed
    features, found = rc.verifier_features(arrays["X"], names, scores, early, arrays)
    if list(found) != list(heads["columns"]):
        raise RuntimeError("Verifier schema differs")
    keep = heads["verifier"].verify(features, scores)
    del features, early
    guard = heads["guard"].predict(rc.local_evidence(arrays["X"], names, rc.router_score_parts(scores)), arrays)
    veto = rc.specialist_veto_masks(guard["router"], guard["feedback"], .10, .10)
    del guard
    if network == "modena":
        general = experts["mixture"].predict(arrays["X"])["mixture"]
    else:
        history, _ = build_bank(arrays, names, manifest, (0.,))
        fitted = joblib.load(history_folder / "history.joblib")
        parts = []
        for start in range(0, len(arrays["labels"]), 50000):
            X = np.column_stack((arrays["X"][start:start + 50000], history[0.][start:start + 50000]))
            prediction = fitted.predict(X)
            parts.append(transformed_mixture(prediction["experts"], prediction["routing"],
                         rule["router_temperature"], rule["uniform_shrinkage"]))
        general = np.concatenate(parts)
        del history, fitted, parts
    payload = {key: arrays[key] for key in META}
    payload["general"] = np.asarray(general, np.float64)
    for column, branch in enumerate(rc.BRANCHES):
        payload[branch] = np.asarray(scores[f"final_{branch}"], np.float64)
        payload[f"verifier_{branch}"] = np.asarray(keep[branch], np.float64)
        payload[f"veto_{branch}"] = np.asarray(veto[:, column], bool)
        allowed = ~payload[f"veto_{branch}"]
        cutoff = rule["verifier_cutoffs"][branch]
        if cutoff > 0:
            allowed &= payload[f"verifier_{branch}"] >= cutoff
        payload[f"allow_{branch}"] = allowed
    for key in ("general", "drift", "noise", "verifier_drift", "verifier_noise"):
        if not np.isfinite(payload[key]).all():
            raise RuntimeError(f"Nonfinite score: {key}")
    del arrays, scores, keep, veto, general, experts, head, heads
    gc.collect()
    return payload
