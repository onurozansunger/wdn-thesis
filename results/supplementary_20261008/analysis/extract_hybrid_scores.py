"""Recover frozen hybrid scores; never fit, calibrate, generate, or open a locked test.

The only writes are under this analysis directory. Existing source-wise input
caches remain immutable. A source is published only after its original decision
has passed the available historical count checks. Calibration is also checked
against the original pooled report after all its sources are present.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

# Set before importing any numerical library; at most two such processes.
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_key, "4")
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CAMPAIGN = ROOT / "runs/operational/early_warning_multiseed_v1"
EXT = CAMPAIGN / "final_ten_model_seeds_v1"
SCORES = HERE / "hybrid_scores"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "thesis_v2/experiments/early_warning")]

import lightgbm as lgb  # Import before sklearn's OpenMP runtime.
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
import run_campaign as rc
from build_feature_cache import globalise_events
from screen_router_temperature import transformed_mixture

FAMILIES = {1: "random", 2: "replay", 3: "drift", 4: "noise", 5: "targeted"}
META = ("labels", "families", "source", "scenario", "event", "timestep", "node")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def relative(path):
    return str(Path(path).relative_to(ROOT))


def paths(network, seed):
    base = CAMPAIGN if seed <= 705 else EXT
    stage = base / f"stage_e_{network}/seed/{seed}"
    history = base / f"protected_history_system_v1/seed/{seed}"
    return stage, history


def frozen_rule(network, seed):
    stage, history = paths(network, seed)
    path = stage / "operating_points.json" if network == "modena" else history / "full_history_selection.json"
    record = read(path)
    rule = record["arms"]["candidate"]["rule"] if network == "modena" else record["rule"]
    thresholds = ({"general": rule["thresholds"]["mixture"],
                   **{b: rule["thresholds"][b] for b in rc.BRANCHES}} if network == "modena" else
                  {"general": rule["mixture_threshold"], **rule["specialist_thresholds"]})
    return path, rule, thresholds


def source_jobs(network, role):
    """Return only the original calibration or published evaluation sources."""
    final = read(EXT / "final_results.json")
    if role == "calibration":
        names = [f"{network}_calibration"]
    elif role == "evaluation":
        names = [(f"modena_eval_seed{s}" if network == "modena" else
                  f"ltown_protected_history_confirmation_seed{s}")
                 for s in final["evaluation_sources"][network]]
    else:
        raise ValueError(role)
    jobs = []
    for name in names:
        path = CAMPAIGN / "features" / name / "manifest.json"
        manifest = read(path)
        for piece in manifest["pieces"]:
            jobs.append((path, manifest, piece))
    return jobs


def validate_model_files(network, seed):
    """Check fitted bundles/rules against already frozen provenance."""
    stage, history = paths(network, seed)
    protocol = read(EXT / "protocol_frozen.json")
    expected = dict(protocol["input_code_sha256"])
    expected.update(read(EXT / "evaluation_frozen.json")["model_rule_sha256"])
    used = [stage / n for n in ("base_bundle.joblib", "delayed_bundle.joblib", "heads_bundle.joblib")]
    used += [frozen_rule(network, seed)[0]]
    if network == "ltown":
        used += [history / "history.joblib"]
    result = {}
    for path in used:
        digest = sha(path)
        if expected.get(relative(path)) != digest:
            raise RuntimeError(f"Frozen artifact missing or hash differs: {path}")
        result[relative(path)] = digest
    # Every imported project source must still match the original code snapshot.
    for name, digest in expected.items():
        if name.endswith(".py") and (name.startswith("src/wdn/") or name.startswith("thesis_v2/experiments/early_warning/")):
            if sha(ROOT / name) != digest:
                raise RuntimeError(f"Frozen inference source differs: {name}")
    return result


@contextmanager
def checked_threads(calls, threads=4):
    """Retain original fitted parameters; verify accelerated outputs per call."""
    originals = (lgb.LGBMClassifier.predict_proba, HistGradientBoostingClassifier.predict_proba)
    def wrap(original, backend, original_threads):
        def predict(model, X, *args, **kwargs):
            params = joblib.hash(model.get_params(deep=False))
            def invoke(matrix, count):
                options = kwargs | ({"num_threads": count} if backend == "lightgbm" else {})
                with threadpool_limits(limits=count, user_api="openmp"):
                    return original(model, matrix, *args, **options)
            start = time.monotonic()
            accelerated = invoke(X, threads)
            idx = np.unique(np.linspace(0, len(X) - 1, min(512, len(X)), dtype=int)) if len(X) else []
            probe = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
            reference = invoke(probe, original_threads) if len(X) else accelerated
            equal = np.array_equal(accelerated[idx], reference)
            answer = accelerated if equal else invoke(X, original_threads)
            if params != joblib.hash(model.get_params(deep=False)):
                raise RuntimeError("Inference changed fitted parameters")
            calls.append({"backend": backend, "rows": len(X), "seconds": time.monotonic() - start,
                          "probe_bitwise_equal": equal, "original_thread_fallback": not equal})
            return answer
        return predict
    lgb.LGBMClassifier.predict_proba = wrap(originals[0], "lightgbm", 1)
    HistGradientBoostingClassifier.predict_proba = wrap(originals[1], "histogram", 2)
    try:
        yield
    finally:
        lgb.LGBMClassifier.predict_proba, HistGradientBoostingClassifier.predict_proba = originals


def count_report(decision, arrays):
    y = arrays["labels"] > 0
    families = arrays["families"]
    report = {}
    for code, name in [(None, "_overall"), *FAMILIES.items()]:
        mask = np.ones(len(y), bool) if code is None else families == code
        tp = int(np.sum(mask & y & decision))
        fp = int(np.sum(mask & ~y & decision))
        fn = int(np.sum(mask & y & ~decision))
        tn = int(np.sum(mask & ~y & ~decision))
        report[name] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "f1": 2*tp/max(1, 2*tp+fp+fn)}
    clean = families == 0
    report["_overall"].update(clean_rows=int(clean.sum()), clean_period_fp=int(np.sum(clean & decision)))
    report["_overall"]["clean_fpr"] = report["_overall"]["clean_period_fp"] / max(1, int(clean.sum()))
    return report


def compare_report(actual, expected, compact=False):
    """Exact saved counts, plus exact/rounded ratios according to original format."""
    if compact:
        for k in ("tp", "fp", "fn", "clean_period_fp"):
            if actual["_overall"][k] != expected[k]:
                raise AssertionError(f"Original {k}: {actual['_overall'][k]} != {expected[k]}")
        for family in FAMILIES.values():
            if round(actual[family]["f1"], 6) != expected[family]:
                raise AssertionError(f"Original rounded {family} F1 differs")
        if round(actual["_overall"]["f1"], 6) != expected["pooled_f1"]:
            raise AssertionError("Original rounded pooled F1 differs")
        if abs(actual["_overall"]["clean_fpr"] - expected["clean_fpr"]) > 1e-14:
            raise AssertionError("Original clean FPR differs")
    else:
        for family, row in expected.items():
            if family not in actual:
                continue
            for k in ("tp", "fp", "fn", "clean_period_fp"):
                if k in row and actual[family][k] != row[k]:
                    raise AssertionError(f"Original {family}/{k} differs")
            for k in ("f1", "clean_fpr"):
                if k in row and abs(actual[family][k] - row[k]) > 1e-14:
                    raise AssertionError(f"Original {family}/{k} differs")


def expected_report(network, seed, role, source=None):
    stage, history = paths(network, seed)
    if network == "modena":
        if role == "calibration":
            return None if source is not None else read(stage / "operating_points.json")["arms"]["candidate"]["report"]
        return read(stage / "evaluation_report.json")["per_data_seed"][str(source)]["arms"]["candidate"]
    if role == "calibration":
        result = read(history / "full_history_selection.json")
        return result["report"] if source is None else result["source_reports"][str(source)]
    return next(p["candidate"] for p in read(history / "confirmation.json")["pairs"] if p["source"] == source)


def check_baseline_alignment(network, role, source, arrays):
    base = CAMPAIGN / "single_model_baseline_v1"
    if role == "evaluation":
        path = base / network / f"evaluation_predictions_{source}.npz"
        with np.load(path) as z:
            for key in ("labels", "families", "source", "scenario", "timestep", "node"):
                if not np.array_equal(arrays[key], z[key]):
                    raise AssertionError(f"Baseline endpoint alignment differs: {key}")
    else:
        path = base / "cache" / f"{network}_calibration" / "metadata.npz"
        with np.load(path) as z:
            mask = z["source"] == source
            for key in ("labels", "families", "source", "scenario", "timestep", "node"):
                if not np.array_equal(arrays[key], z[key][mask]):
                    raise AssertionError(f"Calibration baseline endpoint alignment differs: {key}")


def extract_source(network, seed, role, job, bundles, model_hashes):
    manifest_path, manifest, piece = job
    source = int(piece["seed"])
    folder = SCORES / network / f"seed{seed}"
    target = folder / f"{role}_source{source}.npz"
    sidecar = target.with_suffix(".json")
    if target.exists() or sidecar.exists():
        if not (target.exists() and sidecar.exists()):
            raise RuntimeError(f"Incomplete existing output requires inspection: {target}")
        record = read(sidecar)
        if record["extractor_sha256"] != sha(__file__) or record["model_hashes"] != model_hashes:
            raise RuntimeError(f"Existing extraction provenance differs: {target}")
        print(f"REUSE {network} seed={seed} {role} source={source}", flush=True)
        return record
    start = time.monotonic()
    cache = ROOT / piece["cache"]
    if sha(cache) != piece["cache_sha256"]:
        raise RuntimeError(f"Original feature cache hash differs: {cache}")
    with np.load(cache) as z:
        arrays = {k: z[k] for k in (*META, "X")}
    arrays["event"] = globalise_events(arrays["event"], arrays["source"])
    names = manifest["feature_names"]
    assert len(arrays["X"]) == piece["rows"]
    check_baseline_alignment(network, role, source, arrays)
    rule_path, rule, thresholds = frozen_rule(network, seed)
    experts, head, heads = bundles
    calls = []
    print(f"START {network} seed={seed} {role} source={source} rows={len(arrays['X'])}", flush=True)
    with checked_threads(calls):
        causal, delayed = rc.apply_experts(experts, head, arrays)
        scores = rc.specialist_score_parts(causal, delayed, arrays)
        del causal, delayed
        local = rc.early_local_evidence(arrays["X"], names,
                                       np.column_stack((scores["causal_drift"], scores["causal_noise"])))
        early = heads["early"].predict_groups(local, arrays)
        del local
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
            retained = None
        else:
            history = paths(network, seed)[1]
            retained = history / ("calibration" if role == "calibration" else "confirmation") / f"source_{source}.npz"
            with np.load(retained) as z:
                for key in ("labels", "families", "source", "scenario", "event"):
                    if not np.array_equal(arrays[key], z[key]):
                        raise AssertionError(f"Retained General cache alignment differs: {key}")
                general = transformed_mixture(z["history_experts"], z["history_routing"],
                                              rule["router_temperature"], rule["uniform_shrinkage"])
                original_specialist = z["specialist"]
    payload = {k: arrays[k] for k in META}
    payload["general"] = np.asarray(general, dtype=np.float64)
    specialist = np.zeros(len(general), bool)
    for column, branch in enumerate(rc.BRANCHES):
        payload[branch] = np.asarray(scores[f"final_{branch}"], dtype=np.float64)
        payload[f"verifier_{branch}"] = np.asarray(keep[branch], dtype=np.float64)
        payload[f"veto_{branch}"] = veto[:, column]
        allowed = ~veto[:, column]
        cutoff = rule["verifier_cutoffs"][branch]
        if cutoff > 0:
            allowed &= keep[branch] >= cutoff
        payload[f"allow_{branch}"] = allowed
        specialist |= allowed & (payload[branch] > thresholds[branch])
    if network == "ltown" and not np.array_equal(specialist, original_specialist):
        raise AssertionError("Retained original L-Town specialist Boolean decisions differ")
    payload["original_decision"] = (general > thresholds["general"]) | specialist
    report = count_report(payload["original_decision"], arrays)
    expected = expected_report(network, seed, role, source)
    if expected is not None:
        compare_report(report, expected, compact=network == "modena")
    del arrays, scores, keep, veto, specialist, general
    gc.collect()
    folder.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".partial")
    with tmp.open("xb") as f:
        np.savez_compressed(f, **payload)
    npz_hash = sha(tmp)
    record = {"network": network, "model_seed": seed, "role": role, "source_seed": source,
              "rows": len(payload["labels"]), "thresholds": thresholds,
              "verifier_cutoffs": rule["verifier_cutoffs"], "model_hashes": model_hashes,
              "extractor_sha256": sha(__file__), "feature_cache": relative(cache),
              "feature_cache_sha256": piece["cache_sha256"], "manifest": relative(manifest_path),
              "manifest_sha256": sha(manifest_path), "original_rule": relative(rule_path),
              "retained_general_cache": relative(retained) if retained else None,
              "retained_general_cache_sha256": sha(retained) if retained else None,
              "original_counts": report, "original_report_checked": expected is not None,
              "calibration_pooled_check_required": role == "calibration", "baseline_endpoints_aligned": True,
              "score_dtype": "float64; lossless NPZ compression", "npz_sha256": npz_hash,
              "npz_bytes": tmp.stat().st_size, "elapsed_seconds": time.monotonic() - start,
              "prediction_calls": calls, "fit_called": False, "locked_test_read": False}
    temp_json = sidecar.with_suffix(".partial.json")
    temp_json.write_text(json.dumps(record, indent=2) + "\n")
    tmp.replace(target)
    temp_json.replace(sidecar)
    print(f"DONE {network} seed={seed} {role} source={source} seconds={record['elapsed_seconds']:.1f} MB={record['npz_bytes']/1e6:.1f}", flush=True)
    return record


def validate_calibration(network, seed):
    records = []
    for _, _, piece in source_jobs(network, "calibration"):
        path = SCORES / network / f"seed{seed}" / f"calibration_source{piece['seed']}.json"
        if not path.exists():
            return False
        records.append(read(path))
    report = {}
    for family in ("_overall", *FAMILIES.values()):
        report[family] = {k: sum(r["original_counts"][family][k] for r in records) for k in ("tp", "fp", "fn", "tn")}
        counts = report[family]
        counts["f1"] = 2*counts["tp"]/max(1, 2*counts["tp"]+counts["fp"]+counts["fn"])
    for key in ("clean_rows", "clean_period_fp"):
        report["_overall"][key] = sum(r["original_counts"]["_overall"][key] for r in records)
    report["_overall"]["clean_fpr"] = report["_overall"]["clean_period_fp"] / report["_overall"]["clean_rows"]
    compare_report(report, expected_report(network, seed, "calibration"), compact=network == "modena")
    path = SCORES / network / f"seed{seed}" / "calibration_validation.json"
    content = {"network": network, "model_seed": seed, "sources": [r["source_seed"] for r in records],
               "original_pooled_report_matches": True, "counts": report,
               "extractor_sha256": sha(__file__), "source_npz_sha256": {str(r["source_seed"]): r["npz_sha256"] for r in records}}
    if path.exists() and read(path) != content:
        raise RuntimeError(f"Existing calibration validation differs: {path}")
    if not path.exists():
        path.write_text(json.dumps(content, indent=2) + "\n")
    print(f"CALIBRATION VALIDATED {network} seed={seed}", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", required=True, choices=("modena", "ltown"))
    parser.add_argument("--seed", required=True, type=int, choices=range(701, 711))
    parser.add_argument("--role", default="both", choices=("calibration", "evaluation", "both"))
    parser.add_argument("--source", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and hashes only; do not unpickle or predict.")
    args = parser.parse_args()
    roles = ("calibration", "evaluation") if args.role == "both" else (args.role,)
    jobs = [(role, job) for role in roles for job in source_jobs(args.network, role)
            if args.source is None or int(job[2]["seed"]) == args.source]
    if not jobs:
        parser.error("Source is not an authorized original source for this role/network")
    hashes = validate_model_files(args.network, args.seed)
    for role, (_, _, piece) in jobs:
        if not (ROOT / piece["cache"]).is_file():
            raise FileNotFoundError(piece["cache"])
        if args.network == "ltown":
            history = paths(args.network, args.seed)[1]
            cached = history / ("calibration" if role == "calibration" else "confirmation") / f"source_{piece['seed']}.npz"
            if not cached.is_file():
                raise FileNotFoundError(cached)
    print(json.dumps({"network": args.network, "seed": args.seed,
                      "jobs": [{"role": role, "source": job[2]["seed"], "rows": job[2]["rows"]} for role, job in jobs],
                      "dry_run": args.dry_run, "frozen_artifacts_verified": True}), flush=True)
    if args.dry_run:
        return
    stage, _ = paths(args.network, args.seed)
    bundles = tuple(joblib.load(stage / name) for name in ("base_bundle.joblib", "delayed_bundle.joblib", "heads_bundle.joblib"))
    for role, job in jobs:
        extract_source(args.network, args.seed, role, job, bundles, hashes)
        gc.collect()
    if "calibration" in roles:
        validate_calibration(args.network, args.seed)


if __name__ == "__main__":
    main()
