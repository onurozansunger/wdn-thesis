"""Frozen paired TRAIN replication of the received-pressure history pilot.

No calibration, evaluation, locked test, or deployed artifact is opened. The
five tree components are internal to General; this is not a full-system test.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.metrics import average_precision_score

from build_feature_cache import CAMPAIGN, ROOT, atomic_npz, check_distribution, config_of, sha, write_json
from screen_observed_history import build_bank
from screen_shared_history import BUDGETS, load, predict_chunks, select, streams
from wdn.latency_deployment import FAMILY_NAMES, family_scores
from wdn.observed_history import DELTA_BIN_SIGMA, OBSERVED_LAGS
from wdn.probe_residual_experts import ResidualExpertMixture
from wdn.shared_history import SharedHistoryExpertMixture, fit_presampled

OUTPUT = CAMPAIGN / "observed_history_replication_v1"
FOLDS = CAMPAIGN / "stage_e_ltown/folds"
MANIFEST = CAMPAIGN / "features/ltown_train/manifest.json"
PILOT = CAMPAIGN / "observed_history_pilot_v1/ltown/fold_0_seed_701"
SEEDS = (701, 702, 703, 704, 705)
PAIRS = [(f, s) for f in (1, 2, 3, 4, 0) for s in SEEDS if (f, s) != (0, 701)]
ARMS = ("baseline", "observed_history")
GATE = {"minimum_replay_wins": 16, "minimum_mean_pooled_delta": 0.,
        "minimum_source_pooled_delta": -.005, "minimum_mean_other_family_delta": -.005,
        "minimum_source_other_family_delta": -.01, "maximum_mean_clean_fpr_delta": 0.,
        "maximum_source_clean_fpr_delta": .0001}


def frozen_write(path, value):
    value = json.loads(json.dumps(value))
    if path.exists() and json.loads(path.read_text()) != value:
        raise RuntimeError(f"Frozen artifact differs: {path}")
    write_json(path, value)


def freeze():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    pilot = json.loads((PILOT / "protocol_frozen.json").read_text())
    for path, digest in pilot["source_hashes"].items():
        if sha(ROOT / path) != digest:
            raise RuntimeError(f"Pilot recipe changed: {path}")
    files = [MANIFEST, FOLDS / "signature.json", PILOT / "protocol_frozen.json",
             PILOT / "summary.json"]
    files += sorted((ROOT / "src/wdn").rglob("*.py"))
    files += [Path(__file__), Path(__file__).with_name("screen_observed_history.py"),
              Path(__file__).with_name("screen_shared_history.py"),
              Path(__file__).with_name("build_feature_cache.py")]
    benchmark = config_of(ROOT / "data/thesis_v2" / manifest["pieces"][0]["directory"])
    for piece in manifest["pieces"]:
        directory = ROOT / "data/thesis_v2" / piece["directory"]
        check_distribution(directory, benchmark)
        for name in ("snapshots.pkl", "corrupted.pkl", "generate_config.yaml"):
            path = directory / name
            if sha(path) != pilot["raw_hashes"][piece["directory"]][name]:
                raise RuntimeError(f"Pilot TRAIN source changed: {path}")
            files.append(path)
    for fold in range(5):
        files += [FOLDS / f"fold_{fold}" / name for name in
                  ("features_train.npz", "features_held_out.npz", "reference.joblib")]
    signature = {"pairs": PAIRS, "primary_folds": [1, 2, 3, 4],
        "supplemental_folds": [0], "seeds": SEEDS, "gate": GATE,
        "recipe": {"lags": OBSERVED_LAGS, "delta_bin_sigma": DELTA_BIN_SIGMA,
                   "rounding_m": 0., "added_feature_count": 42,
                   "threshold_budgets": BUDGETS, "sampled_negative_cap": 60000},
        "selection": "Even held-source TRAIN scenario IDs; max pooled F1, then worst-family F1, then lower clean FPR, independently in each arm.",
        "diagnostic": "Odd held-source TRAIN scenarios; all pairs completed before gate assessment; no tuning on replication results.",
        "gate_scope": "Primary 20 pairs only. Positive replay mean in each source and overall, >=16 positive pairs. Numeric guards in gate. Supplemental fold 0 never determines progression.",
        "inference_scope": "General branch residual mixture only: five internal tree components plus internal router; no new top-level expert or replay-specific rule.",
        "next_stage": "Only if gate passes: full TRAIN fit and calibration-only protected full-system selection, then freeze before one fresh independent confirmation.",
        "dependence": "Model seeds share held scenarios. Source means, not 20 independent datasets, determine source consistency; no iid seed-based significance claim.",
        "missing_pressure": .5, "missing_flow": .5,
        "test_evaluated": False, "locked_eval_evaluated": False,
        "calibration_evaluated": False, "deployed_replay_f1": .6757,
        "versions": {"python": sys.version, "numpy": np.__version__, "sklearn": sklearn.__version__, "joblib": joblib.__version__},
        "input_and_code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in files}}
    frozen_write(OUTPUT / "protocol_frozen.json", signature)
    print("Frozen protocol verified", flush=True)


def validate_partition(train, held, manifest, fold):
    source = manifest["pieces"][fold]["seed"]
    expected = {p["seed"] for p in manifest["pieces"]} - {source}
    if set(np.unique(train["source"])) != expected or set(np.unique(held["source"])) != {source}:
        raise RuntimeError("Source-held partition changed")
    for arrays in (train, held):
        for piece in manifest["pieces"]:
            scenarios = arrays["scenario"][arrays["source"] == piece["seed"]]
            if not set(np.unique(scenarios)) <= {piece["seed"] * 1000 + s for s in piece["scenarios"]}:
                raise RuntimeError("Non-TRAIN scenario found")
    selection = held["scenario"] % 2 == 0
    for part in (selection, ~selection):
        if not part.any() or any(not np.any(part & (held["families"] == f) & (held["labels"] > 0)) for f in FAMILY_NAMES):
            raise RuntimeError("Frozen even/odd TRAIN partition lacks a family; no adaptive repartition")


def run_pair(fold, seed):
    if (fold, seed) not in PAIRS:
        raise ValueError("Pair is outside frozen replication")
    output = OUTPUT / f"fold_{fold}_seed_{seed}"
    output.mkdir(exist_ok=True)
    start = time.monotonic()
    def status(phase):
        write_json(output / "status.json", {"phase": phase, "elapsed_seconds": time.monotonic()-start})
        print(f"fold={fold} seed={seed} [{time.monotonic()-start:.0f}s] {phase}", flush=True)
    protocol = json.loads((OUTPUT / "protocol_frozen.json").read_text())
    if protocol["pairs"] != [list(p) for p in PAIRS]:
        raise RuntimeError("Pair plan changed")
    frozen_write(output / "signature.json", {"fold": fold, "seed": seed,
                 "protocol_sha256": sha(OUTPUT / "protocol_frozen.json")})
    if (output / "summary.json").exists():
        for name, digest in json.loads((output / "artifacts_sha256.json").read_text()).items():
            if sha(output / name) != digest:
                raise RuntimeError(f"Completed pair artifact changed: {name}")
        return
    manifest = json.loads(MANIFEST.read_text())
    names = manifest["feature_names"]
    folder = FOLDS / f"fold_{fold}"
    status("loading frozen TRAIN fold")
    train, held = load(folder / "features_train.npz"), load(folder / "features_held_out.npz")
    validate_partition(train, held, manifest, fold)
    positives, negatives = np.flatnonzero(train["labels"] > .5), np.flatnonzero(train["labels"] <= .5)
    chosen = np.random.default_rng(seed).choice(negatives, min(60000, len(negatives)), replace=False)
    subset = np.r_[positives, chosen]
    weights = np.r_[np.ones(len(positives)), np.full(len(chosen), len(negatives)/len(chosen))]
    frozen_write(output / "training_sample.json", {"population_rows": len(train["labels"]),
        "positive_rows": len(positives), "sampled_negative_rows": len(chosen),
        "negative_population": len(negatives), "sample_sha256": hashlib.sha256(subset.tobytes()).hexdigest(),
        "weight_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
        "sources": np.unique(train["source"]).tolist()})
    train = {k: v[subset] for k, v in train.items()}
    del positives, negatives, chosen, subset
    gc.collect()
    status("building unchanged 42-feature received-observation bank")
    history, added = build_bank(train, names, manifest, (0.,))
    frozen_write(output / "feature_names.json", names + added)
    for arm in ARMS:
        path = output / f"{arm}.joblib"
        if path.exists():
            continue
        status(f"fitting {arm}: General's five internal components and router")
        model = ResidualExpertMixture(names, seed) if arm == "baseline" else SharedHistoryExpertMixture(names + added, seed)
        fit_presampled(model, train["X"] if arm == "baseline" else np.column_stack((train["X"], history[0.])),
                       train["labels"], train["families"], weights)
        joblib.dump(model, path.with_suffix(".tmp"))
        path.with_suffix(".tmp").replace(path)
        del model
    del train, history, weights
    gc.collect()
    status("predicting both arms on held-source TRAIN")
    history, _ = build_bank(held, names, manifest, (0.,))
    for arm in ARMS:
        path = output / f"{arm}_predictions.npz"
        if not path.exists():
            model = joblib.load(output / f"{arm}.joblib")
            predicted = predict_chunks(model, held["X"], None if arm == "baseline" else history[0.])
            atomic_npz(path, **predicted)
            del model, predicted
    del history, held["X"]
    gc.collect()
    selection = held["scenario"] % 2 == 0
    diagnostic = ~selection
    predictions = {arm: load(output / f"{arm}_predictions.npz") for arm in ARMS}
    thresholds = {arm: {name: select(score, held, selection) for name, score in streams(p).items()}
                  for arm, p in predictions.items()}
    frozen_write(output / "thresholds_frozen.json", thresholds)
    status("thresholds frozen; computing diagnostic metrics")
    report = {"fold": fold, "seed": seed, "source": int(held["source"][0]),
              "scope": "TRAIN diagnostic of General mixture only; not independent or deployed performance",
              "threshold_scenarios": np.unique(held["scenario"][selection]).tolist(),
              "diagnostic_scenarios": np.unique(held["scenario"][diagnostic]).tolist(),
              "diagnostic_events": {f: len(np.unique(held["event"][diagnostic & (held["families"] == c) & (held["labels"] > .5)])) for c, f in FAMILY_NAMES.items()},
              "results": {}}
    scope = {k: held[k][diagnostic] for k in ("labels", "families")}
    for arm, prediction in predictions.items():
        report["results"][arm] = {}
        for name, score in streams(prediction).items():
            metrics = family_scores(score[diagnostic] > thresholds[arm][name]["threshold"], scope)
            for code, family in FAMILY_NAMES.items():
                mask = diagnostic & np.isin(held["families"], (0, code))
                metrics[family]["auprc_family_plus_clean"] = float(average_precision_score(held["labels"][mask], score[mask]))
            report["results"][arm][name] = metrics
    before, after = (report["results"][a]["mixture"] for a in ARMS)
    report["deltas"] = {f: after[f]["f1"]-before[f]["f1"] for f in FAMILY_NAMES.values()}
    report["deltas"].update(pooled=after["_overall"]["f1"]-before["_overall"]["f1"],
                             clean_fpr=after["_overall"]["clean_fpr"]-before["_overall"]["clean_fpr"])
    report["elapsed_seconds"] = time.monotonic()-start
    write_json(output / "summary.json", report)
    write_json(output / "artifacts_sha256.json", {p.name: sha(p) for p in output.iterdir() if p.name not in ("status.json", "artifacts_sha256.json")})
    status("complete; diagnostic saved without adaptive tuning")


def assess(reports):
    primary = [r for r in reports if r["fold"] in (1, 2, 3, 4)]
    if {(r["fold"], r["seed"]) for r in primary} != {(f, s) for f in (1, 2, 3, 4) for s in SEEDS} or len(primary) != 20:
        raise RuntimeError("All 20 primary pairs are required")
    keys = list(primary[0]["deltas"])
    means = {k: float(np.mean([r["deltas"][k] for r in primary])) for k in keys}
    sources = {str(f): {k: float(np.mean([r["deltas"][k] for r in primary if r["fold"] == f])) for k in keys} for f in (1, 2, 3, 4)}
    wins = sum(r["deltas"]["replay"] > 0 for r in primary)
    checks = {"replay_mean_positive": means["replay"] > 0,
        "replay_positive_every_source": all(s["replay"] > 0 for s in sources.values()),
        "replay_at_least_16_of_20_wins": wins >= GATE["minimum_replay_wins"],
        "pooled_mean_preserved": means["pooled"] >= GATE["minimum_mean_pooled_delta"],
        "pooled_source_guard": all(s["pooled"] >= GATE["minimum_source_pooled_delta"] for s in sources.values()),
        "clean_fpr_mean_preserved": means["clean_fpr"] <= GATE["maximum_mean_clean_fpr_delta"],
        "clean_fpr_source_guard": all(s["clean_fpr"] <= GATE["maximum_source_clean_fpr_delta"] for s in sources.values())}
    for family in FAMILY_NAMES.values():
        if family != "replay":
            checks[f"{family}_mean_guard"] = means[family] >= GATE["minimum_mean_other_family_delta"]
            checks[f"{family}_source_guard"] = all(s[family] >= GATE["minimum_source_other_family_delta"] for s in sources.values())
    arm_means = {a: {**{f: float(np.mean([r["results"][a]["mixture"][f]["f1"] for r in primary])) for f in FAMILY_NAMES.values()},
        **{key: float(np.mean([r["results"][a]["mixture"]["_overall"][metric] for r in primary])) for key, metric in (("pooled", "f1"), ("clean_fpr", "clean_fpr"))}} for a in ARMS}
    return {"passes": all(checks.values()), "checks": checks, "replay_wins": wins,
            "primary_pair_count": len(primary), "mean_deltas": means, "source_mean_deltas": sources,
            "arm_means": arm_means}


def aggregate():
    reports = [json.loads((OUTPUT / f"fold_{f}_seed_{s}/summary.json").read_text()) for f, s in PAIRS]
    gate = assess(reports)
    result = {"scope": "TRAIN replication only; correlated model seeds; no independent or deployed claim",
        "gate": gate, "pairs": reports, "confirmed_full_system_replay_f1": .6757,
        "decision": "proceed_to_full_train_and_calibration" if gate["passes"] else "reject_progression_under_frozen_guards",
        "test_evaluated": False, "locked_eval_evaluated": False, "calibration_evaluated": False}
    write_json(OUTPUT / "summary.json", result)
    lines = ["# Received-observation history: frozen TRAIN replication", "",
        "General branch only, containing five internal tree components. These are TRAIN diagnostics, not independent or deployed performance. Confirmed full-system replay F1 remains **0.6757**.", "",
        f"Progression gate: **{'PASS' if gate['passes'] else 'FAIL'}**. Primary replay gains: {gate['replay_wins']}/20 paired runs across four held sources. Model seeds reuse scenarios and are not independent datasets.", "",
        "| Metric | Baseline mean | History mean | Paired mean change |", "|---|---:|---:|---:|"]
    for metric, delta in gate["mean_deltas"].items():
        lines.append(f"| {metric} | {gate['arm_means']['baseline'][metric]:.6f} | {gate['arm_means']['observed_history'][metric]:.6f} | {delta:+.6f} |")
    lines += ["", "## Frozen guard results", ""]
    lines += [f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in gate["checks"].items()]
    lines += ["", "## All new paired TRAIN diagnostics", "", "| Fold | Seed | Replay before | Replay after | Pooled change | Clean FPR change |", "|---|---|---:|---:|---:|---:|"]
    for r in reports:
        b, a = (r["results"][arm]["mixture"]["replay"]["f1"] for arm in ARMS)
        lines.append(f"| {r['fold']} | {r['seed']} | {b:.4f} | {a:.4f} | {r['deltas']['pooled']:+.6f} | {r['deltas']['clean_fpr']:+.6f} |")
    lines += ["", "Fold 0 is supplemental and excluded from the progression gate. The original fold-0/701 pilot is preserved separately. All 24 new pairs were required before assessing the gate. No feature, hyperparameter, threshold-selection rule, generator, severity, missingness, split, or locked test was changed.", ""]
    (OUTPUT / "report.md").write_text("\n".join(lines))
    print(json.dumps({"decision": result["decision"], "gate": gate}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs=2, type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    if args.pair:
        run_pair(*args.pair)
        return
    freeze()
    if args.freeze_only:
        return
    def worker(pair):
        f, s = pair
        env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with (OUTPUT / f"fold_{f}_seed_{s}.log").open("a") as log:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--pair", str(f), str(s)],
                           env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"Completed fold={f} seed={s}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(worker, PAIRS))
    aggregate()


if __name__ == "__main__":
    main()
