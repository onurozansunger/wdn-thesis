"""Paired TRAIN-only pilot: existing MoE vs the same MoE with shared history.

One pre-existing source-held fold, one model seed, no Optuna. A deterministic
scenario partition *within held-out TRAIN* separates threshold selection from
diagnostics. This is not locked evaluation or a full deployed-system result.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score

from build_feature_cache import CAMPAIGN, ROOT, sha, write_json
from wdn.latency_deployment import FAMILY_NAMES, family_scores, quantile_threshold
from wdn.probe_residual_experts import MECHANISMS, ResidualExpertMixture
from wdn.shared_history import (HISTORY_COLUMNS, HISTORY_OFFSETS,
                                SharedHistoryExpertMixture,
                                causal_history_features, fit_presampled)

BUDGETS = (.00025, .0005, .001, .002, .003, .004, .005)


def load(path):
    with np.load(path) as archive:
        return {k: archive[k] for k in archive.files}


def predict_chunks(model, X, history=None):
    parts = {}
    for start in range(0, len(X), 50000):
        batch = X[start:start + 50000]
        if history is not None:
            batch = np.column_stack((batch, history[start:start + 50000]))
        prediction = model.predict(batch)
        for key, value in prediction.items():
            parts.setdefault(key, []).append(value.astype(np.float32))
    return {key: np.concatenate(value) for key, value in parts.items()}


def streams(prediction):
    return {"mixture": prediction["mixture"], "general": prediction["general"],
            **{f"expert_{name}": prediction["experts"][:, i]
               for i, name in enumerate(MECHANISMS)}}


def select(score, arrays, selection):
    scope = {k: arrays[k][selection] for k in ("labels", "families")}
    candidates = []
    for budget in BUDGETS:
        threshold = quantile_threshold(score, selection & (arrays["families"] == 0), budget)
        metrics = family_scores(score[selection] > threshold, scope)
        candidates.append({"threshold": threshold, "budget": budget, "metrics": metrics})
    # Identical, family-agnostic criterion in both arms; never maximise replay.
    return max(candidates, key=lambda c: (c["metrics"]["_overall"]["f1"],
                                          c["metrics"]["_overall"]["worst_family_f1"],
                                          -c["metrics"]["_overall"]["clean_fpr"]))


def run(network="ltown", fold=0, seed=701):
    folder = CAMPAIGN / f"stage_e_{network}/folds/fold_{fold}"
    output = CAMPAIGN / f"shared_history_pilot_v1/{network}/fold_{fold}_seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        print(f"Already complete: {output / 'summary.json'}", flush=True)
        return
    started = time.time()
    names = json.loads((CAMPAIGN / f"features/{network}_train/manifest.json").read_text())["feature_names"]
    signature = {"network": network, "fold": fold, "seed": seed,
        "history_offsets_hours": HISTORY_OFFSETS, "history_columns": HISTORY_COLUMNS,
        "threshold_budgets": BUDGETS,
        "selection_rule": "max pooled F1, then worst-family F1, then lower clean FPR",
        "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in
            (Path(__file__), ROOT / "src/wdn/shared_history.py",
             ROOT / "src/wdn/probe_residual_experts.py")},
        "input_hashes": {p.name: sha(p) for p in folder.glob("*.npz")},
        "protocol": "Original TRAIN only; existing fold reference fitted excluding held source. "
                    "Even held-source scenario IDs select thresholds; odd IDs diagnose. "
                    "Same training rows, weights, seed, HGB hyperparameters in both arms.",
        "scope": "Five-expert mixture branch only; seasonal branches, verifier, feedback "
                 "and deployed output unchanged. No promotion from this pilot.",
        "test_evaluated": False, "locked_eval_evaluated": False,
        "warmup_caveat": "Only cached endpoints at hour >=15 exist as history context."}
    signature = json.loads(json.dumps(signature))
    signature_path = output / "protocol_frozen.json"
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise RuntimeError("Pilot code or inputs changed; use a new version, not stale artifacts")
    write_json(signature_path, signature)

    def status(phase):
        print(f"[{time.time() - started:.0f}s] {phase}", flush=True)
        write_json(output / "status.json", {"phase": phase, "elapsed_seconds": time.time() - started,
                                            "updated": time.strftime("%Y-%m-%dT%H:%M:%S")})

    if not all((output / f"{arm}.joblib").exists() for arm in ("baseline", "shared_history")):
        status("loading full source-excluded TRAIN fold")
        train = load(folder / "features_train.npz")
        positives = np.flatnonzero(train["labels"] > .5)
        negatives = np.flatnonzero(train["labels"] <= .5)
        chosen = np.random.default_rng(seed).choice(negatives, min(60000, len(negatives)), replace=False)
        subset = np.r_[positives, chosen]
        weights = np.r_[np.ones(len(positives)), np.full(len(chosen), len(negatives) / len(chosen))]
        status(f"building shared history at {len(subset)} sampled training endpoints")
        history, added_names = causal_history_features(train, names, rows=subset)
        X = train["X"][subset]
        labels, families = train["labels"][subset], train["families"][subset]
        write_json(output / "training_sample.json", {
            "population_rows": len(train["labels"]), "positive_rows": len(positives),
            "sampled_clean_rows": len(chosen), "negative_population": len(negatives),
            "sources": np.unique(train["source"]).tolist(), "feature_names": names + added_names})
        del train, positives, negatives, subset
        gc.collect()
        for arm in ("baseline", "shared_history"):
            if (output / f"{arm}.joblib").exists():
                continue
            status(f"fitting {arm}: same five experts plus router")
            model = (ResidualExpertMixture(names, seed) if arm == "baseline" else
                     SharedHistoryExpertMixture(names + added_names, seed))
            fit_presampled(model, X if arm == "baseline" else np.column_stack((X, history)),
                           labels, families, weights)
            temporary = output / f"{arm}.joblib.tmp"
            joblib.dump(model, temporary)
            temporary.replace(output / f"{arm}.joblib")
            del model
        del X, history, labels, families, weights
        gc.collect()

    status("predicting held-out TRAIN source; no EVAL or test opened")
    held = load(folder / "features_held_out.npz")
    history, _ = causal_history_features(held, names)
    predictions = {}
    for arm in ("baseline", "shared_history"):
        path = output / f"{arm}_predictions.npz"
        if not path.exists():
            model = joblib.load(output / f"{arm}.joblib")
            predicted = predict_chunks(model, held["X"], history if arm == "shared_history" else None)
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **predicted)
            temporary.replace(path)
            del model, predicted
        predictions[arm] = load(path)
    del history, held["X"]
    gc.collect()
    selection = held["scenario"] % 2 == 0
    diagnostic = ~selection
    for part in (selection, diagnostic):
        if not part.any() or any(not np.any(part & (held["families"] == f) & (held["labels"] > 0))
                                 for f in FAMILY_NAMES):
            raise RuntimeError("Predeclared inner partition lacks a family; do not adapt using scores")
    frozen = {arm: {name: select(score, held, selection) for name, score in streams(p).items()}
              for arm, p in predictions.items()}
    write_json(output / "thresholds_frozen.json", frozen)
    status("thresholds frozen; computing diagnostic expert audit")
    report = {"protocol": signature, "results": {},
              "threshold_scenarios": np.unique(held["scenario"][selection]).tolist(),
              "diagnostic_scenarios": np.unique(held["scenario"][diagnostic]).tolist()}
    scope = {k: held[k][diagnostic] for k in ("labels", "families")}
    for arm, prediction in predictions.items():
        report["results"][arm] = {}
        for name, score in streams(prediction).items():
            metrics = family_scores(score[diagnostic] > frozen[arm][name]["threshold"], scope)
            for code, family in FAMILY_NAMES.items():
                # Family-event rows plus shared clean episodes; no other attack positives.
                mask = diagnostic & np.isin(held["families"], (0, code))
                metrics[family]["auprc_family_plus_clean"] = float(
                    average_precision_score(held["labels"][mask], score[mask]))
            report["results"][arm][name] = metrics
        report["results"][arm]["routing_positive_mean"] = {
            family: prediction["routing"][diagnostic & (held["families"] == code)
                                           & (held["labels"] > 0)].mean(axis=0).tolist()
            for code, family in FAMILY_NAMES.items()}
    before, after = (report["results"][arm]["mixture"] for arm in ("baseline", "shared_history"))
    report["mixture_f1_deltas"] = {family: after[family]["f1"] - before[family]["f1"]
                                    for family in FAMILY_NAMES.values()}
    report["mixture_f1_deltas"]["pooled"] = after["_overall"]["f1"] - before["_overall"]["f1"]
    report["elapsed_seconds"] = time.time() - started
    write_json(output / "summary.json", report)
    status("complete; TRAIN diagnostic only, deployed model unchanged")
    print(json.dumps(report["mixture_f1_deltas"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", choices=("ltown", "modena"), default="ltown")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=701)
    run(**vars(parser.parse_args()))
