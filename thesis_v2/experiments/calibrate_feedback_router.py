"""Calibrate the feedback-router candidate without reading any EVAL or test set."""
from __future__ import annotations

import fcntl
import gc
import hashlib
import json
import time
from itertools import product
from pathlib import Path

import joblib
import numpy as np

from wdn.delayed_decision_features import delayed_decision_features
from wdn.evidence_feedback import (SCORE_EVIDENCE_NAMES, guarded_specialist_scores,
                                   local_evidence)
from wdn.latency_deployment import (FAMILY_NAMES, family_scores, forward_max,
                                    logit_blend, quantile_threshold,
                                    specialist_bank)
from wdn.run_expert_redesign import CampaignData

from evaluate_cross_mechanism_deployment import branch_scores, decide


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/thesis_v2"
ORIGINAL = DATA / "operational_modena_seed811"
CALIBRATION_SEEDS = (12811, 13811, 14811, 15811)
REFERENCE = ROOT / "runs/operational/seasonal_family_deployment_v2/full/reference.joblib"
FEEDBACK = ROOT / "runs/operational/feedback_router_v2/bundle.joblib"
BASE_RUN = ROOT / "runs/operational/cross_mechanism_deployment_v1"
SPLITS = ROOT / "runs/operational/blind_reference_probe_rank16/splits.json"
PROTOCOL = ROOT / "thesis_v2/FEEDBACK_ROUTER_EVAL3_PROTOCOL.md"
OUTPUT = ROOT / "runs/operational/feedback_router_calibration_v2"
DELTA = 3
CLEAN_BUDGET = .005
ROUTER_STRENGTHS = (.15, .30, .50)
FEEDBACK_STRENGTHS = (.15, .30, .50, .75)
MIXTURE_SHARES = (.05, .10, .15)
DRIFT_SHARES = (.10, .15, .20, .25)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def atomic_npz(path, **arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def score_part(directory, seed, scenarios, models, frozen_piece):
    reference, feedback = models
    data = CampaignData(directory)
    arrays, names = specialist_bank(data, scenarios, reference)
    if tuple(names) != feedback.names:
        raise ValueError("Feature schema differs from the fitted feedback router")
    X = arrays["X"]
    indexed = {**arrays,
        "source": np.full(len(X), seed),
        "scenario": np.asarray(arrays["scenario"]) + seed * 1000}
    for key in ("labels", "families", "source", "scenario", "timestep", "node"):
        value = indexed[key] if key in indexed else arrays[key]
        if not np.array_equal(np.asarray(value), np.asarray(frozen_piece[key])):
            raise ValueError(f"Frozen calibration alignment failed for {key}")
    frozen = np.asarray(frozen_piece["promoted"])
    pooled = np.column_stack((forward_max(frozen[:, 0], indexed, DELTA),
                              forward_max(frozen[:, 1], indexed, DELTA)))
    delayed = np.asarray(frozen_piece["promoted_head"])
    final = np.column_stack((logit_blend(delayed[:, 0], pooled[:, 0], .5),
                             logit_blend(delayed[:, 1], pooled[:, 1], .5)))
    parts = {
        "frozen_drift": frozen[:, 0], "frozen_noise": frozen[:, 1],
        "maxpool_drift": pooled[:, 0], "maxpool_noise": pooled[:, 1],
        "delayed_drift": delayed[:, 0], "delayed_noise": delayed[:, 1],
        "final_drift": final[:, 0], "final_noise": final[:, 1],
    }
    if tuple(parts) != SCORE_EVIDENCE_NAMES:
        raise AssertionError("Score evidence order changed")
    local = local_evidence(X, names, parts)
    predicted = feedback.predict(local, indexed)
    result = {
        "labels": np.asarray(arrays["labels"]),
        "families": np.asarray(arrays["families"]),
        "source": indexed["source"], "scenario": indexed["scenario"],
        "timestep": np.asarray(arrays["timestep"]), "node": np.asarray(arrays["node"]),
        "base_specialists": final,
        "router": predicted["router"], "feedback": predicted["feedback"],
    }
    del data, arrays, X, local, indexed
    gc.collect()
    return result


def report_compact(report):
    return {name: float(report[name]["f1"]) for name in FAMILY_NAMES.values()} | {
        "overall": float(report["_overall"]["f1"]),
        "clean_fpr": float(report["_overall"]["clean_fpr"]),
        "worst": float(report["_overall"]["worst_family_f1"])}


def gate(report, baseline):
    return (report["_overall"]["clean_fpr"] <= CLEAN_BUDGET
        and report["random"]["f1"] >= .90 and report["targeted"]["f1"] >= .90
        and report["replay"]["f1"] >= .82
        and report["drift"]["f1"] >= .81 and report["noise"]["f1"] >= .81
        and report["_overall"]["worst_family_f1"] >= baseline["_overall"]["worst_family_f1"]
        and report["_overall"]["f1"] >= baseline["_overall"]["f1"] - .005)


def replay_diagnostics(decision, corpus):
    labels = np.asarray(corpus["labels"]) > 0
    scope = np.asarray(corpus["families"]) == 2
    return {"tp": int(np.sum(decision & labels & scope)),
            "fp": int(np.sum(decision & ~labels & scope)),
            "fn": int(np.sum(~decision & labels & scope))}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / "campaign.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUTPUT / "selection_frozen.json").exists():
        print("Feedback-router calibration already frozen; no reselection", flush=True)
        return
    started = time.monotonic()
    reference = joblib.load(REFERENCE)
    feedback = joblib.load(FEEDBACK)
    models = reference, feedback
    base = dict(np.load(BASE_RUN / "scores_calibration.npz"))
    split = json.loads(SPLITS.read_text())
    pieces = [(ORIGINAL, 811, split["calibration"])] + [
        (DATA / f"operational_calibration_expansion_seed{seed}", seed, list(range(24)))
        for seed in CALIBRATION_SEEDS]
    saved = []
    for directory, seed, scenarios in pieces:
        cache = OUTPUT / f"feedback_seed{seed}.npz"
        if not cache.exists():
            print("scoring feedback evidence", seed, flush=True)
            mask = np.asarray(base["source"]) == seed
            frozen_piece = {key: np.asarray(value)[mask] for key, value in base.items()}
            atomic_npz(cache, **score_part(directory, seed, scenarios, models, frozen_piece))
        saved.append(dict(np.load(cache)))
    meta = {key: np.concatenate([part[key] for part in saved], axis=0) for key in saved[0]}
    for key in ("labels", "families", "source", "scenario", "timestep", "node"):
        if not np.array_equal(meta[key], base[key]):
            raise ValueError(f"Calibration alignment failed for {key}")
    base_scores = branch_scores(base, 0, "promoted", "blend", DELTA)
    if not np.allclose(meta["base_specialists"][:, 0], base_scores["drift"], atol=1e-12):
        raise ValueError("Recomputed drift scores do not match the frozen corpus")
    if not np.allclose(meta["base_specialists"][:, 1], base_scores["noise"], atol=1e-12):
        raise ValueError("Recomputed noise scores do not match the frozen corpus")

    old_selection = json.loads((BASE_RUN / "selection_frozen.json").read_text())
    old_rule = old_selection["selected_rule"]
    old_decision = decide(old_rule, base_scores)
    baseline = family_scores(old_decision, base)
    candidates = []
    clean = (base["families"] == 0) & (base["labels"] == 0)
    for router_strength, feedback_strength in product(ROUTER_STRENGTHS, FEEDBACK_STRENGTHS):
        guarded = guarded_specialist_scores(
            meta["base_specialists"], meta["router"], meta["feedback"],
            router_strength, feedback_strength)
        scores = {"mixture": base_scores["mixture"],
                  "drift": guarded[:, 0], "noise": guarded[:, 1]}
        for mixture_share, drift_share in product(MIXTURE_SHARES, DRIFT_SHARES):
            noise_share = round(1. - mixture_share - drift_share, 10)
            if noise_share <= 0:
                continue
            shares = (mixture_share, drift_share, noise_share)
            thresholds = {name: quantile_threshold(scores[name], clean, CLEAN_BUDGET * share)
                          for name, share in zip(("mixture", "drift", "noise"), shares)}
            decision = np.logical_or.reduce([
                scores[name] > thresholds[name] for name in ("mixture", "drift", "noise")])
            report = family_scores(decision, base)
            if gate(report, baseline):
                candidates.append(((report["_overall"]["worst_family_f1"],
                                    report["_overall"]["f1"]), {
                    "rule": "feedback_budgeted_or", "mixture_delta": 0,
                    "specialist_delta": DELTA,
                    "router_strength": router_strength,
                    "feedback_strength": feedback_strength,
                    "budget_shares": list(shares), "thresholds": thresholds,
                }, report, decision))
    if candidates:
        _, rule, selected_report, selected_decision = max(candidates, key=lambda item: item[0])
        promoted = True
    else:
        rule, selected_report, selected_decision, promoted = None, None, None, False

    router_prediction = meta["router"].argmax(1)
    target = np.asarray(base["families"]).copy()
    target[target == 5] = 1
    router_eval = {}
    for code, name in enumerate(("clean", "abrupt", "replay", "drift", "noise")):
        selected = target == code
        router_eval[name] = {"rows": int(selected.sum()),
            "accuracy": float(np.mean(router_prediction[selected] == code)),
            "mean_probability": float(meta["router"][selected, code].mean())}
    result = {
        "status": "promoted" if promoted else "rejected",
        "scope": "99-scenario calibration only",
        "protocol_sha256": sha(PROTOCOL),
        "feedback_bundle_sha256": sha(FEEDBACK),
        "base_selection_sha256": sha(BASE_RUN / "selection_frozen.json"),
        "candidate_grid": {"router_strength": ROUTER_STRENGTHS,
                           "feedback_strength": FEEDBACK_STRENGTHS,
                           "mixture_share": MIXTURE_SHARES,
                           "drift_share": DRIFT_SHARES},
        "baseline": report_compact(baseline),
        "baseline_replay": replay_diagnostics(old_decision, base),
        "router_calibration": router_eval,
        "selected_rule": rule,
        "selected": report_compact(selected_report) if promoted else None,
        "selected_replay": replay_diagnostics(selected_decision, base) if promoted else None,
        "passing_candidates": len(candidates),
        "eval1_evaluated": False, "eval2_evaluated": False,
        "eval3_evaluated": False, "test_evaluated": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(OUTPUT / "selection_frozen.json", result)
    atomic_npz(OUTPUT / "feedback_calibration.npz",
               router=meta["router"], feedback=meta["feedback"],
               labels=meta["labels"], families=meta["families"],
               source=meta["source"], scenario=meta["scenario"],
               timestep=meta["timestep"], node=meta["node"])
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
