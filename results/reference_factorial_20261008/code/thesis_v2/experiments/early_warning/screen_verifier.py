"""Stage C: the bounded three-candidate pooled-F1 screen, on Modena calibration.

Candidates, frozen in EARLY_WARNING_MULTISEED_PROTOCOL.md section 7:

* **C1** stricter branch-wise calibration — threshold-only control,
* **C2** a learned per-branch alarm verifier over causal and bounded-future
  evidence, reference reliability, support and temporal consistency,
* **C3** C2 plus the early-warning trajectory and consistency signal.

The verifiers are fitted on TRAIN out-of-fold only. Selection happens on
calibration against the gates frozen in the protocol, optimising **calibration
pooled F1** subject to those gates — not worst-family F1 alone. If nothing
passes, that is recorded as the campaign's answer; the gates are not relaxed.

    /opt/miniconda3/bin/python thesis_v2/experiments/early_warning/screen_verifier.py
"""
from __future__ import annotations

import fcntl
import itertools
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from wdn.alarm_verifier import (BRANCHES, BRANCH_FAMILY, AlarmVerifier,
                                candidate_rows, verifier_features)
from wdn.early_warning import early_local_evidence
from wdn.evidence_feedback import balanced_weights, contiguous_groups, specialist_veto_masks
from wdn.latency_deployment import (FAMILY_NAMES, forward_max, logit_blend,
                                    quantile_threshold)

from build_feature_cache import ROOT, sha, write_json
from campaign_data import load_calibration, load_train_oof

CAMPAIGN = ROOT / "runs/operational/early_warning_multiseed_v1"
EARLY = CAMPAIGN / "early_head_v1"
OUTPUT = CAMPAIGN / "verifier_screen_v1"
VETO = ROOT / "runs/operational/feedback_router_veto_v1/selection_frozen.json"

DELTA = 3
CLEAN_BUDGET = .005
#: The deployed budgeted-OR share grid, unchanged.
SHARE_STEPS = [round(.05 * k, 2) for k in range(1, 19)]
#: 0.0 means "verifier off for this branch", so C1 is nested inside C2 and C3.
CUTOFFS = (0., .20, .35, .50, .65, .80)
#: Candidate region: the top 2% of clean rows by that branch's final score.
CANDIDATE_QUANTILE = .02

VERIFIER_MODEL = dict(max_iter=200, max_leaf_nodes=23, min_samples_leaf=40,
                      learning_rate=.06, l2_regularization=12., early_stopping=False)

#: Frozen selection gates (protocol section 6).
GATES = {"drift": .80, "noise": .80, "random": .90, "replay": .90, "targeted": .90}
CLEAN_FPR_GATE = .005
MAX_FAMILY_DETERIORATION = .01


def specialist_score_parts(causal, head, arrays, delta=DELTA):
    """The eight score channels for one corpus, on the declared decision clock."""
    pooled = np.column_stack((forward_max(causal[:, 0], arrays, delta),
                              forward_max(causal[:, 1], arrays, delta)))
    final = np.column_stack((logit_blend(head[:, 0], pooled[:, 0], .5),
                             logit_blend(head[:, 1], pooled[:, 1], .5)))
    return {"causal_drift": causal[:, 0], "causal_noise": causal[:, 1],
            "maxpool_drift": pooled[:, 0], "maxpool_noise": pooled[:, 1],
            "delayed_drift": head[:, 0], "delayed_noise": head[:, 1],
            "final_drift": final[:, 0], "final_noise": final[:, 1]}


def fast_scorer(corpus):
    """Precompute the masks every candidate is scored against, once."""
    labels = np.asarray(corpus["labels"]) > 0
    families = np.asarray(corpus["families"])
    masks = {name: families == code for code, name in FAMILY_NAMES.items()}
    clean = families == 0
    negative = ~labels

    def score(decision):
        result = {}
        worst = 1.
        for name, scope in masks.items():
            tp = int(np.sum(labels & decision & scope))
            fp = int(np.sum(negative & decision & scope))
            fn = int(np.sum(labels & ~decision & scope))
            f1 = 2 * tp / max(1, 2 * tp + fp + fn)
            result[name] = {"f1": f1, "tp": tp, "fp": fp, "fn": fn}
            worst = min(worst, f1)
        tp = int(np.sum(labels & decision))
        fp = int(np.sum(negative & decision))
        fn = int(np.sum(labels & ~decision))
        result["_overall"] = {
            "f1": 2 * tp / max(1, 2 * tp + fp + fn),
            "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
            "tp": tp, "fp": fp, "fn": fn,
            "clean_fpr": float(decision[clean].mean()),
            "all_negative_fpr": float(decision[negative].mean()),
            "worst_family_f1": worst,
            "clean_period_fp": int(np.sum(negative & decision & clean))}
        return result

    return score


def passes(report, baseline):
    if report["_overall"]["clean_fpr"] > CLEAN_FPR_GATE:
        return False
    for name, floor in GATES.items():
        if report[name]["f1"] < floor:
            return False
        if report[name]["f1"] < baseline[name]["f1"] - MAX_FAMILY_DETERIORATION:
            return False
    return True


def compact(report):
    return {name: round(report[name]["f1"], 6) for name in FAMILY_NAMES.values()} | {
        "pooled_f1": round(report["_overall"]["f1"], 6),
        "clean_fpr": report["_overall"]["clean_fpr"],
        "fp": report["_overall"]["fp"], "tp": report["_overall"]["tp"],
        "fn": report["_overall"]["fn"],
        "clean_period_fp": report["_overall"]["clean_period_fp"],
        "worst_family_f1": round(report["_overall"]["worst_family_f1"], 6)}


def fit_verifier(corpus, scores, early, uses_early, seed):
    features, columns = verifier_features(
        corpus["X"], corpus["names"], scores,
        early if uses_early else None, corpus if uses_early else None)
    labels = np.asarray(corpus["labels"]) > 0
    families = np.asarray(corpus["families"])
    clean = (families == 0) & ~labels
    models, pre = {}, {}
    diagnostics = {}
    for branch in BRANCHES:
        inside, threshold = candidate_rows(scores[f"final_{branch}"], clean,
                                           CANDIDATE_QUANTILE)
        target = (labels & (families == BRANCH_FAMILY[branch]))[inside]
        if target.all() or not target.any():
            raise ValueError(f"Candidate region for {branch} has a single class")
        model = HistGradientBoostingClassifier(random_state=seed, **VERIFIER_MODEL)
        model.fit(features[inside], target.astype(int),
                  sample_weight=balanced_weights(target.astype(int)))
        models[branch] = model
        pre[branch] = threshold
        diagnostics[branch] = {"candidate_rows": int(inside.sum()),
                               "candidate_positives": int(target.sum()),
                               "pre_threshold": threshold}
    return (AlarmVerifier(tuple(columns), models, pre, uses_early), columns, diagnostics)


def early_result_from_oof(corpus):
    """Rebuild the graph-time early result from the saved TRAIN OOF probabilities."""
    saved = dict(np.load(EARLY / "train_oof_predictions.npz"))
    group, starts = contiguous_groups(corpus)
    keys = np.column_stack([np.asarray(corpus[key])[starts]
                            for key in ("source", "scenario", "timestep")]).astype(np.int64)
    if not np.array_equal(keys, saved["keys"]):
        raise ValueError("TRAIN OOF graph-time keys differ from the saved early predictions")
    probabilities = saved["probabilities"]
    threshold = json.loads((EARLY / "summary.json").read_text())["abstention"]["selected"]
    return {"probabilities": probabilities, "prediction": probabilities.argmax(1),
            "confidence": probabilities.max(1),
            "abstain": probabilities.max(1) < threshold,
            "group": group, "starts": starts, "keys": keys}


def search(scores, corpus, veto, keeps, score, baseline, cutoff_grid, status_every=0):
    """Best operating point for one candidate: budgeted-OR shares x verifier cutoffs."""
    clean = (np.asarray(corpus["families"]) == 0) & (np.asarray(corpus["labels"]) == 0)
    branch_score = {"mixture": corpus["mixture"],
                    "drift": scores["final_drift"], "noise": scores["final_noise"]}
    best, evaluated, feasible_count = None, 0, 0
    shares = [triple for triple in itertools.product(SHARE_STEPS, repeat=3)
              if abs(sum(triple) - 1.) < 1e-9]
    cache = {}
    for triple in shares:
        thresholds = {}
        for name, share in zip(("mixture", "drift", "noise"), triple):
            key = (name, round(share, 4))
            if key not in cache:
                cache[key] = quantile_threshold(branch_score[name], clean,
                                                CLEAN_BUDGET * share)
            thresholds[name] = cache[key]
        mixture_alarm = branch_score["mixture"] > thresholds["mixture"]
        raw = {branch: branch_score[branch] > thresholds[branch] for branch in BRANCHES}
        for cutoffs in cutoff_grid:
            decision = mixture_alarm.copy()
            for column, branch in enumerate(BRANCHES):
                alarm = raw[branch] & ~veto[:, column]
                if keeps is not None and cutoffs[branch] > 0:
                    alarm = alarm & (keeps[branch] >= cutoffs[branch])
                decision |= alarm
            report = score(decision)
            evaluated += 1
            if not passes(report, baseline):
                continue
            feasible_count += 1
            key = (report["_overall"]["f1"], report["_overall"]["worst_family_f1"])
            if best is None or key > best[0]:
                best = (key, {"budget_shares": list(triple), "thresholds": thresholds,
                              "verifier_cutoffs": dict(cutoffs)}, report)
    return best, {"points_evaluated": evaluated, "points_passing_gates": feasible_count}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / "campaign.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUTPUT / "selection_frozen.json").exists():
        print("Verifier screen already frozen; not re-selecting", flush=True)
        return
    started = time.monotonic()
    veto_rule = json.loads(VETO.read_text())["rule"]
    early_bundle = joblib.load(EARLY / "bundle.joblib")
    head = early_bundle["head"]

    print("fitting verifiers on TRAIN out-of-fold", flush=True)
    train = load_train_oof()
    train_scores = specialist_score_parts(train["causal"], train["delayed"], train)
    train_early = early_result_from_oof(train)
    verifiers, columns, diagnostics = {}, {}, {}
    for name, uses_early in (("C2", False), ("C3", True)):
        verifiers[name], columns[name], diagnostics[name] = fit_verifier(
            train, train_scores, train_early, uses_early, 7300 + len(verifiers))
        print(f"  {name}: {len(columns[name])} columns, {diagnostics[name]}", flush=True)
    del train, train_scores, train_early

    print("scoring calibration", flush=True)
    corpus = load_calibration()
    scores = specialist_score_parts(corpus["causal"], corpus["promoted_head"], corpus)
    veto = specialist_veto_masks(corpus["router"], corpus["feedback"],
                                 veto_rule["margin"], veto_rule["feedback_cutoff"])
    local = early_local_evidence(corpus["X"], corpus["names"], corpus["causal"])
    early = head.predict_groups(local, corpus)
    del local

    keeps = {}
    for name in ("C2", "C3"):
        features, found = verifier_features(
            corpus["X"], corpus["names"], scores,
            early if name == "C3" else None, corpus if name == "C3" else None)
        if list(found) != list(columns[name]):
            raise ValueError(f"{name} verifier column schema differs on calibration")
        keeps[name] = verifiers[name].verify(features, scores)
        del features
    score = fast_scorer(corpus)

    # Paired baseline: the deployed rule, unchanged, on this corpus.
    deployed = veto_rule["base_thresholds"]
    baseline_decision = (corpus["mixture"] > deployed["mixture"])
    for column, branch in enumerate(BRANCHES):
        baseline_decision |= (scores[f"final_{branch}"] > deployed[branch]) & ~veto[:, column]
    baseline = score(baseline_decision)
    print("baseline pooled F1", round(baseline["_overall"]["f1"], 6), flush=True)

    off = [{"drift": 0., "noise": 0.}]
    grid = [{"drift": d, "noise": n} for d in CUTOFFS for n in CUTOFFS]
    results = {}
    for name, keep, cutoff_grid in (("C1", None, off),
                                    ("C2", keeps["C2"], grid),
                                    ("C3", keeps["C3"], grid)):
        print(f"searching {name}", flush=True)
        best, stats = search(scores, corpus, veto, keep, score, baseline, cutoff_grid)
        if best is None:
            results[name] = {"passes_gates": False, **stats,
                             "note": "no operating point satisfied the frozen gates"}
            print(f"  {name}: no feasible point", flush=True)
            continue
        _, rule, report = best
        results[name] = {"passes_gates": True, "rule": rule, "report": compact(report),
                         "delta_vs_baseline": {
                             name_: round(report[name_]["f1"] - baseline[name_]["f1"], 6)
                             for name_ in FAMILY_NAMES.values()} | {
                             "pooled_f1": round(report["_overall"]["f1"]
                                                - baseline["_overall"]["f1"], 6),
                             "fp": report["_overall"]["fp"] - baseline["_overall"]["fp"]},
                         **stats}
        print(f"  {name}: pooled F1 {report['_overall']['f1']:.6f} "
              f"(baseline {baseline['_overall']['f1']:.6f})", flush=True)

    passing = {name: entry for name, entry in results.items() if entry["passes_gates"]}
    winner = max(passing, key=lambda name: passing[name]["report"]["pooled_f1"]) \
        if passing else None

    selection = {
        "status": "selected" if winner else "no_candidate_passed",
        "scope": "Modena calibration, 99 scenarios; TRAIN out-of-fold fitted verifiers",
        "objective": "calibration pooled F1 subject to the frozen protocol gates",
        "gates": {"family_floors": GATES, "clean_fpr": CLEAN_FPR_GATE,
                  "max_family_deterioration": MAX_FAMILY_DETERIORATION},
        "baseline": compact(baseline),
        "candidates": results,
        "selected": winner,
        "verifier_diagnostics": diagnostics,
        "candidate_quantile": CANDIDATE_QUANTILE,
        "verifier_model": VERIFIER_MODEL,
        "early_head_sha256": sha(EARLY / "bundle.joblib"),
        "veto_selection_sha256": sha(VETO),
        "calibration_only": True,
        "eval_evaluated": False, "test_evaluated": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    joblib.dump({"verifiers": verifiers, "columns": columns,
                 "early_head": head, "veto_rule": veto_rule},
                OUTPUT / "bundle.joblib")
    selection["bundle_sha256"] = sha(OUTPUT / "bundle.joblib")
    write_json(OUTPUT / "selection_frozen.json", selection)
    print(json.dumps({"baseline": compact(baseline),
                      "candidates": {k: v.get("report", v) for k, v in results.items()},
                      "selected": winner}, indent=2), flush=True)


if __name__ == "__main__":
    main()
