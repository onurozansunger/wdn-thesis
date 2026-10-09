"""Feedback-aware recalibration of the existing detector's final alarm budget.

This experiment does not add or refit an expert.  It fixes an ordering issue in
the Stage-E calibration: branch thresholds were chosen before the router veto
and verifier, so their suppression could leave most of the declared clean-FPR
budget unused.  Here the already-fitted router, feedback and verifier are
applied first, and the three existing branch thresholds are selected on that
final decision surface.

Selection reads TRAIN-fitted models and calibration only.  Stage-E evaluation
reports are read solely by ``snapshot`` to preserve the pre-change result; no
evaluation prediction is an input to calibration.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from build_feature_cache import ROOT, write_json
from run_campaign import (BRANCHES, CAMPAIGN, CLEAN_BUDGET, NETWORKS,
                          SHARE_STEPS, TRAINING_SEEDS, compact, fast_scorer,
                          score_corpus_with)
from wdn.evidence_feedback import specialist_veto_masks
from wdn.latency_deployment import quantile_threshold


# v1 forced the recalibrated surface to consume the full budget.  That first
# bounded screen was negative on L-Town seed 701, because a budget is a ceiling,
# not a spending target.  v2 includes nested utilisation levels and therefore
# contains both conservative and fully-utilised operating points.
OUTPUT = CAMPAIGN / "final_budget_recalibration_v2"
PROTECTED = ("random", "targeted", "drift", "noise")
MAX_PROTECTED_DROP = .01
BUDGET_SCALES = (.10, .25, .50, .75, 1.0)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean(values):
    return float(np.mean(np.asarray(values, dtype=float)))


def snapshot(network: str):
    """Freeze hashes and aggregate metrics for the completed Stage-E reports."""
    reports = []
    files = []
    for seed in TRAINING_SEEDS:
        path = CAMPAIGN / f"stage_e_{network}" / "seed" / str(seed) / "evaluation_report.json"
        if not path.exists():
            raise FileNotFoundError(f"Stage E is incomplete: {path}")
        reports.append(json.loads(path.read_text()))
        files.append({"path": str(path.relative_to(ROOT)), "sha256": file_sha(path)})
    aggregate = {}
    for arm in ("baseline", "threshold", "candidate"):
        rows = [entry["arms"][arm] for report in reports
                for entry in report["per_data_seed"].values()]
        aggregate[arm] = {
            name: {"mean": _mean([row[name] for row in rows]),
                   "minimum": float(min(row[name] for row in rows)),
                   "maximum": float(max(row[name] for row in rows))}
            for name in ("pooled_f1", "random", "replay", "drift", "noise",
                         "targeted", "clean_fpr")}
    result = {"network": network, "status": "stage_e_frozen_before_recalibration",
              "reports": files, "training_seeds": list(TRAINING_SEEDS),
              "data_seeds": list(NETWORKS[network]["eval_seeds"]),
              "aggregate": aggregate, "locked_test_evaluated": False,
              "evaluation_used_for_recalibration_selection": False}
    target = OUTPUT / network / "stage_e_snapshot.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, result)
    return result


def gated_branch_scores(corpus, scores, keep, verifier_cutoffs):
    """Apply existing router/feedback/verifier before threshold calibration."""
    veto = specialist_veto_masks(corpus["router"], corpus["feedback"], .10, .10)
    result = {"mixture": np.asarray(corpus["mixture"], dtype=float)}
    for column, branch in enumerate(BRANCHES):
        allowed = ~veto[:, column]
        cutoff = float(verifier_cutoffs.get(branch, 0.))
        if cutoff > 0:
            allowed &= np.asarray(keep[branch]) >= cutoff
        value = np.asarray(scores[f"final_{branch}"], dtype=float).copy()
        value[~allowed] = -np.inf
        result[branch] = value
    return result


def search_final_budget(branch_scores, clean, score, reference,
                        shares=SHARE_STEPS, budget_scales=BUDGET_SCALES):
    """Search only branch budget shares; all expert/model parameters are frozen."""
    triples = [t for t in itertools.product(shares, repeat=3)
               if abs(sum(t) - 1.) < 1e-9]
    cache = {}
    best = None
    frontier = {"best_pooled": None, "best_worst_family": None}
    for scale in budget_scales:
        for triple in triples:
            thresholds = {}
            decision = np.zeros(len(clean), dtype=bool)
            for name, share in zip(("mixture", "drift", "noise"), triple):
                effective = float(scale) * float(share)
                key = (name, round(effective, 6))
                if key not in cache:
                    cache[key] = quantile_threshold(
                        branch_scores[name], clean, CLEAN_BUDGET * effective)
                thresholds[name] = cache[key]
                decision |= branch_scores[name] > thresholds[name]
            report = score(decision)
            rule = {"budget_scale": float(scale), "budget_shares": list(triple),
                    "thresholds": thresholds}
            if (frontier["best_pooled"] is None
                    or report["_overall"]["f1"]
                    > frontier["best_pooled"]["report"]["_overall"]["f1"]):
                frontier["best_pooled"] = {"rule": rule, "report": report}
            if (frontier["best_worst_family"] is None
                    or report["_overall"]["worst_family_f1"]
                    > frontier["best_worst_family"]["report"]["_overall"]["worst_family_f1"]):
                frontier["best_worst_family"] = {"rule": rule, "report": report}
            feasible = (report["_overall"]["clean_fpr"] <= CLEAN_BUDGET
                        and report["_overall"]["f1"] + 1e-12 >= reference["pooled_f1"]
                        and all(report[name]["f1"] + 1e-12
                                >= reference[name] - MAX_PROTECTED_DROP
                                for name in PROTECTED))
            if not feasible:
                continue
            key = (report["_overall"]["worst_family_f1"],
                   report["_overall"]["f1"])
            if best is None or key > best[0]:
                best = (key, rule, report)
    return best, len(triples) * len(tuple(budget_scales)), frontier


def calibrate(network: str, seed: int):
    target = OUTPUT / network / "seed" / str(seed) / "selection_frozen.json"
    if target.exists():
        return json.loads(target.read_text())
    stage = CAMPAIGN / f"stage_e_{network}" / "seed" / str(seed)
    points_path = stage / "operating_points.json"
    points = json.loads(points_path.read_text())
    reference = points["arms"]["candidate"]["report"]
    print(network, seed, "loading calibration and applying frozen models", flush=True)
    corpus, scores, keep, _early, manifest = score_corpus_with(
        network, seed, NETWORKS[network]["calibration"])
    print(network, seed, "building post-feedback decision surface", flush=True)
    clean = ((np.asarray(corpus["families"]) == 0)
             & (np.asarray(corpus["labels"]) == 0))
    cutoffs = points["arms"]["candidate"]["rule"]["verifier_cutoffs"]
    branch_scores = gated_branch_scores(corpus, scores, keep, cutoffs)
    found, evaluated, frontier = search_final_budget(
        branch_scores, clean, fast_scorer(corpus), reference)
    print(network, seed, "final-budget search complete", flush=True)
    result = {
        "network": network, "training_seed": seed,
        "status": "selected" if found else "no_feasible_improvement",
        "selection_surface": "calibration_only",
        "calibration_scenarios": manifest["total_scenarios"],
        "calibration_rows": manifest["total_rows"],
        "points_evaluated": evaluated,
        "budget_scales": list(BUDGET_SCALES),
        "architecture": "same experts/router/feedback/verifier; thresholds after feedback",
        "reference_stage_e_candidate": reference,
        "reference_operating_points_sha256": file_sha(points_path),
        "constraints": {"clean_fpr": CLEAN_BUDGET,
                        "pooled_f1_not_below_reference": True,
                        "protected_families": list(PROTECTED),
                        "maximum_protected_family_drop": MAX_PROTECTED_DROP},
        "evaluation_used_for_selection": False,
        "locked_test_evaluated": False,
        "frontier": {name: {"rule": value["rule"],
                             "report": compact(value["report"])}
                     for name, value in frontier.items()},
    }
    if found:
        result["rule"] = {**found[1], "verifier_cutoffs": cutoffs,
                          "router_veto_margin": .10,
                          "feedback_veto_cutoff": .10}
        result["report"] = compact(found[2])
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", choices=tuple(NETWORKS), required=True)
    parser.add_argument("--stage", choices=("snapshot", "calibrate", "all"), default="all")
    parser.add_argument("--seed", type=int, choices=TRAINING_SEEDS,
                        help="Run one resumable training seed instead of all five")
    args = parser.parse_args()
    if args.stage in ("snapshot", "all"):
        snapshot(args.network)
    if args.stage in ("calibrate", "all"):
        for seed in ((args.seed,) if args.seed is not None else TRAINING_SEEDS):
            result = calibrate(args.network, seed)
            print(args.network, seed, result["status"],
                  result.get("report", {}).get("pooled_f1"),
                  result.get("report", {}).get("worst_family_f1"), flush=True)


if __name__ == "__main__":
    main()
