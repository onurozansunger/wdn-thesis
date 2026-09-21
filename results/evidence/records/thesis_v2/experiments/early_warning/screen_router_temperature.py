"""General, family-label-free recalibration of the existing MoE router.

No expert is added or refitted.  The same temperature and uniform shrinkage are
applied symmetrically to all five router outputs.  Selection is on calibration;
completed Stage-E evaluation data are never loaded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from build_feature_cache import write_json
from recalibrate_final_budget import (MAX_PROTECTED_DROP, PROTECTED,
                                      gated_branch_scores)
from run_campaign import (BRANCHES, CAMPAIGN, CLEAN_BUDGET, NETWORKS,
                          TRAINING_SEEDS, compact, fast_scorer,
                          score_corpus_with)
from wdn.latency_deployment import quantile_threshold


OUTPUT = CAMPAIGN / "router_temperature_screen_v1"
TEMPERATURES = (.25, .4, .6, .8, 1., 1.25, 1.6, 2.)
UNIFORM_SHRINKAGE = (0., .05, .10, .20, .35)
MIXTURE_BUDGETS = tuple(np.linspace(.0001, CLEAN_BUDGET, 25))


def transformed_mixture(expert_scores, routing, temperature, shrinkage):
    """Apply one symmetric calibration rule to every router class."""
    power = 1. / float(temperature)
    weight = np.power(np.clip(routing, 1e-12, 1.), power)
    weight /= weight.sum(1, keepdims=True)
    weight = (1. - float(shrinkage)) * weight + float(shrinkage) / weight.shape[1]
    return np.sum(expert_scores * weight, axis=1)


def screen(network, seed):
    target = OUTPUT / network / "seed" / str(seed) / "selection_frozen.json"
    if target.exists():
        return json.loads(target.read_text())
    stage = CAMPAIGN / f"stage_e_{network}" / "seed" / str(seed)
    points = json.loads((stage / "operating_points.json").read_text())
    reference = points["arms"]["candidate"]["report"]
    print(network, seed, "loading calibration and frozen pipeline", flush=True)
    corpus, scores, keep, _early, manifest = score_corpus_with(
        network, seed, NETWORKS[network]["calibration"])
    bundle = joblib.load(stage / "base_bundle.joblib")["mixture"]
    print(network, seed, "scoring existing experts and router", flush=True)
    prediction = bundle.predict(corpus["X"])
    clean = ((np.asarray(corpus["families"]) == 0)
             & (np.asarray(corpus["labels"]) == 0))
    cutoffs = points["arms"]["candidate"]["rule"]["verifier_cutoffs"]
    gated = gated_branch_scores(corpus, scores, keep, cutoffs)
    rule0 = points["arms"]["candidate"]["rule"]
    specialist_alarm = np.zeros(len(clean), dtype=bool)
    for branch in BRANCHES:
        specialist_alarm |= gated[branch] > rule0["thresholds"][branch]
    score = fast_scorer(corpus)
    best = None
    frontier = {"best_pooled": None, "best_worst_family": None}
    evaluated = 0
    for temperature in TEMPERATURES:
        for shrinkage in UNIFORM_SHRINKAGE:
            mixture = transformed_mixture(prediction["experts"], prediction["routing"],
                                          temperature, shrinkage)
            for budget in MIXTURE_BUDGETS:
                threshold = quantile_threshold(mixture, clean, float(budget))
                report = score(specialist_alarm | (mixture > threshold))
                rule = {"router_temperature": temperature,
                        "uniform_shrinkage": shrinkage,
                        "mixture_clean_budget": float(budget),
                        "mixture_threshold": threshold,
                        "specialist_thresholds": {b: rule0["thresholds"][b]
                                                  for b in BRANCHES},
                        "verifier_cutoffs": cutoffs}
                evaluated += 1
                for key, metric in (("best_pooled", "f1"),
                                    ("best_worst_family", "worst_family_f1")):
                    old = frontier[key]
                    value = report["_overall"][metric]
                    if old is None or value > old[0]:
                        frontier[key] = (value, rule, report)
                feasible = (report["_overall"]["clean_fpr"] <= CLEAN_BUDGET
                            and report["_overall"]["f1"] >= reference["pooled_f1"]
                            and all(report[name]["f1"]
                                    >= reference[name] - MAX_PROTECTED_DROP
                                    for name in PROTECTED))
                if not feasible:
                    continue
                key = (report["_overall"]["worst_family_f1"],
                       report["_overall"]["f1"])
                if best is None or key > best[0]:
                    best = (key, rule, report)
    result = {
        "network": network, "training_seed": seed,
        "status": "selected" if best else "no_feasible_improvement",
        "selection_surface": "calibration_only",
        "calibration_scenarios": manifest["total_scenarios"],
        "calibration_rows": manifest["total_rows"],
        "points_evaluated": evaluated,
        "change": "symmetric temperature and uniform shrinkage of existing router",
        "new_expert_added": False, "family_label_used_at_inference": False,
        "evaluation_used_for_selection": False, "locked_test_evaluated": False,
        "reference_stage_e_candidate": reference,
        "frontier": {name: {"rule": value[1], "report": compact(value[2])}
                     for name, value in frontier.items()},
    }
    if best:
        result["rule"], result["report"] = best[1], compact(best[2])
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, result)
    print(network, seed, result["status"], result.get("report"), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", choices=tuple(NETWORKS), required=True)
    parser.add_argument("--seed", type=int, choices=TRAINING_SEEDS, required=True)
    args = parser.parse_args()
    screen(args.network, args.seed)


if __name__ == "__main__":
    main()
