"""Paired old-vs-recalibrated comparison on the fresh confirmation corpus.

The recalibrated rules were frozen before these data existed.  This script does
not select or alter a rule; it applies the previously frozen Stage-E candidate
to the exact same rows so the reported delta is paired rather than a comparison
between unrelated generator seeds.
"""
from __future__ import annotations

import argparse
import gc
import json
import math

import numpy as np

from build_feature_cache import write_json
from confirm_router_temperature import DATA_SEEDS, OUTPUT, corpus_name, register
from recalibrate_final_budget import gated_branch_scores
from run_campaign import (BRANCHES, CAMPAIGN, TRAINING_SEEDS, compact,
                          fast_scorer, score_corpus_with)


METRICS = ("pooled_f1", "random", "replay", "drift", "noise", "targeted",
           "clean_fpr")


def evaluate_reference(training_seed):
    target = OUTPUT / "seed" / str(training_seed) / "stage_e_reference_report.json"
    if target.exists():
        return json.loads(target.read_text())
    points_path = (CAMPAIGN / "stage_e_ltown" / "seed" / str(training_seed)
                   / "operating_points.json")
    rule = json.loads(points_path.read_text())["arms"]["candidate"]["rule"]
    per_seed = {}
    for data_seed in DATA_SEEDS:
        register(data_seed)
        print("reference", training_seed, data_seed, flush=True)
        corpus, scores, keep, _early, manifest = score_corpus_with(
            "ltown", training_seed, corpus_name(data_seed))
        gated = gated_branch_scores(corpus, scores, keep, rule["verifier_cutoffs"])
        decision = np.asarray(corpus["mixture"]) > rule["thresholds"]["mixture"]
        for branch in BRANCHES:
            decision |= gated[branch] > rule["thresholds"][branch]
        per_seed[str(data_seed)] = {
            "rows": manifest["total_rows"], "scenarios": manifest["total_scenarios"],
            "metrics": compact(fast_scorer(corpus)(decision))}
        del corpus, scores, keep, gated, decision
        gc.collect()
    result = {"network": "ltown", "training_seed": training_seed,
              "arm": "stage_e_candidate_unchanged",
              "confirmation_generator_seeds": list(DATA_SEEDS),
              "used_for_selection": False, "locked_test_evaluated": False,
              "per_data_seed": per_seed}
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, result)
    return result


def summarise():
    old, new = [], []
    for training_seed in TRAINING_SEEDS:
        a = json.loads((OUTPUT / "seed" / str(training_seed)
                        / "stage_e_reference_report.json").read_text())
        b = json.loads((OUTPUT / "seed" / str(training_seed)
                        / "confirmation_report.json").read_text())
        for data_seed in DATA_SEEDS:
            old.append(a["per_data_seed"][str(data_seed)]["metrics"])
            new.append(b["per_data_seed"][str(data_seed)]["metrics"])
    summary = {}
    rng = np.random.default_rng(20260905)
    for metric in METRICS:
        before = np.asarray([row[metric] for row in old], dtype=float)
        after = np.asarray([row[metric] for row in new], dtype=float)
        delta = after - before
        se = delta.std(ddof=1) / math.sqrt(len(delta))
        matrix = delta.reshape(len(TRAINING_SEEDS), len(DATA_SEEDS))
        boot = np.empty(20000, dtype=float)
        for i in range(len(boot)):
            train_index = rng.integers(0, len(TRAINING_SEEDS), len(TRAINING_SEEDS))
            data_index = rng.integers(0, len(DATA_SEEDS), len(DATA_SEEDS))
            boot[i] = matrix[np.ix_(train_index, data_index)].mean()
        summary[metric] = {
            "reference_mean": float(before.mean()),
            "recalibrated_mean": float(after.mean()),
            "paired_mean_delta": float(delta.mean()),
            "paired_delta_95ci_normal": [float(delta.mean() - 1.96 * se),
                                          float(delta.mean() + 1.96 * se)],
            "paired_delta_95ci_two_way_bootstrap": [
                float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            "improved_pairs": int(np.sum(delta > 0)),
            "equal_pairs": int(np.sum(delta == 0)),
            "worsened_pairs": int(np.sum(delta < 0)),
        }
    result = {"status": "paired_fresh_confirmation_complete", "network": "ltown",
              "pairs": len(old), "training_seeds": list(TRAINING_SEEDS),
              "generator_seeds": list(DATA_SEEDS), "metrics": summary,
              "rules_selected_before_confirmation": True,
              "confirmation_used_for_selection": False,
              "locked_test_evaluated": False}
    write_json(OUTPUT / "paired_summary.json", result)
    lines = ["# Paired Fresh Confirmation", "",
             "The unchanged Stage-E candidate and the frozen symmetric router",
             "recalibration were scored on the same 30 model/data combinations.", "",
             "| Metric | Stage E | Recalibrated | Paired delta | 95% CI | W/T/L |",
             "|---|---:|---:|---:|---:|---:|"]
    for metric, value in summary.items():
        lo, hi = value["paired_delta_95ci_two_way_bootstrap"]
        lines.append(
            f"| {metric} | {value['reference_mean']:.4f} | "
            f"{value['recalibrated_mean']:.4f} | {value['paired_mean_delta']:+.4f} | "
            f"[{lo:+.4f}, {hi:+.4f}] | {value['improved_pairs']}/"
            f"{value['equal_pairs']}/{value['worsened_pairs']} |")
    (OUTPUT / "PAIRED_REPORT.md").write_text("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=TRAINING_SEEDS)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not args.report_only:
        seeds = (args.seed,) if args.seed is not None else TRAINING_SEEDS
        for seed in seeds:
            evaluate_reference(seed)
    if args.seed is None:
        summarise()


if __name__ == "__main__":
    main()
