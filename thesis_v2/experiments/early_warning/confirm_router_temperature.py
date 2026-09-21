"""Fresh confirmation of the frozen, symmetric router recalibration.

The six generator seeds in this file were reserved before generation and were
not used by the Stage-E evaluation.  This script never selects a rule: it only
applies each training seed's already-frozen calibration rule.
"""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np

from build_feature_cache import CORPORA, ROOT, build, write_json
from recalibrate_final_budget import gated_branch_scores
from run_campaign import (BRANCHES, CAMPAIGN, TRAINING_SEEDS, compact,
                          fast_scorer, reference_path, score_corpus_with)
from screen_router_temperature import OUTPUT as SELECTION, transformed_mixture


PURPOSE = "ltown_router_confirmation"
DATA_SEEDS = (100811, 101811, 102811, 103811, 104811, 105811)
OUTPUT = CAMPAIGN / "router_temperature_confirmation_v1" / "ltown"


def corpus_name(seed):
    return f"ltown_router_confirmation_seed{seed}"


def register(seed):
    name = f"ew_ltown_{PURPOSE}_seed{seed}"
    CORPORA[corpus_name(seed)] = {
        "reference": reference_path("ltown"), "network": "ltown",
        "pieces": [(name, seed, "all:24")],
    }
    return name


def prepare_data():
    for seed in DATA_SEEDS:
        name = register(seed)
        directory = ROOT / "data/thesis_v2" / name
        if not directory.exists():
            subprocess.run([
                sys.executable, str(Path(__file__).with_name("generate_corpus.py")),
                "--network", "ltown", "--purpose", PURPOSE,
                "--seed", str(seed), "--scenarios", "24"], cwd=ROOT, check=True)
        manifest = CAMPAIGN / "features" / corpus_name(seed) / "manifest.json"
        if not manifest.exists():
            build(corpus_name(seed))
    write_json(OUTPUT / "data_ready.json", {
        "purpose": PURPOSE, "generator_seeds": list(DATA_SEEDS),
        "scenarios_per_seed": 24, "reserved_before_generation": True,
        "selection_used_these_seeds": False,
        "pressure_missing_probability": .5, "flow_missing_probability": .5})


def evaluate(training_seed):
    target = OUTPUT / "seed" / str(training_seed) / "confirmation_report.json"
    if target.exists():
        return json.loads(target.read_text())
    selection_path = (SELECTION / "ltown" / "seed" / str(training_seed)
                      / "selection_frozen.json")
    selection = json.loads(selection_path.read_text())
    if selection["status"] != "selected":
        raise RuntimeError(f"No frozen selected rule for seed {training_seed}")
    rule = selection["rule"]
    stage = CAMPAIGN / "stage_e_ltown" / "seed" / str(training_seed)
    per_seed = {}
    for data_seed in DATA_SEEDS:
        register(data_seed)
        print("confirm", training_seed, data_seed, flush=True)
        corpus, scores, keep, _early, manifest = score_corpus_with(
            "ltown", training_seed, corpus_name(data_seed))
        mixture_model = joblib.load(stage / "base_bundle.joblib")["mixture"]
        prediction = mixture_model.predict(corpus["X"])
        mixture = transformed_mixture(
            prediction["experts"], prediction["routing"],
            rule["router_temperature"], rule["uniform_shrinkage"])
        gated = gated_branch_scores(corpus, scores, keep, rule["verifier_cutoffs"])
        decision = mixture > rule["mixture_threshold"]
        for branch in BRANCHES:
            decision |= gated[branch] > rule["specialist_thresholds"][branch]
        per_seed[str(data_seed)] = {
            "rows": manifest["total_rows"], "scenarios": manifest["total_scenarios"],
            "metrics": compact(fast_scorer(corpus)(decision))}
        del corpus, scores, keep, prediction, mixture, gated, decision, mixture_model
        gc.collect()
    result = {"network": "ltown", "training_seed": training_seed,
              "selection_rule_path": str(selection_path.relative_to(ROOT)),
              "selection_surface": "calibration_only",
              "confirmation_generator_seeds": list(DATA_SEEDS),
              "per_data_seed": per_seed, "locked_test_evaluated": False}
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, result)
    return result


def report():
    reports = [json.loads((OUTPUT / "seed" / str(seed)
                          / "confirmation_report.json").read_text())
               for seed in TRAINING_SEEDS]
    rows = [entry["metrics"] for item in reports
            for entry in item["per_data_seed"].values()]
    metrics = {}
    for name in ("pooled_f1", "random", "replay", "drift", "noise",
                 "targeted", "clean_fpr"):
        values = np.asarray([row[name] for row in rows], dtype=float)
        metrics[name] = {"mean": float(values.mean()),
                         "std": float(values.std(ddof=1)),
                         "minimum": float(values.min()),
                         "maximum": float(values.max()),
                         "below_0_8": int(np.sum(values < .8))}
    result = {"status": "fresh_confirmation_complete", "network": "ltown",
              "training_seeds": list(TRAINING_SEEDS),
              "generator_seeds": list(DATA_SEEDS), "comparisons": len(rows),
              "metrics": metrics, "selection_used_confirmation": False,
              "locked_test_evaluated": False}
    write_json(OUTPUT / "summary.json", result)
    lines = ["# Symmetric Router Recalibration — Fresh L-Town Confirmation", "",
             "No expert was added. Rules were frozen on calibration before these six",
             "generator seeds were produced or scored. The original locked test was not read.", "",
             "| Metric | Mean | SD | Min | Max | < 0.80 |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, value in metrics.items():
        lines.append(f"| {name} | {value['mean']:.4f} | {value['std']:.4f} | "
                     f"{value['minimum']:.4f} | {value['maximum']:.4f} | "
                     f"{value['below_0_8']} |")
    (OUTPUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("data", "evaluate", "report", "all"),
                        default="all")
    parser.add_argument("--seed", type=int, choices=TRAINING_SEEDS)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("data", "all"):
        prepare_data()
    if args.stage in ("evaluate", "all"):
        seeds = (args.seed,) if args.seed is not None else TRAINING_SEEDS
        for seed in seeds:
            evaluate(seed)
    if args.stage in ("report", "all"):
        report()


if __name__ == "__main__":
    main()
