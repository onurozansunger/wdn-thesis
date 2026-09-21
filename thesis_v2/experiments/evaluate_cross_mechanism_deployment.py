"""Recalibrate on four calibration seeds, then confirm once on locked EVAL-2.

Two changes from the first deployment, both diagnosed on EVAL-1 and both
confirmed on a set that has never been read:

* the weak-family experts may come from the cross-mechanism refit, which is
  trained to stay quiet inside another mechanism's event,
* thresholds are selected on 99 calibration scenarios instead of 27, because
  the replay branch's threshold had been estimated from 254 positives.

Everything else is unchanged: the strong-family mixture, the feature bank, the
declared three-hour decision latency, the 0.005 clean false-alarm budget, the
objective and its constraints, and every generator setting.

Predeclared candidate axes: expert source {promoted, cross_mechanism},
specialist variant {maxpool, blend}, mixture latency {0, 3}, decision rule
{max_evidence, budgeted_or}.

    python3 thesis_v2/experiments/evaluate_cross_mechanism_deployment.py --stage 4
    python3 thesis_v2/experiments/evaluate_cross_mechanism_deployment.py --stage 5
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import json
import time
from itertools import product
from pathlib import Path

import joblib
import lightgbm  # noqa: F401  loaded before scikit-learn's OpenMP
import numpy as np

from wdn.delayed_decision_features import delayed_decision_features
from wdn.latency_deployment import (FAMILY_NAMES, family_scores, forward_max, logit,
                                    logit_blend, quantile_threshold, specialist_bank,
                                    specialist_scores)
from wdn.run_expert_redesign import CampaignData, read_json, sha, write_json

DATA = Path("data/thesis_v2")
ORIGINAL = DATA / "operational_modena_seed811"
CALIBRATION_SEEDS = (12811, 13811, 14811, 15811)
EVAL_SETS = {
    "eval2": {"prefix": "operational_eval2_seed",
              "seeds": (20811, 21811, 22811, 23811, 24811, 25811),
              "report": "eval2_report.json"},
    # A third locked corpus, generated after EVAL-2 was reported. Same
    # generator config apart from the seed; the frozen operating point is
    # applied unchanged, nothing is re-selected.
    "eval3": {"prefix": "operational_eval3_seed",
              "seeds": (30811, 31811, 32811, 33811, 34811, 35811),
              "report": "eval3_report.json"},
}
DETECTOR = Path("runs/operational/expanded_detector_v1/bundle.joblib")
PROMOTED_HEAD = Path("runs/operational/delayed_head_full_v1/bundle.joblib")
CROSS = Path("runs/operational/cross_mechanism_experts_v1/bundle.joblib")
REFERENCE = Path("runs/operational/seasonal_family_deployment_v2/full/reference.joblib")
SPLITS = Path("runs/operational/blind_reference_probe_rank16/splits.json")
OUTPUT = Path("runs/operational/cross_mechanism_deployment_v1")
DELTA = 3
MIXTURE_DELTAS = (0, 3)
EXPERT_SOURCES = ("promoted", "cross_mechanism")
SPECIALIST_VARIANTS = ("maxpool", "blend")
CLEAN_BUDGET = .005
PENALTIES = (0., .5, 1., 1.5, 2.)
CONSTRAINTS = {"replay": .82, "random": .90, "targeted": .90}
TARGET = .80
BRANCHES = ("mixture", "drift", "noise")
POST_EVENT_HOURS = 6


def atomic_npz(path, **arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def post_event_flags(data, arrays):
    flags = {"post_drift": np.zeros(len(arrays["labels"]), dtype=bool),
             "post_noise": np.zeros(len(arrays["labels"]), dtype=bool)}
    scenario = np.asarray(arrays["scenario"])
    timestep = np.asarray(arrays["timestep"])
    quiet = np.asarray(arrays["labels"]) == 0
    for event in data.events:
        key = {"stealthy": "post_drift", "noise": "post_noise"}.get(event["family"])
        if key is None:
            continue
        end = event["start_timestep"] + event["actual_steps"]
        flags[key] |= ((scenario == event["scenario_id"]) & (timestep >= end)
                       & (timestep < end + POST_EVENT_HOURS) & quiet)
    return flags


def score_corpus(pieces, models, status):
    detector, promoted_head, cross, reference = models
    parts = []
    for directory, seed, scenarios in pieces:
        status("scoring corpus", directory=directory.name, scenarios=len(scenarios))
        data = CampaignData(directory)
        arrays, found = specialist_bank(data, scenarios, reference)
        if found != detector["names"]:
            raise ValueError(f"Feature schema differs on {directory.name}")
        entry = {key: np.asarray(arrays[key]) for key in ("labels", "families", "timestep", "node")}
        entry.update(post_event_flags(data, arrays))
        entry["scenario"] = np.asarray(arrays["scenario"]) + seed * 1000
        entry["source"] = np.full(len(entry["labels"]), seed)
        entry["mixture"] = detector["mixture"].predict(arrays["X"])["mixture"]
        entry["promoted"] = specialist_scores(detector, arrays["X"])
        entry["cross_mechanism"] = specialist_scores(cross, arrays["X"])
        indexed = {**arrays, "source": entry["source"], "scenario": entry["scenario"]}
        forward, forward_names = delayed_decision_features(indexed, detector["names"], DELTA)
        if forward_names != promoted_head["forward_names"] or forward_names != cross["forward_names"]:
            raise ValueError("Forward feature schema differs from a fitted head")
        extended = np.column_stack((arrays["X"], forward)).astype(np.float32)
        for label, model in (("promoted_head", promoted_head["delayed"]),
                             ("cross_head", cross["delayed"])):
            predicted = model.predict(extended)
            entry[label] = np.column_stack((predicted["drift"], predicted["noise"]))
        parts.append(entry)
        del data, arrays, forward, extended, indexed
        gc.collect()
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in parts[0]}


def branch_scores(corpus, mixture_delta, source, variant, delta=DELTA):
    raw = corpus[source]
    head = corpus["promoted_head" if source == "promoted" else "cross_head"]
    pooled = {"drift": forward_max(raw[:, 0], corpus, delta),
              "noise": forward_max(raw[:, 1], corpus, delta)}
    if variant == "blend" and delta:
        for column, name in enumerate(("drift", "noise")):
            pooled[name] = logit_blend(head[:, column], pooled[name], .5)
    elif variant not in SPECIALIST_VARIANTS:
        raise ValueError(f"Unknown specialist variant {variant}")
    return {"mixture": forward_max(corpus["mixture"], corpus, mixture_delta), **pooled}


def feasible(report):
    return (report["_overall"]["clean_fpr"] <= CLEAN_BUDGET
            and all(report[name]["f1"] >= floor for name, floor in CONSTRAINTS.items()))


def decide(rule, scores):
    if rule["rule"] == "max_evidence":
        margins = np.column_stack([
            logit(scores[name]) - rule["centers"][name] - (0. if name == "mixture" else rule["penalty"])
            for name in BRANCHES])
        return np.max(margins, axis=1) > rule["threshold"]
    return np.logical_or.reduce([scores[name] > rule["thresholds"][name] for name in BRANCHES])


def search(corpus, status, settings=None, delta=DELTA):
    clean = (corpus["families"] == 0) & (corpus["labels"] == 0)
    grid = settings or list(product(MIXTURE_DELTAS, EXPERT_SOURCES, SPECIALIST_VARIANTS))
    best = None
    for mixture_delta, source, variant in grid:
        scores = branch_scores(corpus, mixture_delta, source, variant, delta)
        logits = {name: logit(scores[name]) for name in BRANCHES}
        centers = {name: float(logit(quantile_threshold(scores[name], clean, CLEAN_BUDGET)))
                   for name in BRANCHES}
        status("searching", mixture_delta=mixture_delta, expert_source=source, variant=variant)
        for penalty in PENALTIES:
            margins = np.max(np.column_stack([
                logits[name] - centers[name] - (0. if name == "mixture" else penalty)
                for name in BRANCHES]), axis=1)
            for threshold in np.unique(np.quantile(margins, np.linspace(.90, 1., 400))):
                report = family_scores(margins > threshold, corpus)
                if not feasible(report):
                    continue
                key = (report["_overall"]["worst_family_f1"], report["_overall"]["f1"])
                if best is None or key > best[0]:
                    best = (key, {"rule": "max_evidence", "mixture_delta": mixture_delta,
                                  "expert_source": source, "specialist_variant": variant,
                                  "specialist_delta": delta, "penalty": float(penalty),
                                  "centers": centers, "threshold": float(threshold)}, report)
        for shares in product([round(.05 * k, 2) for k in range(1, 19)], repeat=3):
            if abs(sum(shares) - 1.) > 1e-9:
                continue
            thresholds = {name: quantile_threshold(scores[name], clean, CLEAN_BUDGET * share)
                          for name, share in zip(BRANCHES, shares)}
            report = family_scores(
                np.logical_or.reduce([scores[name] > thresholds[name] for name in BRANCHES]), corpus)
            if not feasible(report):
                continue
            key = (report["_overall"]["worst_family_f1"], report["_overall"]["f1"])
            if best is None or key > best[0]:
                best = (key, {"rule": "budgeted_or", "mixture_delta": mixture_delta,
                              "expert_source": source, "specialist_variant": variant,
                              "specialist_delta": delta, "budget_shares": list(shares),
                              "thresholds": thresholds}, report)
    if best is None:
        raise RuntimeError("No calibration operating point satisfies the declared constraints")
    return best[1], best[2]


def specialist_point(score, corpus, family_id):
    clean = (corpus["families"] == 0) & (corpus["labels"] == 0)
    scope = corpus["families"] == family_id
    labels = corpus["labels"][scope] > 0
    order = np.argsort(-score[scope], kind="stable")
    ranked, y = score[scope][order], labels[order]
    ends = np.r_[np.flatnonzero(ranked[:-1] != ranked[1:]), len(ranked) - 1]
    predicted, tp = ends + 1, np.cumsum(y)[ends]
    f1 = 2 * tp / (labels.sum() + predicted)
    thresholds = np.nextafter(ranked[ends], np.array(-np.inf, dtype=ranked.dtype))
    clean_sorted = np.sort(score[clean])
    clean_fpr = (len(clean_sorted) - np.searchsorted(clean_sorted, thresholds, side="right")) / len(clean_sorted)
    allowed = np.flatnonzero(clean_fpr <= CLEAN_BUDGET)
    if not len(allowed):
        raise ValueError("No feasible specialist threshold")
    best = allowed[int(np.argmax(f1[allowed]))]
    return {"threshold": float(thresholds[best]), "calibration_f1": float(f1[best]),
            "calibration_clean_fpr": float(clean_fpr[best])}


def bootstrap(decision, corpus, family_id, draws=2000, seed=811):
    scope = corpus["families"] == family_id
    labels = corpus["labels"][scope] > 0
    predicted = decision[scope]
    groups = corpus["scenario"][scope]
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    index = {value: np.flatnonzero(groups == value) for value in unique}
    values = []
    for _ in range(draws):
        picked = np.concatenate([index[value] for value in rng.choice(unique, len(unique))])
        y, p = labels[picked], predicted[picked]
        tp = int(np.sum(y & p)); fp = int(np.sum(~y & p)); fn = int(np.sum(y & ~p))
        values.append(2 * tp / max(1, 2 * tp + fp + fn))
    low, high = np.percentile(values, [2.5, 97.5])
    return {"ci_low": float(low), "ci_high": float(high), "event_scenarios": int(len(unique))}


def expert_report(score, corpus, family_id, point):
    scope = corpus["families"] == family_id
    labels = corpus["labels"] > 0
    decision = score > point["threshold"]
    tp = int(np.sum(labels & decision & scope)); fp = int(np.sum(~labels & decision & scope))
    fn = int(np.sum(labels & ~decision & scope))
    clean = (corpus["families"] == 0) & ~labels
    return {"f1": 2 * tp / max(1, 2 * tp + fp + fn), "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + fn), "tp": tp, "fp": fp, "fn": fn,
            "clean_fpr": float(decision[clean].mean()), "threshold": point["threshold"],
            **bootstrap(decision, corpus, family_id)}


def run(stage, eval_set="eval2"):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / "campaign.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Cross-mechanism deployment run is already active") from error
    started = time.monotonic()

    def status(phase, **details):
        write_json(OUTPUT / "status.json", {"phase": phase, "stage": stage,
            "elapsed_seconds": time.monotonic() - started, **details})
        print(phase, details, flush=True)

    models = (joblib.load(DETECTOR), joblib.load(PROMOTED_HEAD),
              joblib.load(CROSS), joblib.load(REFERENCE))
    splits = read_json(SPLITS)

    if stage == 4:
        if (OUTPUT / "selection_frozen.json").exists():
            print("Selection already frozen; not re-selecting", flush=True)
            return
        pieces = [(ORIGINAL, 811, splits["calibration"])] + [
            (DATA / f"operational_calibration_expansion_seed{seed}", seed, list(range(24)))
            for seed in CALIBRATION_SEEDS]
        corpus = score_corpus(pieces, models, status)
        atomic_npz(OUTPUT / "scores_calibration.npz", **corpus)
        rule, report = search(corpus, status)
        scores = branch_scores(corpus, rule["mixture_delta"], rule["expert_source"],
                               rule["specialist_variant"])
        experts = {"drift": specialist_point(scores["drift"], corpus, 3),
                   "noise": specialist_point(scores["noise"], corpus, 4)}
        status("calibrating the zero-latency control")
        control_rule, control_report = search(
            corpus, status, [(0, source, "maxpool") for source in EXPERT_SOURCES], delta=0)
        control_scores = branch_scores(corpus, 0, control_rule["expert_source"], "maxpool", 0)
        control_experts = {"drift": specialist_point(control_scores["drift"], corpus, 3),
                           "noise": specialist_point(control_scores["noise"], corpus, 4)}
        selection = {"declared_delta_hours": DELTA, "clean_fpr_budget": CLEAN_BUDGET,
            "objective": "max worst-family F1 subject to clean FPR, replay and strong-family floors",
            "constraints": CONSTRAINTS, "selected_rule": rule, "calibration_report": report,
            "specialist_points": experts,
            "zero_latency_control": {"rule": control_rule, "calibration_report": control_report,
                                     "specialist_points": control_experts},
            "calibration_scenarios": {"original": splits["calibration"],
                                      "expansion_seeds": list(CALIBRATION_SEEDS),
                                      "total": len(splits["calibration"]) + 24 * len(CALIBRATION_SEEDS)},
            "detector_sha256": sha(DETECTOR), "promoted_head_sha256": sha(PROMOTED_HEAD),
            "cross_mechanism_sha256": sha(CROSS),
            "eval_sets_evaluated": [], "test_evaluated": False}
        write_json(OUTPUT / "selection_frozen.json", selection)
        print(json.dumps({"rule": {k: v for k, v in rule.items() if k != "centers"},
                          "calibration": {name: round(report[name]["f1"], 4)
                                          for name in FAMILY_NAMES.values()},
                          "overall_f1": round(report["_overall"]["f1"], 4),
                          "clean_fpr": round(report["_overall"]["clean_fpr"], 5),
                          "experts": {k: round(v["calibration_f1"], 4) for k, v in experts.items()},
                          "control": {name: round(control_report[name]["f1"], 4)
                                      for name in FAMILY_NAMES.values()}}, indent=2), flush=True)
        return

    selection = read_json(OUTPUT / "selection_frozen.json")
    spec = EVAL_SETS[eval_set]
    report_path = OUTPUT / spec["report"]
    if report_path.exists():
        print(f"Locked {eval_set} already evaluated once; refusing to repeat", flush=True)
        return
    for seed in spec["seeds"]:
        directory = DATA / f"{spec['prefix']}{seed}"
        config = (directory / "generate_config.yaml").read_text().splitlines()
        reference = (DATA / "operational_modena_seed811" / "generate_config.yaml").read_text().splitlines()
        skip = ("seed:", "output_dir:", "profile:", "num_scenarios:")
        strip = lambda lines: [l for l in lines if not l.startswith(skip)]
        if strip(config) != strip(reference):
            raise ValueError(f"{directory.name} was generated with a different distribution")
    pieces = [(DATA / f"{spec['prefix']}{seed}", seed, list(range(24))) for seed in spec["seeds"]]
    corpus = score_corpus(pieces, models, status)
    atomic_npz(OUTPUT / f"scores_{eval_set}.npz", **corpus)
    rule = selection["selected_rule"]
    control = selection["zero_latency_control"]["rule"]
    result = {"declared_delta_hours": DELTA, "eval_set": eval_set, "eval_seeds": list(spec["seeds"]),
              "selection_sha256": sha(OUTPUT / "selection_frozen.json"),
              "rows": int(len(corpus["labels"])), "families": {}}
    status("evaluating the frozen operating point once")
    for label, active, delta in (("selected", rule, DELTA), ("zero_latency", control, 0)):
        scores = branch_scores(corpus, active["mixture_delta"], active["expert_source"],
                               active["specialist_variant"], delta)
        decision = decide(active, scores)
        report = family_scores(decision, corpus)
        for code, name in FAMILY_NAMES.items():
            report[name].update(bootstrap(decision, corpus, code))
        report["_overall"]["post_event_fp"] = {
            "drift": int(decision[corpus["post_drift"]].sum()),
            "noise": int(decision[corpus["post_noise"]].sum()),
            "drift_points": int(corpus["post_drift"].sum()),
            "noise_points": int(corpus["post_noise"].sum())}
        result["families"][label] = report
    scores = branch_scores(corpus, rule["mixture_delta"], rule["expert_source"],
                           rule["specialist_variant"])
    zero = branch_scores(corpus, 0, control["expert_source"], "maxpool", 0)
    result["experts"] = {
        "selected": {name: expert_report(scores[name], corpus, code,
                                         selection["specialist_points"][name])
                     for name, code in (("drift", 3), ("noise", 4))},
        "zero_latency": {name: expert_report(
            zero[name], corpus, code,
            selection["zero_latency_control"]["specialist_points"][name])
            for name, code in (("drift", 3), ("noise", 4))}}
    result["all_families_at_or_above_target"] = bool(
        all(result["families"]["selected"][name]["f1"] >= TARGET for name in FAMILY_NAMES.values()))
    result["all_experts_at_or_above_target"] = bool(
        all(entry["f1"] >= TARGET for entry in result["experts"]["selected"].values()))
    result["evaluated_once"] = True
    result["test_evaluated"] = False
    write_json(report_path, result)
    print(json.dumps({
        "selected": {name: round(result["families"]["selected"][name]["f1"], 4)
                     for name in FAMILY_NAMES.values()},
        "zero_latency": {name: round(result["families"]["zero_latency"][name]["f1"], 4)
                         for name in FAMILY_NAMES.values()},
        "experts": {k: round(v["f1"], 4) for k, v in result["experts"]["selected"].items()},
        "all_families": result["all_families_at_or_above_target"],
        "all_experts": result["all_experts_at_or_above_target"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=(4, 5), required=True)
    parser.add_argument("--set", choices=tuple(EVAL_SETS), default="eval2")
    args = parser.parse_args()
    run(args.stage, args.set)
