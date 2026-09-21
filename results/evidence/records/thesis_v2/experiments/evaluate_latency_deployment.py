"""Stage 4 calibration selection and Stage 5 locked-EVAL evaluation.

Both stages score the Stage 4a detector, which was refitted on all 110 TRAIN
scenarios with one shared feature bank. Stage 4 builds one operating point on
calibration only — the three original calibration scenarios plus the 24 of the
fresh calibration seed 12811 — and then exits. Stage 5 refuses to start until
that selection is frozen on disk, and evaluates it once on the six locked EVAL
seeds.

Predeclared decision rules:

    max_evidence   one threshold on the penalised maximum of the mixture and
                   specialist margins, penalties 0, 0.5, 1, 1.5, 2
    budgeted_or    separate mixture / drift / noise thresholds whose clean
                   false-alarm budgets sum to 0.005, over a simplex grid

Predeclared latency: the decision for hour t is finalised by t+3. That is a
deadline, not an obligation, so each branch may use less; the mixture branch's
latency is selected on calibration from {0, 3} because a forward maximum helps
a persistent attack and hurts an intermittently labelled one. The drift and
noise branches always use the full three hours.

Objective: the largest minimum family F1 subject to clean FPR <= 0.005,
replay F1 >= 0.82, and random and targeted F1 >= 0.90.

    python3 thesis_v2/experiments/evaluate_latency_deployment.py --stage 4
    python3 thesis_v2/experiments/evaluate_latency_deployment.py --stage 5
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
CALIBRATION_SEED = 12811
EVAL_SEEDS = (4811, 5811, 6811, 7811, 8811, 9811)
DETECTOR = Path("runs/operational/expanded_detector_v1/bundle.joblib")
DELAYED_HEAD = Path("runs/operational/delayed_head_full_v1/bundle.joblib")
REFERENCE = Path("runs/operational/seasonal_family_deployment_v2/full/reference.joblib")
SPLITS = Path("runs/operational/blind_reference_probe_rank16/splits.json")
PLAN = Path("thesis_v2/ALL_FAMILIES_080_PLAN.md")
STAGE1 = Path("runs/operational/delayed_decision_head_v1/summary.json")
DELTA = 3
MIXTURE_DELTAS = (0, 3)
SPECIALIST_VARIANTS = ("maxpool", "blend")
POST_EVENT_HOURS = 6
CLEAN_BUDGET = .005
PENALTIES = (0., .5, 1., 1.5, 2.)
CONSTRAINTS = {"replay": .82, "random": .90, "targeted": .90}
TARGET = .80
BRANCHES = ("mixture", "drift", "noise")


def atomic_npz(path, **arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def post_event_flags(data, arrays, seed):
    """Clean rows within POST_EVENT_HOURS after a weak-family event ended."""
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
        window = ((scenario == event["scenario_id"]) & (timestep >= end)
                  & (timestep < end + POST_EVENT_HOURS) & quiet)
        flags[key] |= window
    return flags


def score_corpus(pieces, bundle, head, reference, status):
    parts = []
    for directory, seed, scenarios in pieces:
        status("scoring corpus", directory=directory.name, scenarios=len(scenarios))
        data = CampaignData(directory)
        arrays, found = specialist_bank(data, scenarios, reference)
        if found != bundle["names"]:
            raise ValueError(f"Feature schema differs on {directory.name}")
        entry = {key: np.asarray(arrays[key]) for key in ("labels", "families", "timestep", "node")}
        entry.update(post_event_flags(data, arrays, seed))
        entry["scenario"] = np.asarray(arrays["scenario"]) + seed * 1000
        entry["source"] = np.full(len(entry["labels"]), seed)
        entry["mixture"] = bundle["mixture"].predict(arrays["X"])["mixture"]
        entry["specialists"] = specialist_scores(bundle, arrays["X"])
        indexed = {**arrays, "source": entry["source"], "scenario": entry["scenario"]}
        forward, forward_names = delayed_decision_features(indexed, bundle["names"], DELTA)
        if forward_names != head["forward_names"]:
            raise ValueError("Forward feature schema differs from the fitted head")
        predicted = head["delayed"].predict(np.column_stack((arrays["X"], forward)).astype(np.float32))
        entry["delayed"] = np.column_stack((predicted["drift"], predicted["noise"]))
        parts.append(entry)
        del data, arrays, forward, indexed
        gc.collect()
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in parts[0]}


def branch_scores(corpus, mixture_delta, variant="maxpool", delta=DELTA):
    pooled = {"drift": forward_max(corpus["specialists"][:, 0], corpus, delta),
              "noise": forward_max(corpus["specialists"][:, 1], corpus, delta)}
    if variant == "blend" and delta:
        for column, name in enumerate(("drift", "noise")):
            pooled[name] = logit_blend(corpus["delayed"][:, column], pooled[name], .5)
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
    """Best feasible operating point over the predeclared candidate set.

    `settings` restricts the (mixture latency, specialist variant) pairs; the
    zero-latency control is the same search with `delta = 0`, so it is a
    properly calibrated control rather than the latency point's thresholds
    applied without their latency.
    """
    clean = (corpus["families"] == 0) & (corpus["labels"] == 0)
    best = None
    for mixture_delta, variant in (settings or list(product(MIXTURE_DELTAS, SPECIALIST_VARIANTS))):
        scores = branch_scores(corpus, mixture_delta, variant, delta)
        logits = {name: logit(scores[name]) for name in BRANCHES}
        centers = {name: float(logit(quantile_threshold(scores[name], clean, CLEAN_BUDGET)))
                   for name in BRANCHES}
        status("searching", mixture_delta=mixture_delta, specialist_variant=variant)
        for penalty in PENALTIES:
            margins = np.max(np.column_stack([
                logits[name] - centers[name] - (0. if name == "mixture" else penalty)
                for name in BRANCHES]), axis=1)
            for threshold in np.unique(np.quantile(margins, np.linspace(.90, 1., 400))):
                rule = {"rule": "max_evidence", "mixture_delta": mixture_delta,
                        "specialist_variant": variant, "specialist_delta": delta,
                        "penalty": float(penalty), "centers": centers,
                        "threshold": float(threshold)}
                report = family_scores(margins > threshold, corpus)
                if not feasible(report):
                    continue
                key = (report["_overall"]["worst_family_f1"], report["_overall"]["f1"])
                if best is None or key > best[0]:
                    best = (key, rule, report)
        steps = [round(.05 * k, 2) for k in range(1, 19)]
        for shares in product(steps, repeat=3):
            if abs(sum(shares) - 1.) > 1e-9:
                continue
            thresholds = {name: quantile_threshold(scores[name], clean, CLEAN_BUDGET * share)
                          for name, share in zip(BRANCHES, shares)}
            decision = np.logical_or.reduce([scores[name] > thresholds[name] for name in BRANCHES])
            report = family_scores(decision, corpus)
            if not feasible(report):
                continue
            key = (report["_overall"]["worst_family_f1"], report["_overall"]["f1"])
            if best is None or key > best[0]:
                best = (key, {"rule": "budgeted_or", "mixture_delta": mixture_delta,
                              "specialist_variant": variant, "specialist_delta": delta,
                              "budget_shares": list(shares), "thresholds": thresholds}, report)
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


def run(stage, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "campaign.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Latency deployment run is already active") from error
    started = time.monotonic()

    def status(phase, **details):
        write_json(output / "status.json", {"phase": phase, "stage": stage,
            "elapsed_seconds": time.monotonic() - started, **details})
        print(phase, details, flush=True)

    if not read_json(STAGE1)["stage1_passed"]:
        raise RuntimeError("Stage 1 did not pass; Stage 4 must not run")
    splits = read_json(SPLITS)
    bundle = joblib.load(DETECTOR)
    head = joblib.load(DELAYED_HEAD)
    reference = joblib.load(REFERENCE)

    if stage == 4:
        if (output / "selection_frozen.json").exists():
            print("Stage 4 selection already frozen; not re-selecting", flush=True)
            return
        pieces = [(ORIGINAL, 811, splits["calibration"]),
                  (DATA / f"operational_calibration_expansion_seed{CALIBRATION_SEED}",
                   CALIBRATION_SEED, list(range(24)))]
        corpus = score_corpus(pieces, bundle, head, reference, status)
        atomic_npz(output / "scores_calibration.npz", **corpus)
        rule, report = search(corpus, status)
        scores = branch_scores(corpus, rule["mixture_delta"], rule["specialist_variant"])
        experts = {"drift": specialist_point(scores["drift"], corpus, 3),
                   "noise": specialist_point(scores["noise"], corpus, 4)}
        status("calibrating the zero-latency control")
        control_rule, control_report = search(corpus, status, [(0, "maxpool")], delta=0)
        control_scores = branch_scores(corpus, 0, "maxpool", 0)
        control_experts = {"drift": specialist_point(control_scores["drift"], corpus, 3),
                           "noise": specialist_point(control_scores["noise"], corpus, 4)}
        selection = {"declared_delta_hours": DELTA, "clean_fpr_budget": CLEAN_BUDGET,
            "objective": "max worst-family F1 subject to clean FPR, replay and strong-family floors",
            "constraints": CONSTRAINTS, "selected_rule": rule, "calibration_report": report,
            "specialist_points": experts,
            "zero_latency_control": {"rule": control_rule, "calibration_report": control_report,
                                     "specialist_points": control_experts},
            "calibration_scenarios": {"original": splits["calibration"],
                                      "expansion_seed": CALIBRATION_SEED},
            "detector_sha256": sha(DETECTOR), "delayed_head_sha256": sha(DELAYED_HEAD),
            "plan_sha256": sha(PLAN),
            "stage1_summary_sha256": sha(STAGE1),
            "locked_eval_evaluated": False, "test_evaluated": False}
        write_json(output / "selection_frozen.json", selection)
        print(json.dumps({"rule": {k: v for k, v in rule.items() if k != "centers"},
                          "calibration": {name: round(report[name]["f1"], 4)
                                          for name in FAMILY_NAMES.values()},
                          "overall_f1": round(report["_overall"]["f1"], 4),
                          "clean_fpr": round(report["_overall"]["clean_fpr"], 5),
                          "experts": {k: round(v["calibration_f1"], 4) for k, v in experts.items()}},
                         indent=2), flush=True)
        return

    selection = read_json(output / "selection_frozen.json")
    if (output / "eval_report.json").exists():
        print("Locked EVAL already evaluated once; refusing to repeat", flush=True)
        return
    pieces = [(DATA / f"operational_eval_seed{seed}", seed, list(range(24))) for seed in EVAL_SEEDS]
    corpus = score_corpus(pieces, bundle, head, reference, status)
    atomic_npz(output / "scores_eval.npz", **corpus)
    rule = selection["selected_rule"]
    result = {"declared_delta_hours": DELTA, "eval_seeds": list(EVAL_SEEDS),
              "selection_sha256": sha(output / "selection_frozen.json"),
              "rows": int(len(corpus["labels"])), "families": {}}
    status("evaluating the frozen operating point once")
    control = selection["zero_latency_control"]["rule"]
    for label, active, delta in (("selected", rule, DELTA), ("zero_latency", control, 0)):
        scores = branch_scores(corpus, active["mixture_delta"], active["specialist_variant"], delta)
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
    scores = branch_scores(corpus, rule["mixture_delta"], rule["specialist_variant"])
    zero = branch_scores(corpus, 0, "maxpool", 0)
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
    write_json(output / "eval_report.json", result)
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
    parser.add_argument("--output-dir", default="runs/operational/latency_deployment_v4")
    args = parser.parse_args()
    run(args.stage, args.output_dir)
