"""Stage E: resumable repeated-fit campaign, one network at a time.

Structure, and what varies within it:

* The **normal reference** and the **feature bank** are deterministic given a
  scenario set, so they are built once per network and shared by every training
  seed. Nothing is randomised here to manufacture spread.
* The **classifiers** are refitted per training seed: the mixture, the tuned
  drift tree, the seasonal specialists, the delayed head, the early-warning head
  and the verifier all take that seed.
* A **baseline and its candidate share the same fitted base experts** inside a
  seed pair. The candidate adds a decision layer; refitting the base twice would
  be waste, not evidence.

Three arms are scored per seed, because the candidate changes both the
architecture and the calibration objective and those must be separable:

* ``baseline``  — the deployed guarded hybrid, calibrated by its own recorded
  recipe (budgeted OR, worst-family objective, deployed constraints),
* ``threshold`` — the same architecture calibrated for pooled F1 (Stage C's C1),
* ``candidate`` — verifier plus early-warning signal (Stage C's C3).

Stages, each resumable and each refusing an incompatible cached configuration:

    folds      per-fold source-held references and features (once per network)
    oof        per-seed source-held expert fits, producing honest OOF scores
    fit        per-seed full-TRAIN fits
    heads      per-seed early-warning head and verifier
    calibrate  per-seed operating points, frozen before any evaluation
    evaldata   generate and cache the fresh evaluation corpora
    evaluate   score the fresh corpora with the already-frozen rules

    /opt/miniconda3/bin/python thesis_v2/experiments/early_warning/run_campaign.py \
        --network modena --stage all
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import itertools
import json
import time
from pathlib import Path

import joblib
import lightgbm  # noqa: F401  loaded before scikit-learn's OpenMP
import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier

from wdn.alarm_verifier import BRANCHES, verifier_features
from wdn.delayed_decision_features import delayed_decision_features
from wdn.early_warning import early_local_evidence
from wdn.evidence_feedback import (FeedbackRouterBundle, ROUTER_CLASSES,
                                   SCORE_EVIDENCE_NAMES, aggregate_router_evidence,
                                   balanced_weights, contiguous_groups, group_values,
                                   local_evidence, router_targets,
                                   specialist_veto_masks)
from wdn.latency_deployment import (FAMILY_NAMES, quantile_threshold,
                                    specialist_bank, specialist_scores)
from wdn.models.delayed_decision import DelayedDecisionExperts
from wdn.models.seasonal_family import SeasonalFamilyExperts
from wdn.models.tuned_family_tree import TunedFamilyTreeConfig, TunedFamilyTrees
from wdn.probe_residual_experts import ResidualExpertMixture
from wdn.run_expert_redesign import CampaignData

from build_feature_cache import (CORPORA, ROOT, build, globalise_events,
                                 load_corpus, sha, atomic_npz, write_json)
from fit_early_warning import (ABSTAIN_GRID, CLEAN_WARNING_BUDGET, MODEL as EARLY_MODEL,
                               build_surface, choose_abstention, early_phase_weights,
                               fit_model as fit_early_model)
from screen_verifier import (CLEAN_BUDGET, CUTOFFS, MAX_FAMILY_DETERIORATION,
                             SHARE_STEPS, compact, fast_scorer, fit_verifier,
                             passes, specialist_score_parts)
from campaign_data import events_by_scenario

CAMPAIGN = ROOT / "runs/operational/early_warning_multiseed_v1"
DELTA = 3
TRAINING_SEEDS = (701, 702, 703, 704, 705)

#: The deployed baseline's own recorded recipe, reproduced rather than reused.
BASELINE_CONSTRAINTS = {"replay": .82, "random": .90, "targeted": .90}

NETWORKS = {
    "modena": {
        "train": "modena_train", "calibration": "modena_calibration",
        "eval_seeds": (40811, 41811, 42811, 43811, 44811, 45811),
        "eval_purpose": "modena_eval", "inp": "data/modena.inp",
    },
    "ltown": {
        "train": "ltown_train", "calibration": "ltown_calibration",
        "eval_seeds": (50811, 51811, 52811, 53811, 54811, 55811),
        "eval_purpose": "ltown_eval", "inp": "data/L-Town.inp",
    },
}


def out(network, *parts):
    path = CAMPAIGN / f"stage_e_{network}"
    for part in parts:
        path = path / str(part)
    return path


def status(network, phase, **details):
    path = out(network)
    path.mkdir(parents=True, exist_ok=True)
    write_json(path / "status.json", {"phase": phase, "network": network,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "locked_test_evaluated": False, "locked_eval_seeds_read": False, **details})
    print(f"[{network}] {phase} {details}", flush=True)


# --------------------------------------------------------------------------
# folds: source-held references and features, deterministic and shared
# --------------------------------------------------------------------------

def stage_folds(network):
    spec = NETWORKS[network]
    corpus_name = spec["train"]
    manifest = json.loads((CAMPAIGN / "features" / corpus_name / "manifest.json").read_text())
    sources = [piece["seed"] for piece in manifest["pieces"]]
    folder = out(network, "folds")
    folder.mkdir(parents=True, exist_ok=True)
    signature = {"corpus": corpus_name, "sources": sources,
                 "reference_recipe": "BlindPressureReference(rank=16) + RobustBlindReference",
                 "note": "one held-out generator source per fold"}
    path = folder / "signature.json"
    if path.exists() and json.loads(path.read_text()) != signature:
        raise SystemExit(f"Cached fold configuration for {network} is incompatible; "
                         "refusing to overwrite it")
    write_json(path, signature)

    from wdn.models.blind_reference import BlindPressureReference
    from wdn.models.robust_reference import RobustBlindReference

    for fold, held in enumerate(sources):
        target = folder / f"fold_{fold}"
        target.mkdir(exist_ok=True)
        if (target / "features_held_out.npz").exists():
            continue
        status(network, "building source-held fold", fold=fold, held_source=held)
        fit_sources = [s for s in sources if s != held]
        values, masks = [], []
        for piece in manifest["pieces"]:
            if piece["seed"] not in fit_sources:
                continue
            # The feature cache holds derived columns, not raw readings, so the
            # source corpus is reopened to fit a reference on normal rows only.
            data = CampaignData(ROOT / "data/thesis_v2" / piece["directory"])
            for sid in piece["scenarios"]:
                arrays = data.scenario(sid)
                normal = arrays["families"] == 0
                values.append(arrays["values"][normal])
                masks.append(arrays["mask"][normal])
            del data
            gc.collect()
        reference = RobustBlindReference(
            BlindPressureReference(rank=16).fit(np.concatenate(values),
                                                np.concatenate(masks)))
        reference.calibrate_scale(np.concatenate(values), np.concatenate(masks))
        joblib.dump(reference, target / "reference.joblib")
        del values, masks
        gc.collect()

        for split, wanted in (("train", fit_sources), ("held_out", [held])):
            parts = []
            for piece in manifest["pieces"]:
                if piece["seed"] not in wanted:
                    continue
                data = CampaignData(ROOT / "data/thesis_v2" / piece["directory"])
                arrays, _ = specialist_bank(data, piece["scenarios"], reference)
                rows = len(arrays["labels"])
                parts.append({
                    "X": np.asarray(arrays["X"], np.float32),
                    "labels": np.asarray(arrays["labels"], np.int8),
                    "families": np.asarray(arrays["families"], np.int8),
                    # Raw per-source event ids, as the shared feature caches also
                    # store them; both are globalised when they are loaded.
                    "event": np.asarray(arrays["event"], np.int32),
                    "scenario": np.asarray(arrays["scenario"], np.int64) + piece["seed"] * 1000,
                    "source": np.full(rows, piece["seed"], np.int64),
                    "timestep": np.asarray(arrays["timestep"], np.int32),
                    "node": np.asarray(arrays["node"], np.int32)})
                del data, arrays
                gc.collect()
            merged = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
            atomic_npz(target / f"features_{split}.npz", **merged)
            del parts, merged
            gc.collect()
    status(network, "folds complete", folds=len(sources))


# --------------------------------------------------------------------------
# expert fitting helpers, shared by the OOF and full-TRAIN stages
# --------------------------------------------------------------------------

def add_early_flag(arrays, events):
    """Mark the first three hours of every event, as the expert weighting expects.

    ``_balanced_weights`` upweights these rows. The flag is derived from the
    event ledger, which is evaluator metadata used for supervised training only;
    it is never a feature and never reaches inference.
    """
    scenario = np.asarray(arrays["scenario"])
    timestep = np.asarray(arrays["timestep"])
    early = np.zeros(len(timestep), dtype=bool)
    present = set(np.unique(scenario).tolist())
    for sid, entries in events.items():
        if sid not in present:
            continue
        here = scenario == sid
        for event in entries:
            early |= here & (timestep >= event["start"]) & (timestep < event["start"] + 3)
    return {**arrays, "early": early}


def fit_experts(train, bank, base_names, seed):
    mixture = ResidualExpertMixture(bank, seed=600 + seed).fit(
        train["X"], train["labels"], train["families"])
    tuned = TunedFamilyTrees(base_names, TunedFamilyTreeConfig(
        max_leaf_nodes=23, min_samples_leaf=50, learning_rate=.1,
        l2_regularization=5., max_iter=150, seed=2600 + seed)).fit(
        {**train, "X": train["X"][:, :len(base_names)]})
    seasonal = SeasonalFamilyExperts(bank, seed=4100 + seed).fit(train)
    return {"mixture": mixture, "seasonal": seasonal, "tuned_drift": tuned,
            "names": bank, "base_feature_count": len(base_names),
            "score_blends": {"drift_seasonal": .90, "noise_fast": .95}}


def fit_delayed(train, bank, seed):
    forward, forward_names = delayed_decision_features(train, bank, DELTA)
    extended = {**train, "X": np.column_stack((train["X"], forward)).astype(np.float32)}
    names = list(bank) + list(forward_names)
    model = DelayedDecisionExperts(names, DELTA, seed=5100 + seed).fit(extended)
    del extended, forward
    gc.collect()
    return {"delayed": model, "names": names, "bank": list(bank),
            "forward_names": list(forward_names), "delta": DELTA}


def apply_experts(bundle, head, arrays):
    """Causal and delayed specialist score channels for one corpus."""
    causal = specialist_scores(bundle, arrays["X"])
    forward, forward_names = delayed_decision_features(arrays, bundle["names"], DELTA)
    if forward_names != head["forward_names"]:
        raise ValueError("Forward feature schema differs from the fitted head")
    predicted = head["delayed"].predict(
        np.column_stack((arrays["X"], forward)).astype(np.float32))
    del forward
    gc.collect()
    delayed = np.column_stack((predicted["drift"], predicted["noise"]))
    return causal, delayed


def stage_oof(network, seed):
    folder = out(network, "seed", seed)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "oof_scores.npz"
    if target.exists():
        return
    folds = sorted((out(network, "folds")).glob("fold_*"))
    bank = json.loads((CAMPAIGN / "features" / NETWORKS[network]["train"]
                       / "manifest.json").read_text())["feature_names"]
    base_names = bank[:109]
    train_events = events_for(network, NETWORKS[network]["train"])
    parts = []
    for fold, directory in enumerate(folds):
        status(network, "fitting source-held experts", seed=seed, fold=fold)
        train = dict(np.load(directory / "features_train.npz"))
        train["event"] = globalise_events(train["event"], train["source"])
        train = add_early_flag(train, train_events)
        experts = fit_experts(train, bank, base_names, seed)
        head = fit_delayed(train, bank, seed)
        del train
        gc.collect()
        held = dict(np.load(directory / "features_held_out.npz"))
        held["event"] = globalise_events(held["event"], held["source"])
        causal, delayed = apply_experts(experts, head, held)
        mixture = experts["mixture"].predict(held["X"])["mixture"]
        parts.append({key: held[key] for key in
                      ("labels", "families", "event", "scenario", "source",
                       "timestep", "node")}
                     | {"causal": causal, "delayed": delayed, "mixture": mixture})
        del held, experts, head, causal, delayed
        gc.collect()
    merged = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    atomic_npz(target, **merged)
    status(network, "oof complete", seed=seed, rows=int(len(merged["labels"])))


def stage_fit(network, seed):
    folder = out(network, "seed", seed)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "base_bundle.joblib").exists():
        return
    status(network, "fitting full-TRAIN experts", seed=seed)
    train, manifest = load_corpus(NETWORKS[network]["train"])
    bank = manifest["feature_names"]
    base_names = bank[:109]
    train.pop("_feature_names", None)
    train = add_early_flag(train, events_for(network, NETWORKS[network]["train"]))
    experts = fit_experts(train, bank, base_names, seed)
    joblib.dump(experts, folder / "base_bundle.joblib")
    head = fit_delayed(train, bank, seed)
    joblib.dump(head, folder / "delayed_bundle.joblib")
    write_json(folder / "fit_summary.json", {
        "training_seed": seed, "network": network,
        "train_scenarios": manifest["total_scenarios"],
        "train_rows": manifest["total_rows"],
        "reference_sha256": manifest["reference_sha256"],
        "reference_is_deterministic_across_training_seeds": True,
        "base_bundle_sha256": sha(folder / "base_bundle.joblib"),
        "delayed_bundle_sha256": sha(folder / "delayed_bundle.joblib")})
    del train, experts, head
    gc.collect()
    status(network, "full-TRAIN fit complete", seed=seed)


def router_score_parts(scores):
    """Rename this campaign's score channels to the router's declared vocabulary."""
    parts = {
        "frozen_drift": scores["causal_drift"], "frozen_noise": scores["causal_noise"],
        "maxpool_drift": scores["maxpool_drift"], "maxpool_noise": scores["maxpool_noise"],
        "delayed_drift": scores["delayed_drift"], "delayed_noise": scores["delayed_noise"],
        "final_drift": scores["final_drift"], "final_noise": scores["final_noise"]}
    if tuple(parts) != SCORE_EVIDENCE_NAMES:
        raise AssertionError("Score evidence order changed")
    return parts


def feedback_sample(labels, families, base_score, family_id, rng):
    """The deployed feedback head's training sample, reproduced verbatim.

    Hard negatives are the highest-scoring negatives; replay negatives are
    sampled separately because replay is the mechanism the guard exists to
    separate. Mining stays inside whatever corpus is passed in, which is TRAIN.
    """
    labels = np.asarray(labels) > 0
    families = np.asarray(families)
    positive = np.flatnonzero(labels & (families == family_id))
    other_positive = np.flatnonzero(labels & (families != family_id))
    negative = np.flatnonzero(~labels)
    order = negative[np.argsort(-np.asarray(base_score)[negative], kind="stable")]
    hard = order[:min(60000, len(order))]
    replay_negative = np.flatnonzero((~labels) & (families == 2))
    if len(replay_negative) > 30000:
        replay_negative = rng.choice(replay_negative, 30000, replace=False)
    remaining = np.setdiff1d(negative, np.union1d(hard, replay_negative))
    random_negative = rng.choice(remaining, min(40000, len(remaining)), replace=False)
    return np.unique(np.r_[positive, other_positive, hard, replay_negative,
                           random_negative])


def _router_model(seed):
    return HistGradientBoostingClassifier(max_iter=180, max_leaf_nodes=23,
        min_samples_leaf=30, learning_rate=.06, l2_regularization=12.,
        early_stopping=False, random_state=seed)


def fit_feedback_router(corpus, scores, seed):
    """Refit the deployed guard's router and feedback heads for this training seed.

    The baseline arm is the *guarded* hybrid, so its guard has to be refitted
    alongside its experts rather than borrowed from the recorded deployment,
    which was fitted to different expert scores.
    """
    local = local_evidence(corpus["X"], corpus["names"], router_score_parts(scores))
    group, starts = contiguous_groups(corpus)
    aggregate = aggregate_router_evidence(local, group, starts)
    target = router_targets(group_values(corpus["families"], starts, "family"))
    group_source = group_values(corpus["source"], starts, "source")

    sources = np.unique(group_source)
    if len(sources) < 2:
        raise ValueError(
            "The guard's router needs at least two generator sources to hold one "
            f"out; this corpus has {len(sources)}")
    oof = np.zeros((len(starts), len(ROUTER_CLASSES)), dtype=np.float32)
    for fold, source in enumerate(sources):
        train = group_source != source
        model = _router_model(6100 + seed * 10 + fold)
        model.fit(aggregate[train], target[train],
                  sample_weight=balanced_weights(target[train]))
        oof[~train] = model.predict_proba(aggregate[~train])
    router = _router_model(6200 + seed)
    router.fit(aggregate, target, sample_weight=balanced_weights(target))

    feedback_X = np.column_stack((local, oof[group])).astype(np.float32)
    rng = np.random.default_rng(6300 + seed)
    heads = []
    for column, (name, family_id) in enumerate((("drift", 3), ("noise", 4))):
        selected = feedback_sample(corpus["labels"], corpus["families"],
                                   scores[f"final_{name}"], family_id, rng)
        y = ((np.asarray(corpus["labels"])[selected] > 0)
             & (np.asarray(corpus["families"])[selected] == family_id)).astype(int)
        fitted = _router_model(6400 + seed * 10 + column)
        fitted.fit(feedback_X[selected], y, sample_weight=balanced_weights(y))
        heads.append(fitted)
    return FeedbackRouterBundle(tuple(corpus["names"]), router, heads[0], heads[1])


def stage_heads(network, seed):
    folder = out(network, "seed", seed)
    if (folder / "heads_bundle.joblib").exists():
        return
    status(network, "fitting early head and verifier", seed=seed)
    oof = dict(np.load(folder / "oof_scores.npz"))
    bank = json.loads((CAMPAIGN / "features" / NETWORKS[network]["train"]
                       / "manifest.json").read_text())["feature_names"]
    fold_features = []
    for directory in sorted((out(network, "folds")).glob("fold_*")):
        with np.load(directory / "features_held_out.npz") as loaded:
            fold_features.append(loaded["X"])
    corpus = {**oof, "X": np.concatenate(fold_features), "names": bank}
    del fold_features
    gc.collect()

    surface = build_surface(corpus)
    events = events_for(network, NETWORKS[network]["train"])
    events = {k: v for k, v in events.items() if k in set(surface["scenario"].tolist())}
    weight, _ = early_phase_weights(surface, events)
    sources = np.unique(surface["source"])
    early_oof = np.zeros((len(surface["target"]), 5))
    for fold, source in enumerate(sources):
        held = surface["source"] == source
        model = fit_early_model(surface["features"][~held], surface["target"][~held],
                                weight[~held], 7100 + seed * 10 + fold)
        early_oof[held] = model.predict_proba(surface["features"][held])
    threshold, grid, counts = choose_abstention(early_oof, surface["target"],
                                                surface["keys"], events)
    final = fit_early_model(surface["features"], surface["target"], weight, 7200 + seed)
    from wdn.early_warning import EarlyWarningHead
    head = EarlyWarningHead(tuple(surface["names"]), final, threshold)

    group, starts = contiguous_groups(corpus)
    early_result = {"probabilities": early_oof, "prediction": early_oof.argmax(1),
                    "confidence": early_oof.max(1),
                    "abstain": early_oof.max(1) < threshold,
                    "group": group, "starts": starts, "keys": surface["keys"]}
    scores = specialist_score_parts(corpus["causal"], corpus["delayed"], corpus)
    verifier, columns, diagnostics = fit_verifier(corpus, scores, early_result,
                                                  True, 7300 + seed)
    status(network, "refitting the baseline guard for this seed", seed=seed)
    guard = fit_feedback_router(corpus, scores, seed)
    joblib.dump({"early": head, "verifier": verifier, "columns": columns,
                 "abstain_threshold": threshold, "guard": guard},
                folder / "heads_bundle.joblib")
    write_json(folder / "heads_summary.json", {
        "training_seed": seed, "network": network,
        "graph_time_rows": int(len(surface["starts"])),
        "early_features": len(surface["names"]),
        "early_model": EARLY_MODEL,
        "abstention": {"grid": list(ABSTAIN_GRID), "budget": CLEAN_WARNING_BUDGET,
                       "selected": threshold, "sweep": grid, **counts},
        "verifier_diagnostics": diagnostics,
        "heads_bundle_sha256": sha(folder / "heads_bundle.joblib")})
    del corpus, surface, oof
    gc.collect()
    status(network, "heads complete", seed=seed, abstain_threshold=threshold)


def events_for(network, corpus_name):
    manifest = json.loads((CAMPAIGN / "features" / corpus_name / "manifest.json").read_text())
    return events_by_scenario([(piece["directory"], piece["seed"])
                               for piece in manifest["pieces"]])


# --------------------------------------------------------------------------
# calibrate: freeze one operating point per arm, per seed
# --------------------------------------------------------------------------

def score_corpus_with(network, seed, corpus_name):
    """Every branch score this campaign needs, for one corpus and one seed."""
    folder = out(network, "seed", seed)
    experts = joblib.load(folder / "base_bundle.joblib")
    head = joblib.load(folder / "delayed_bundle.joblib")
    heads = joblib.load(folder / "heads_bundle.joblib")
    corpus, manifest = load_corpus(corpus_name)
    corpus.pop("_feature_names", None)
    corpus["names"] = manifest["feature_names"]
    causal, delayed = apply_experts(experts, head, corpus)
    corpus["causal"], corpus["delayed"] = causal, delayed
    corpus["mixture"] = experts["mixture"].predict(corpus["X"])["mixture"]
    scores = specialist_score_parts(causal, delayed, corpus)
    local = early_local_evidence(corpus["X"], corpus["names"], causal)
    early = heads["early"].predict_groups(local, corpus)
    del local
    gc.collect()
    features, found = verifier_features(corpus["X"], corpus["names"], scores,
                                        early, corpus)
    if list(found) != list(heads["columns"]):
        raise ValueError("Verifier column schema differs on this corpus")
    keep = heads["verifier"].verify(features, scores)
    del features
    gc.collect()
    guarded = heads["guard"].predict(
        local_evidence(corpus["X"], corpus["names"], router_score_parts(scores)), corpus)
    corpus["router"], corpus["feedback"] = guarded["router"], guarded["feedback"]
    return corpus, scores, keep, early, manifest


def search_arm(scores, corpus, keep, score, arm, baseline_report=None):
    clean = (np.asarray(corpus["families"]) == 0) & (np.asarray(corpus["labels"]) == 0)
    branch_score = {"mixture": corpus["mixture"],
                    "drift": scores["final_drift"], "noise": scores["final_noise"]}
    veto = np.zeros((len(corpus["labels"]), 2), dtype=bool)
    if "router" in corpus:
        # The guard's router and feedback heads are refitted per training seed,
        # but its (margin, cutoff) = (0.10, 0.10) rule is the recorded deployed
        # one and is held fixed: re-selecting it per seed would change the
        # baseline recipe rather than repeat it.
        veto = specialist_veto_masks(corpus["router"], corpus["feedback"], .10, .10)
    shares = [t for t in itertools.product(SHARE_STEPS, repeat=3)
              if abs(sum(t) - 1.) < 1e-9]
    uses_verifier = arm.startswith("candidate")
    cutoff_grid = ([{"drift": d, "noise": n} for d in CUTOFFS for n in CUTOFFS]
                   if uses_verifier else [{"drift": 0., "noise": 0.}])
    cache, best = {}, None
    for triple in shares:
        thresholds = {}
        for name, share in zip(("mixture", "drift", "noise"), triple):
            key = (name, round(share, 4))
            if key not in cache:
                cache[key] = quantile_threshold(branch_score[name], clean,
                                                CLEAN_BUDGET * share)
            thresholds[name] = cache[key]
        mixture_alarm = branch_score["mixture"] > thresholds["mixture"]
        raw = {b: branch_score[b] > thresholds[b] for b in BRANCHES}
        for cutoffs in cutoff_grid:
            decision = mixture_alarm.copy()
            for column, branch in enumerate(BRANCHES):
                alarm = raw[branch] & ~veto[:, column]
                if uses_verifier and cutoffs[branch] > 0:
                    alarm = alarm & (keep[branch] >= cutoffs[branch])
                decision |= alarm
            report = score(decision)
            within_budget = report["_overall"]["clean_fpr"] <= CLEAN_BUDGET
            if arm == "baseline":
                feasible = within_budget and all(report[n]["f1"] >= f
                                                 for n, f in BASELINE_CONSTRAINTS.items())
                key = (report["_overall"]["worst_family_f1"], report["_overall"]["f1"])
            elif arm == "baseline_budget_only":
                # The deployed family floors proved unreachable on this network, so
                # only the false-alarm budget is enforced. The objective is
                # unchanged, and the floors are still reported.
                feasible = within_budget
                key = (report["_overall"]["worst_family_f1"], report["_overall"]["f1"])
            elif arm.endswith("_budget_only"):
                # The absolute floors are dropped for every arm alike, but the
                # relative protection is kept: a candidate may not win by giving up
                # a family the paired baseline held.
                feasible = within_budget and all(
                    report[n]["f1"] >= baseline_report[n]["f1"] - MAX_FAMILY_DETERIORATION
                    for n in FAMILY_NAMES.values())
                key = (report["_overall"]["f1"], report["_overall"]["worst_family_f1"])
            else:
                feasible = passes(report, baseline_report)
                key = (report["_overall"]["f1"], report["_overall"]["worst_family_f1"])
            if not feasible:
                continue
            if best is None or key > best[0]:
                best = (key, {"budget_shares": list(triple), "thresholds": thresholds,
                              "verifier_cutoffs": dict(cutoffs), "arm": arm}, report)
    return best


def constraint_frontier(scores, corpus, score):
    """Best value each baseline constraint can reach, and where they conflict.

    Called only when the deployed recipe turns out to have no feasible operating
    point on a network. It answers *which* constraint is unreachable and by how
    much, so an infeasible baseline is reported as a measured difficulty
    difference rather than as a crash.
    """
    clean = (np.asarray(corpus["families"]) == 0) & (np.asarray(corpus["labels"]) == 0)
    branch_score = {"mixture": corpus["mixture"],
                    "drift": scores["final_drift"], "noise": scores["final_noise"]}
    veto = specialist_veto_masks(corpus["router"], corpus["feedback"], .10, .10)
    shares = [t for t in itertools.product(SHARE_STEPS, repeat=3)
              if abs(sum(t) - 1.) < 1e-9]
    cache = {}
    best = {name: None for name in list(BASELINE_CONSTRAINTS) + ["clean_fpr"]}
    best_within_budget = {name: None for name in BASELINE_CONSTRAINTS}
    for triple in shares:
        thresholds = {}
        for name, share in zip(("mixture", "drift", "noise"), triple):
            key = (name, round(share, 4))
            if key not in cache:
                cache[key] = quantile_threshold(branch_score[name], clean,
                                                CLEAN_BUDGET * share)
            thresholds[name] = cache[key]
        decision = branch_score["mixture"] > thresholds["mixture"]
        for column, branch in enumerate(BRANCHES):
            decision |= (branch_score[branch] > thresholds[branch]) & ~veto[:, column]
        report = score(decision)
        within = report["_overall"]["clean_fpr"] <= CLEAN_BUDGET
        for name in BASELINE_CONSTRAINTS:
            value = report[name]["f1"]
            if best[name] is None or value > best[name]["value"]:
                best[name] = {"value": value, "clean_fpr": report["_overall"]["clean_fpr"],
                              "budget_shares": list(triple)}
            if within and (best_within_budget[name] is None
                           or value > best_within_budget[name]["value"]):
                best_within_budget[name] = {"value": value, "budget_shares": list(triple)}
        fpr = report["_overall"]["clean_fpr"]
        if best["clean_fpr"] is None or fpr < best["clean_fpr"]["value"]:
            best["clean_fpr"] = {"value": fpr, "budget_shares": list(triple)}
    return {
        "constraints": BASELINE_CONSTRAINTS,
        "clean_fpr_budget": CLEAN_BUDGET,
        "best_family_f1_anywhere_in_grid": {k: best[k] for k in BASELINE_CONSTRAINTS},
        "best_family_f1_within_the_clean_budget": best_within_budget,
        "lowest_clean_fpr_in_grid": best["clean_fpr"],
        "unreachable_within_budget": sorted(
            name for name, floor in BASELINE_CONSTRAINTS.items()
            if best_within_budget[name] is None
            or best_within_budget[name]["value"] < floor),
        "note": "the grid is the deployed budgeted-OR share grid; a constraint listed "
                "as unreachable cannot be met anywhere in it at this clean-FPR budget",
    }


def stage_calibrate(network, seed):
    folder = out(network, "seed", seed)
    if (folder / "operating_points.json").exists():
        return
    status(network, "calibrating", seed=seed)
    corpus, scores, keep, early, manifest = score_corpus_with(
        network, seed, NETWORKS[network]["calibration"])
    score = fast_scorer(corpus)
    result = {"training_seed": seed, "network": network,
              "calibration_scenarios": manifest["total_scenarios"],
              "calibration_rows": manifest["total_rows"], "arms": {}}
    baseline = search_arm(scores, corpus, keep, score, "baseline")
    if baseline is None:
        # The deployed recipe's own constraints are unreachable on this network.
        # That is a result about the network, not a reason to stop: it is measured
        # here, and both arms then fall back to the *same* declared rule so the
        # comparison stays paired. See the protocol's dated amendment.
        status(network, "deployed baseline constraints are infeasible", seed=seed)
        result["baseline_constraints_infeasible"] = True
        result["constraint_frontier"] = constraint_frontier(scores, corpus, score)
        baseline = search_arm(scores, corpus, keep, score, "baseline_budget_only")
        if baseline is None:
            raise RuntimeError(
                f"No operating point for seed {seed} even under the clean-FPR budget "
                "alone; the network cannot be calibrated by this recipe at all")
        result["baseline_fallback"] = (
            "maximise worst-family F1 subject to the 0.005 clean-FPR budget alone; "
            "the deployed recipe's family floors are reported, not enforced. The "
            "same fallback is applied to every arm so the pairing is unchanged.")
    else:
        result["baseline_constraints_infeasible"] = False
    result["arms"]["baseline"] = {"rule": baseline[1], "report": compact(baseline[2])}
    fallback = result.get("baseline_constraints_infeasible", False)
    for arm in ("threshold", "candidate"):
        found = search_arm(scores, corpus, keep, score,
                           f"{arm}_budget_only" if fallback else arm, baseline[2])
        result["arms"][arm] = ({"rule": found[1], "report": compact(found[2])}
                               if found else {"passes_gates": False})
    result["frozen_before_evaluation"] = True
    result["eval_evaluated"] = False
    write_json(folder / "operating_points.json", result)
    del corpus, scores, keep, early
    gc.collect()
    status(network, "calibration frozen", seed=seed,
           baseline=result["arms"]["baseline"]["report"]["pooled_f1"],
           candidate=result["arms"]["candidate"].get("report", {}).get("pooled_f1"))


# --------------------------------------------------------------------------
# fresh evaluation
# --------------------------------------------------------------------------

def stage_evaldata(network):
    import subprocess
    import sys
    spec = NETWORKS[network]
    for seed in spec["eval_seeds"]:
        name = f"ew_{network}_{spec['eval_purpose']}_seed{seed}"
        if not (ROOT / "data/thesis_v2" / name).exists():
            status(network, "generating fresh evaluation corpus", seed=seed)
            subprocess.run([sys.executable,
                            str(Path(__file__).with_name("generate_corpus.py")),
                            "--network", network, "--purpose", spec["eval_purpose"],
                            "--seed", str(seed), "--scenarios", "24"],
                           cwd=ROOT, check=True)
        corpus_name = f"{network}_eval_seed{seed}"
        if corpus_name not in CORPORA:
            CORPORA[corpus_name] = {
                "reference": reference_path(network), "network": network,
                "pieces": [(name, seed, "all:24")]}
        if not (CAMPAIGN / "features" / corpus_name / "manifest.json").exists():
            status(network, "caching fresh evaluation features", seed=seed)
            build(corpus_name)
    status(network, "evaluation data ready", seeds=list(spec["eval_seeds"]))


def reference_path(network):
    if network == "modena":
        return ROOT / "runs/operational/seasonal_family_deployment_v2/full/reference.joblib"
    return out(network, "reference.joblib")


def stage_evaluate(network, seed):
    folder = out(network, "seed", seed)
    target = folder / "evaluation_report.json"
    if target.exists():
        return
    points = json.loads((folder / "operating_points.json").read_text())
    spec = NETWORKS[network]
    per_seed = {}
    for data_seed in spec["eval_seeds"]:
        corpus_name = f"{network}_eval_seed{data_seed}"
        status(network, "scoring fresh evaluation corpus",
               training_seed=seed, data_seed=data_seed)
        corpus, scores, keep, early, manifest = score_corpus_with(
            network, seed, corpus_name)
        score = fast_scorer(corpus)
        entry = {"rows": manifest["total_rows"], "scenarios": manifest["total_scenarios"],
                 "arms": {}}
        for arm, active in points["arms"].items():
            if "rule" not in active:
                continue
            rule = active["rule"]
            veto = specialist_veto_masks(corpus["router"], corpus["feedback"], .10, .10)
            decision = corpus["mixture"] > rule["thresholds"]["mixture"]
            for column, branch in enumerate(BRANCHES):
                alarm = ((scores[f"final_{branch}"] > rule["thresholds"][branch])
                         & ~veto[:, column])
                cutoff = rule["verifier_cutoffs"].get(branch, 0.)
                if arm == "candidate" and cutoff > 0:
                    alarm = alarm & (keep[branch] >= cutoff)
                decision |= alarm
            entry["arms"][arm] = compact(score(decision))
        per_seed[str(data_seed)] = entry
        del corpus, scores, keep, early
        gc.collect()
    write_json(target, {"training_seed": seed, "network": network,
                        "frozen_rules_sha256": sha(folder / "operating_points.json"),
                        "per_data_seed": per_seed,
                        "locked_test_evaluated": False})
    status(network, "evaluation complete", seed=seed)


STAGES = ("folds", "oof", "fit", "heads", "calibrate", "evaldata", "evaluate")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", required=True, choices=sorted(NETWORKS))
    parser.add_argument("--stage", default="all", choices=("all",) + STAGES)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(TRAINING_SEEDS))
    args = parser.parse_args()

    out(args.network).mkdir(parents=True, exist_ok=True)
    lock = (out(args.network) / "campaign.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"A {args.network} campaign run is already active; "
                         "do not duplicate it")

    if args.network == "ltown":
        # The L-Town corpora and their network-specific reference are declared by
        # the setup module, so the shared cache builder knows about them before
        # any stage asks for them.
        import ltown_setup
        ltown_setup.register()
        if not ltown_setup.REFERENCE.exists():
            raise SystemExit(
                "The L-Town reference has not been fitted. Run, in order:\n"
                "  ltown_setup.py --stage generate\n"
                "  ltown_setup.py --stage reference\n"
                "  ltown_setup.py --stage features")

    wanted = STAGES if args.stage == "all" else (args.stage,)
    if "folds" in wanted:
        stage_folds(args.network)
    for stage, function in (("oof", stage_oof), ("fit", stage_fit),
                            ("heads", stage_heads), ("calibrate", stage_calibrate)):
        if stage in wanted:
            for seed in args.seeds:
                function(args.network, seed)
    if "evaldata" in wanted:
        stage_evaldata(args.network)
    if "evaluate" in wanted:
        stage_evaldata(args.network)
        for seed in args.seeds:
            stage_evaluate(args.network, seed)
    status(args.network, "runner finished", stages=list(wanted), seeds=args.seeds)


if __name__ == "__main__":
    main()
