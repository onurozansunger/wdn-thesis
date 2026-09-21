"""Stage A: decompose the deployed detector's false alarms.

Two surfaces, kept apart on purpose:

* **calibration** (99 scenarios) carries all three branches — general mixture,
  drift specialist, noise specialist — plus the evidence router and the feedback
  veto, so the branch attribution is complete here. The models were fitted on
  TRAIN and never saw these rows, but the deployed thresholds were *chosen* on
  exactly these rows, so calibration false-positive counts at those thresholds
  are optimistically biased. Every calibration number below carries that caveat.

* **TRAIN out-of-fold** (62 scenarios, four generator-source-held folds) carries
  the two specialists only. Each fold fitted its own normal reference
  (``expanded_train_weak_experts_v1/fold_*/reference.joblib``), so these scores
  are full-pipeline OOF rather than fixed-reference conditional. There is no
  matching out-of-fold artifact for the general mixture on that corpus, so the
  OOF surface cannot attribute anything to the general branch and does not
  pretend to.

Nothing here fits, calibrates or selects. It reads frozen artifacts and counts.

    /opt/miniconda3/bin/python thesis_v2/experiments/early_warning/audit_false_alarms.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from wdn.evidence_feedback import specialist_veto_masks
from wdn.latency_deployment import FAMILY_NAMES, family_scores, forward_max, logit_blend

from build_feature_cache import ROOT, load_corpus, sha, write_json

CAMPAIGN = ROOT / "runs/operational/early_warning_multiseed_v1"
OUTPUT = CAMPAIGN / "stage_a_audit"
FIGURES = ROOT / "thesis_v2/outputs/early_warning_multiseed"
BASE = ROOT / "runs/operational/cross_mechanism_deployment_v1"
META = ROOT / "runs/operational/feedback_router_calibration_v2"
VETO = ROOT / "runs/operational/feedback_router_veto_v1/selection_frozen.json"
SEASONAL_OOF = ROOT / "runs/operational/seasonal_family_experts_v3/oof_predictions.npz"
DELAYED_OOF = ROOT / "runs/operational/delayed_decision_head_v1/oof_predictions.npz"
OOF_FOLDS = ROOT / "runs/operational/expanded_train_weak_experts_v1"

#: Audit covariates, drawn from the causal feature bank.
COVARIATES = ("reference_support", "normal_error_scale", "last_gap", "abs_residual",
              "seq_seen", "seasonal_support_24")

#: Two alarmed endpoints of one sensor series join the same run when the later
#: one is at most this many hours after the earlier. At a 0.50 missing rate most
#: series are gappy, so a strictly consecutive-hour rule would shatter every run.
RUN_GAP_HOURS = 3

#: A quiet endpoint within this many hours after an event closes is a tail row.
POST_EVENT_HOURS = 6

BRANCHES = ("general", "drift", "noise")


def branch_alarms(corpus, veto_rule):
    """Reconstruct the deployed per-branch alarms exactly as EVAL-3 computed them."""
    thresholds = veto_rule["base_thresholds"]
    raw = corpus["promoted"]
    head = corpus["promoted_head"]
    pooled = np.column_stack((forward_max(raw[:, 0], corpus, veto_rule["specialist_delta"]),
                              forward_max(raw[:, 1], corpus, veto_rule["specialist_delta"])))
    final = np.column_stack((logit_blend(head[:, 0], pooled[:, 0], .5),
                             logit_blend(head[:, 1], pooled[:, 1], .5)))
    mixture = forward_max(corpus["mixture"], corpus, veto_rule["mixture_delta"])
    veto = specialist_veto_masks(corpus["router"], corpus["feedback"],
                                 veto_rule["margin"], veto_rule["feedback_cutoff"])
    alarms = {
        "general": mixture > thresholds["mixture"],
        "drift": (final[:, 0] > thresholds["drift"]) & ~veto[:, 0],
        "noise": (final[:, 1] > thresholds["noise"]) & ~veto[:, 1],
    }
    unguarded = {"drift": final[:, 0] > thresholds["drift"],
                 "noise": final[:, 1] > thresholds["noise"]}
    return alarms, unguarded, veto, {"mixture": mixture, "specialists": final}


def series_runs(corpus, alarm, gap=RUN_GAP_HOURS):
    """Group alarmed endpoints into runs within one (source, scenario, node) series.

    Returns a run id per alarmed row (-1 elsewhere), the run length in endpoints,
    and the run span in hours.
    """
    index = np.flatnonzero(alarm)
    if not len(index):
        return np.full(len(alarm), -1, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
    key = (corpus["source"].astype(np.int64) * 10**9
           + corpus["scenario"].astype(np.int64) * 10**5
           + corpus["node"].astype(np.int64))
    order = index[np.lexsort((corpus["timestep"][index], key[index]))]
    same = key[order[1:]] == key[order[:-1]]
    close = (corpus["timestep"][order[1:]] - corpus["timestep"][order[:-1]]) <= gap
    starts = np.r_[True, ~(same & close)]
    run_of_sorted = np.cumsum(starts) - 1
    run_id = np.full(len(alarm), -1, np.int64)
    run_id[order] = run_of_sorted
    lengths = np.bincount(run_of_sorted)
    boundaries = np.flatnonzero(starts)
    ends = np.r_[boundaries[1:] - 1, len(order) - 1]
    spans = corpus["timestep"][order[ends]] - corpus["timestep"][order[boundaries]] + 1
    return run_id, lengths, spans


def post_event_mask(corpus, events_by_scenario, hours=POST_EVENT_HOURS):
    """Quiet endpoints shortly after an event closes, by the closing family."""
    quiet = corpus["labels"] == 0
    masks = {name: np.zeros(len(quiet), bool) for name in FAMILY_NAMES.values()}
    for scenario, events in events_by_scenario.items():
        selected = corpus["scenario"] == scenario
        if not selected.any():
            continue
        time_here = corpus["timestep"][selected]
        for event in events:
            end = event["start_timestep"] + event["actual_steps"]
            name = {"stealthy": "drift"}.get(event["family"], event["family"])
            if name not in masks:
                continue
            local = (time_here >= end) & (time_here < end + hours)
            block = np.zeros(len(quiet), bool)
            block[selected] = local
            masks[name] |= block & quiet
    return masks


def decile_table(values, false_positive, negative):
    """False-alarm rate of negatives by decile of a covariate."""
    edges = np.unique(np.quantile(values[negative], np.linspace(0, 1, 11)))
    if len(edges) < 3:
        return {"note": "covariate is near-constant on negatives", "edges": edges.tolist()}
    bucket = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        selected = negative & (bucket == b)
        count = int(selected.sum())
        rows.append({"bin": b, "low": float(edges[b]), "high": float(edges[b + 1]),
                     "negatives": count,
                     "false_alarms": int(np.sum(false_positive & selected)),
                     "rate": float(np.mean(false_positive[selected])) if count else None})
    return {"edges": edges.tolist(), "bins": rows}


def concentration(keys, false_positive):
    """How concentrated the false alarms are over a grouping key."""
    hit = keys[false_positive]
    if not len(hit):
        return {"units_with_false_alarms": 0}
    unique, counts = np.unique(hit, return_counts=True)
    order = np.argsort(-counts)
    counts = counts[order]
    total = counts.sum()
    top = max(1, int(np.ceil(.10 * len(np.unique(keys)))))
    return {
        "distinct_units": int(len(np.unique(keys))),
        "units_with_false_alarms": int(len(unique)),
        "total_false_alarms": int(total),
        "share_in_worst_10pct_of_units": float(counts[:top].sum() / total),
        "worst_units": [{"unit": int(unique[order][i]), "false_alarms": int(counts[i])}
                        for i in range(min(10, len(counts)))],
    }


def attribution(alarms, false_positive):
    """Which branches produced each false alarm, and which ones did so alone."""
    active = {name: alarms[name] & false_positive for name in BRANCHES}
    result = {"total_false_alarms": int(false_positive.sum())}
    for name in BRANCHES:
        others = np.logical_or.reduce([alarms[other] for other in BRANCHES if other != name])
        result[name] = {
            "raises": int(active[name].sum()),
            "raises_alone": int(np.sum(active[name] & ~others)),
            "removable_by_silencing_this_branch": int(np.sum(active[name] & ~others)),
        }
    result["combinations"] = {}
    for pattern in range(1, 8):
        chosen = [BRANCHES[i] for i in range(3) if pattern >> i & 1]
        mask = np.ones(len(false_positive), bool)
        for i, name in enumerate(BRANCHES):
            mask &= alarms[name] if pattern >> i & 1 else ~alarms[name]
        count = int(np.sum(mask & false_positive))
        if count:
            result["combinations"]["+".join(chosen)] = count
    covered = sum(result["combinations"].values())
    if covered != result["total_false_alarms"]:
        raise AssertionError("Branch attribution does not partition the false alarms")
    return result


def counterfactual(alarms, corpus, keep):
    """Family and pooled metrics when only the named branches may fire."""
    decision = np.logical_or.reduce([alarms[name] for name in keep])
    report = family_scores(decision, corpus)
    labels = corpus["labels"] > 0
    return {
        "branches": list(keep),
        "pooled_f1": float(report["_overall"]["f1"]),
        "clean_fpr": float(report["_overall"]["clean_fpr"]),
        "tp": int(np.sum(labels & decision)),
        "fp": int(np.sum(~labels & decision)),
        "fn": int(np.sum(labels & ~decision)),
        "family_f1": {name: float(report[name]["f1"]) for name in FAMILY_NAMES.values()
                      if name in report},
    }


def audit_surface(corpus, alarms, covariate_names, events_by_scenario=None,
                  branches=BRANCHES):
    labels = corpus["labels"] > 0
    decision = np.logical_or.reduce([alarms[name] for name in branches])
    false_positive = decision & ~labels
    negative = ~labels
    clean = (corpus["families"] == 0) & negative
    report = family_scores(decision, corpus)

    per_family = {}
    for code, name in [(0, "clean")] + sorted(FAMILY_NAMES.items()):
        scope = (corpus["families"] == code) & negative
        per_family[name] = {
            "negative_endpoints": int(scope.sum()),
            "false_alarms": int(np.sum(false_positive & scope)),
            "rate": float(np.mean(false_positive[scope])) if scope.any() else None,
        }

    run_id, lengths, spans = series_runs(corpus, decision)
    fp_runs = np.unique(run_id[false_positive])
    fp_runs = fp_runs[fp_runs >= 0]
    purity = np.zeros(len(lengths))
    np.add.at(purity, run_id[decision], (~labels[decision]).astype(float))
    all_false = purity == lengths
    duration = {
        "rule": f"same sensor series, next alarm within {RUN_GAP_HOURS} h",
        "alarm_runs": int(len(lengths)),
        "runs_containing_a_false_alarm": int(len(fp_runs)),
        "entirely_false_runs": int(np.sum(all_false)),
        "false_alarms_in_isolated_runs": int(np.sum(false_positive & (lengths[run_id] == 1))),
        "false_alarms_in_short_runs_2_3": int(np.sum(
            false_positive & (lengths[run_id] >= 2) & (lengths[run_id] <= 3))),
        "false_alarms_in_persistent_runs_4plus": int(np.sum(
            false_positive & (lengths[run_id] >= 4))),
        "entirely_false_run_length_quantiles": np.quantile(
            lengths[all_false], [.5, .9, .99]).tolist() if all_false.any() else [],
        "entirely_false_run_span_hours_quantiles": np.quantile(
            spans[all_false], [.5, .9, .99]).tolist() if all_false.any() else [],
    }

    phase = corpus["timestep"] % 24
    phase_rows = []
    for hour in range(24):
        selected = negative & (phase == hour)
        phase_rows.append({"hour_of_day": hour, "negatives": int(selected.sum()),
                           "false_alarms": int(np.sum(false_positive & selected)),
                           "rate": float(np.mean(false_positive[selected])) if selected.any() else None})

    result = {
        "rows": int(len(labels)),
        "positives": int(labels.sum()),
        "negatives": int(negative.sum()),
        "clean_endpoints": int(clean.sum()),
        "pooled": {"f1": float(report["_overall"]["f1"]),
                   "clean_fpr": float(report["_overall"]["clean_fpr"]),
                   "all_negative_fpr": float(report["_overall"]["all_negative_fpr"]),
                   "tp": int(np.sum(labels & decision)),
                   "fp": int(false_positive.sum()),
                   "fn": int(np.sum(labels & ~decision))},
        "family_f1": {name: float(report[name]["f1"]) for name in FAMILY_NAMES.values()
                      if name in report},
        "false_alarms_by_family_period": per_family,
        "branch_attribution": attribution({k: alarms[k] for k in branches}, false_positive)
        if set(branches) == set(BRANCHES) else None,
        "temporal_duration": duration,
        "daily_phase": phase_rows,
        "concentration_by_sensor": concentration(corpus["node"].astype(np.int64), false_positive),
        "concentration_by_scenario": concentration(corpus["scenario"].astype(np.int64), false_positive),
        "covariates": {name: decile_table(corpus["X"][:, i], false_positive, negative)
                       for i, name in enumerate(covariate_names)},
    }
    if events_by_scenario is not None:
        tails = post_event_mask(corpus, events_by_scenario)
        result["post_event_tails"] = {
            name: {"quiet_endpoints": int(mask.sum()),
                   "false_alarms": int(np.sum(false_positive & mask)),
                   "rate": float(np.mean(false_positive[mask])) if mask.any() else None}
            for name, mask in tails.items()}
        result["post_event_tails"]["share_of_all_false_alarms"] = float(
            np.sum(false_positive & np.logical_or.reduce(list(tails.values())))
            / max(1, false_positive.sum()))
    return result, decision, false_positive


def calibration_events():
    """Event ledgers of the calibration corpus, keyed by global scenario id."""
    import yaml
    directories = [("operational_modena_seed811", 811)] + [
        (f"operational_calibration_expansion_seed{seed}", seed)
        for seed in (12811, 13811, 14811, 15811)]
    by_scenario = {}
    for name, seed in directories:
        path = ROOT / "data/thesis_v2" / name
        for event in json.loads((path / "events.json").read_text()):
            key = int(event["scenario_id"]) + seed * 1000
            by_scenario.setdefault(key, []).append(event)
    del yaml
    return by_scenario


def run_calibration(veto_rule):
    corpus, manifest = load_corpus("modena_calibration", columns=COVARIATES)
    base = dict(np.load(BASE / "scores_calibration.npz"))
    meta = dict(np.load(META / "feedback_calibration.npz"))
    for key in ("labels", "families", "timestep", "node", "scenario", "source"):
        if not np.array_equal(np.asarray(base[key]), np.asarray(corpus[key])):
            raise ValueError(f"Calibration cache and frozen scores disagree on {key}")
        if not np.array_equal(np.asarray(meta[key]), np.asarray(corpus[key])):
            raise ValueError(f"Feedback predictions and frozen scores disagree on {key}")
    corpus.update({key: base[key] for key in ("mixture", "promoted", "promoted_head")})
    corpus.update({"router": meta["router"], "feedback": meta["feedback"]})

    alarms, unguarded, veto, scores = branch_alarms(corpus, veto_rule)
    surface, decision, false_positive = audit_surface(
        corpus, alarms, COVARIATES, calibration_events())

    labels = corpus["labels"] > 0
    surface["guard_effect"] = {
        "veto_rows": int(veto.sum()),
        "drift_alarms_before_guard": int(unguarded["drift"].sum()),
        "noise_alarms_before_guard": int(unguarded["noise"].sum()),
        "alarms_removed_by_guard": int(np.sum(
            (np.logical_or.reduce([alarms["general"], unguarded["drift"], unguarded["noise"]]))
            & ~decision)),
        "removed_true_alarms": int(np.sum(
            (np.logical_or.reduce([alarms["general"], unguarded["drift"], unguarded["noise"]]))
            & ~decision & labels)),
        "removed_clean_false_alarms": int(np.sum(
            (np.logical_or.reduce([alarms["general"], unguarded["drift"], unguarded["noise"]]))
            & ~decision & ~labels & (corpus["families"] == 0))),
    }
    surface["counterfactuals"] = {
        "all_branches": counterfactual(alarms, corpus, BRANCHES),
        "general_only": counterfactual(alarms, corpus, ("general",)),
        "specialists_only": counterfactual(alarms, corpus, ("drift", "noise")),
        "general_plus_drift": counterfactual(alarms, corpus, ("general", "drift")),
        "general_plus_noise": counterfactual(alarms, corpus, ("general", "noise")),
    }
    surface["scenarios"] = manifest["total_scenarios"]
    surface["caveat"] = (
        "Models never saw these rows, but the deployed thresholds were selected on "
        "them; false-positive counts here are optimistically biased.")
    return surface, corpus, alarms, false_positive


def run_train_oof():
    """Specialist-only decomposition on generator-source-held TRAIN OOF.

    The covariates come from the same fold caches the OOF scores were built
    from, concatenated in the same fold order, so nothing is realigned by
    guesswork: the row-order check below fails loudly if that ever changes.
    """
    seasonal = dict(np.load(SEASONAL_OOF))
    delayed = dict(np.load(DELAYED_OOF))
    for key in ("labels", "families", "scenario", "source", "timestep", "node"):
        if not np.array_equal(seasonal[key], delayed[key]):
            raise ValueError(f"TRAIN OOF artifacts disagree on {key}")

    bank = json.loads((OOF_FOLDS / "feature_names.json").read_text())
    covariates = tuple(name for name in COVARIATES if name in bank)
    keep = [bank.index(name) for name in covariates]
    parts = []
    for fold in range(4):
        with np.load(OOF_FOLDS / f"fold_{fold}/features_held_out.npz") as loaded:
            parts.append({"X": loaded["X"][:, keep],
                          **{key: loaded[key] for key in
                             ("labels", "families", "scenario", "source", "timestep", "node")}})
    corpus = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    for key in ("labels", "families", "scenario", "source", "timestep", "node"):
        if not np.array_equal(np.asarray(corpus[key]), np.asarray(delayed[key])):
            raise ValueError(
                f"TRAIN OOF fold caches do not reproduce the OOF score row order on {key}")

    negative = delayed["labels"] == 0
    clean = (delayed["families"] == 0) & negative
    scores = {"drift": delayed["delayed_blend"][:, 0], "noise": delayed["delayed_blend"][:, 1]}
    # Same calibration *procedure* as deployment, applied to this surface: a
    # per-branch quantile threshold at the deployed 0.20 / 0.75 budget shares.
    from wdn.latency_deployment import quantile_threshold
    thresholds = {"drift": quantile_threshold(scores["drift"], clean, .005 * .20),
                  "noise": quantile_threshold(scores["noise"], clean, .005 * .75)}
    alarms = {name: scores[name] > thresholds[name] for name in ("drift", "noise")}
    surface, _, _ = audit_surface(corpus, alarms, covariates, branches=("drift", "noise"))
    surface["thresholds"] = thresholds
    surface["scope"] = (
        "62-scenario TRAIN, four generator-source-held folds, each with its own "
        "normal reference: full-pipeline OOF, not fixed-reference conditional.")
    surface["general_branch"] = (
        "No out-of-fold artifact exists for the general mixture on this corpus, so "
        "nothing here is attributed to the general branch.")
    surface["threshold_note"] = (
        "Thresholds re-derived on this surface by the deployed quantile procedure at "
        "the deployed budget shares; they are not the deployed threshold values.")
    surface["covariates_available"] = list(covariates)
    return surface


def figures(corpus, alarms, false_positive, surface):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    combos = surface["branch_attribution"]["combinations"]
    order = sorted(combos, key=lambda k: -combos[k])
    axes[0].bar(range(len(order)), [combos[k] for k in order], color="#4c6ef5")
    axes[0].set_xticks(range(len(order)))
    axes[0].set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("false alarms")
    axes[0].set_title("Which branches fire on a false alarm")

    families = surface["false_alarms_by_family_period"]
    keys = [k for k in families if families[k]["negative_endpoints"]]
    axes[1].bar(range(len(keys)), [families[k]["false_alarms"] for k in keys], color="#f76707")
    axes[1].set_xticks(range(len(keys)))
    axes[1].set_xticklabels(keys, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("false alarms")
    axes[1].set_title("False alarms by family period")

    duration = surface["temporal_duration"]
    bars = {"isolated": duration["false_alarms_in_isolated_runs"],
            "2-3 h run": duration["false_alarms_in_short_runs_2_3"],
            "persistent 4+": duration["false_alarms_in_persistent_runs_4plus"]}
    axes[2].bar(range(3), list(bars.values()), color="#2b8a3e")
    axes[2].set_xticks(range(3))
    axes[2].set_xticklabels(list(bars), fontsize=8)
    axes[2].set_ylabel("false alarms")
    axes[2].set_title("False alarms by run length")

    fig.suptitle("Stage A false-alarm audit — Modena calibration (99 scenarios)")
    fig.tight_layout()
    fig.savefig(FIGURES / "stage_a_false_alarm_audit.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    phase = surface["daily_phase"]
    axes[0].plot([r["hour_of_day"] for r in phase], [r["rate"] for r in phase], marker="o")
    axes[0].set_xlabel("hour of day"); axes[0].set_ylabel("false-alarm rate")
    axes[0].set_title("Daily phase")
    support = surface["covariates"].get("reference_support", {}).get("bins", [])
    if support:
        axes[1].plot([r["bin"] for r in support], [r["rate"] for r in support], marker="s",
                     color="#c92a2a")
        axes[1].set_xlabel("reference-support decile")
        axes[1].set_ylabel("false-alarm rate")
        axes[1].set_title("Reference support")
    fig.tight_layout()
    fig.savefig(FIGURES / "stage_a_false_alarm_covariates.png", dpi=160)
    plt.close(fig)


def csv_export(surface, path):
    rows = ["section,key,subkey,value"]

    def emit(section, key, subkey, value):
        rows.append(f"{section},{key},{subkey},{value}")

    for name, entry in surface["false_alarms_by_family_period"].items():
        for field, value in entry.items():
            emit("family_period", name, field, value)
    if surface.get("branch_attribution"):
        for name in BRANCHES:
            for field, value in surface["branch_attribution"][name].items():
                emit("branch", name, field, value)
        for name, value in surface["branch_attribution"]["combinations"].items():
            emit("branch_combination", name, "false_alarms", value)
    for field, value in surface["temporal_duration"].items():
        if not isinstance(value, list):
            emit("duration", field, "", value)
    for row in surface["daily_phase"]:
        emit("daily_phase", row["hour_of_day"], "rate", row["rate"])
    for name, table in surface["covariates"].items():
        for row in table.get("bins", []):
            emit("covariate", name, f"bin{row['bin']}", row["rate"])
    for name, entry in surface.get("counterfactuals", {}).items():
        for field in ("pooled_f1", "clean_fpr", "tp", "fp", "fn"):
            emit("counterfactual", name, field, entry[field])
    Path(path).write_text("\n".join(rows) + "\n")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    veto_rule = json.loads(VETO.read_text())["rule"]

    print("auditing calibration surface", flush=True)
    calibration, corpus, alarms, false_positive = run_calibration(veto_rule)
    print("auditing TRAIN out-of-fold surface", flush=True)
    train_oof = run_train_oof()

    result = {
        "campaign": "early_warning_multiseed_v1",
        "stage": "A",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "deployed_rule": veto_rule,
        "surfaces": {"calibration": calibration, "train_oof": train_oof},
        "locked_test_evaluated": False,
        "eval1_evaluated": False, "eval2_evaluated": False, "eval3_evaluated": False,
        "artifact_sha256": {
            "calibration_scores": sha(BASE / "scores_calibration.npz"),
            "feedback_predictions": sha(META / "feedback_calibration.npz"),
            "veto_selection": sha(VETO),
            "delayed_oof": sha(DELAYED_OOF),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(OUTPUT / "false_alarm_audit.json", result)
    csv_export(calibration, OUTPUT / "false_alarm_audit_calibration.csv")
    csv_export(train_oof, OUTPUT / "false_alarm_audit_train_oof.csv")
    figures(corpus, alarms, false_positive, calibration)

    print(json.dumps({
        "calibration_pooled_f1": calibration["pooled"]["f1"],
        "calibration_fp": calibration["pooled"]["fp"],
        "false_alarms_by_family_period": {
            k: v["false_alarms"] for k, v in
            calibration["false_alarms_by_family_period"].items()},
        "branch_raises_alone": {
            k: calibration["branch_attribution"][k]["raises_alone"] for k in BRANCHES},
        "counterfactual_pooled_f1": {
            k: round(v["pooled_f1"], 4) for k, v in calibration["counterfactuals"].items()},
        "run_length": {
            k: calibration["temporal_duration"][k] for k in
            ("false_alarms_in_isolated_runs", "false_alarms_in_short_runs_2_3",
             "false_alarms_in_persistent_runs_4plus")},
        "train_oof_fp": train_oof["pooled"]["fp"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
