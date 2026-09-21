"""Warning-to-confirmation behaviour, and early sensor-level evidence.

Two things the protocol asks for that neither the Stage B report nor the Stage C
screen covers on its own:

* **Explicit timestamps.** For every event: the true onset, the hour the network
  first warned, the hour whose decision first confirmed, and the wall-clock hour
  that confirmation was actually available — which is the decision hour plus the
  declared three-hour latency, not the decision hour itself.
* **Cancellation.** A warning episode that closes with no confirmation is a
  cancellation, and it is reported next to the confirmations rather than dropped.

There is also a deliberate negative deliverable here. The early head is a
**network-level family warning**; it does not localise. Rather than leave that
as an assertion, this script reports the sensor-level *causal* specialist recall
at the same deadlines, as a separate early-localisation-evidence metric that is
explicitly not the warning's own accuracy.

Calibration only. No locked EVAL or test corpus is read.

    /opt/miniconda3/bin/python thesis_v2/experiments/early_warning/warning_to_confirmation.py
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np

from wdn.alarm_verifier import BRANCHES, verifier_features
from wdn.early_warning import early_local_evidence
from wdn.evidence_feedback import specialist_veto_masks

from build_feature_cache import ROOT, sha, write_json
from campaign_data import load_calibration
from screen_verifier import specialist_score_parts

CAMPAIGN = ROOT / "runs/operational/early_warning_multiseed_v1"
SCREEN = CAMPAIGN / "verifier_screen_v1"
OUTPUT = CAMPAIGN / "warning_to_confirmation_v1"
DELTA = 3
DEADLINES = (0, 1, 2)
BRANCH_FAMILY = {"drift": 3, "noise": 4}


def confirmed_decision(corpus, scores, keep, rule):
    """The selected candidate's confirmed sensor-endpoint decision."""
    veto = specialist_veto_masks(corpus["router"], corpus["feedback"], .10, .10)
    decision = corpus["mixture"] > rule["thresholds"]["mixture"]
    for column, branch in enumerate(BRANCHES):
        alarm = (scores[f"final_{branch}"] > rule["thresholds"][branch]) & ~veto[:, column]
        cutoff = rule["verifier_cutoffs"].get(branch, 0.)
        if cutoff > 0:
            alarm = alarm & (keep[branch] >= cutoff)
        decision |= alarm
    return decision


def scenario_hours(values, scenario, timestep, mask):
    """Sorted hours of one scenario at which `mask` is true for any endpoint."""
    selected = (scenario == values) & mask
    return np.unique(timestep[selected]) if selected.any() else np.empty(0, np.int64)


def trace(corpus, early, decision, events):
    """One row per drift/noise event, with every timestamp written down."""
    scenario = np.asarray(corpus["scenario"])
    timestep = np.asarray(corpus["timestep"])
    labels = np.asarray(corpus["labels"]) > 0
    warned = (early["prediction"] != 0) & ~early["abstain"]
    keys = early["keys"]

    rows = []
    for sid, entries in sorted(events.items()):
        group_here = keys[:, 1] == sid
        group_time = keys[group_here, 2]
        group_warned = warned[group_here]
        group_class = early["prediction"][group_here]
        for event in entries:
            if event["family"] not in BRANCH_FAMILY:
                continue
            onset = event["start"]
            end = onset + event["steps"]
            inside = (group_time >= onset) & (group_time < end)
            warn_hours = np.sort(group_time[inside & group_warned])
            correct = group_class == BRANCH_FAMILY[event["family"]]
            correct_hours = np.sort(group_time[inside & group_warned & correct])

            confirm_hours = scenario_hours(sid, scenario, timestep,
                                           decision & labels
                                           & (timestep >= onset) & (timestep < end))
            first_warning = int(warn_hours[0]) if len(warn_hours) else None
            first_correct = int(correct_hours[0]) if len(correct_hours) else None
            first_confirm = int(confirm_hours[0]) if len(confirm_hours) else None
            rows.append({
                "scenario": int(sid), "family": event["family"],
                "onset_hour": int(onset), "event_hours": int(event["steps"]),
                "first_warning_hour": first_warning,
                "first_correct_family_warning_hour": first_correct,
                "first_confirmed_decision_hour": first_confirm,
                # The decision for hour t is finalised at t + delta, so this is
                # when an operator could actually act on the confirmation.
                "confirmation_available_hour": (None if first_confirm is None
                                                else first_confirm + DELTA),
                "warning_lead_over_confirmation_hours": (
                    None if first_warning is None or first_confirm is None
                    else first_confirm + DELTA - first_warning),
                "warned_before_confirmation": (
                    None if first_warning is None or first_confirm is None
                    else bool(first_warning < first_confirm + DELTA)),
                "warning_delay_from_onset": (None if first_warning is None
                                             else first_warning - onset),
                "confirmation_delay_from_onset": (
                    None if first_confirm is None else first_confirm + DELTA - onset),
            })
    return rows


def episodes_and_cancellations(corpus, early, decision, events):
    """Every warning episode, and whether it was ever confirmed."""
    keys = early["keys"]
    warned = (early["prediction"] != 0) & ~early["abstain"]
    scenario = np.asarray(corpus["scenario"])
    timestep = np.asarray(corpus["timestep"])
    open_event = {}
    for sid, entries in events.items():
        open_event[sid] = [(e["start"], e["start"] + e["steps"]) for e in entries]

    index = np.flatnonzero(warned)
    if not len(index):
        return {"episodes": 0}
    order = index[np.lexsort((keys[index, 2], keys[index, 1]))]
    starts = np.r_[True, (keys[order[1:], 1] != keys[order[:-1], 1])
                   | ((keys[order[1:], 2] - keys[order[:-1], 2]) != 1)]
    boundaries = np.flatnonzero(starts)
    ends = np.r_[boundaries[1:] - 1, len(order) - 1]

    confirmed, cancelled, during_event, on_clean = 0, 0, 0, 0
    lengths = []
    for start, stop in zip(boundaries, ends):
        sid = int(keys[order[start], 1])
        first, last = int(keys[order[start], 2]), int(keys[order[stop], 2])
        lengths.append(last - first + 1)
        window = (scenario == sid) & (timestep >= first) & (timestep <= last + DELTA)
        was_confirmed = bool(decision[window].any())
        confirmed += was_confirmed
        cancelled += not was_confirmed
        overlaps = any(not (last < s or first >= e) for s, e in open_event.get(sid, []))
        during_event += overlaps
        on_clean += not overlaps
    return {
        "episodes": len(lengths),
        "median_episode_hours": float(np.median(lengths)),
        "p90_episode_hours": float(np.percentile(lengths, 90)),
        "confirmed_within_episode_plus_delta": confirmed,
        "cancelled_without_confirmation": cancelled,
        "episodes_overlapping_a_real_event": during_event,
        "episodes_entirely_on_clean_hours": on_clean,
        "rule": f"an episode is confirmed if any confirmed alarm falls in its "
                f"scenario between its first hour and its last hour + {DELTA}",
    }


def early_sensor_evidence(corpus, scores, events):
    """Sensor-level causal recall at the warning deadlines — not localisation."""
    scenario = np.asarray(corpus["scenario"])
    timestep = np.asarray(corpus["timestep"])
    labels = np.asarray(corpus["labels"]) > 0
    families = np.asarray(corpus["families"])
    clean = (families == 0) & ~labels
    result = {}
    for branch, code in BRANCH_FAMILY.items():
        score = scores[f"causal_{branch}"]
        # A threshold at the same 0.5% clean budget the deployed branch uses, so
        # the recall below is comparable to a deployed operating point.
        values = np.sort(score[clean])
        threshold = float(values[int(np.floor(.995 * len(values)))])
        entry = {"causal_threshold_at_0.005_clean_budget": threshold, "by_deadline": {}}
        for deadline in DEADLINES:
            selected = np.zeros(len(labels), bool)
            for sid, entries in events.items():
                for event in entries:
                    if event["family"] != branch:
                        continue
                    selected |= ((scenario == sid) & (timestep >= event["start"])
                                 & (timestep <= event["start"] + deadline))
            scope = selected & labels & (families == code)
            entry["by_deadline"][f"onset_plus_{deadline}"] = {
                "attacked_endpoints": int(scope.sum()),
                "detected": int(np.sum(scope & (score > threshold))),
                "recall": float(np.mean(score[scope] > threshold)) if scope.any() else None,
            }
        result[branch] = entry
    result["note"] = ("this is sensor-level causal evidence at the warning "
                      "deadlines, reported separately. It is NOT the network-level "
                      "warning's accuracy, and it is not early localisation by the "
                      "warning head, which does not localise.")
    return result


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    selection = json.loads((SCREEN / "selection_frozen.json").read_text())
    if selection["status"] != "selected":
        raise SystemExit("No candidate was selected; there is nothing to trace")
    rule = selection["candidates"][selection["selected"]]["rule"]
    bundle = joblib.load(SCREEN / "bundle.joblib")
    head = bundle["early_head"]
    verifier = bundle["verifiers"][selection["selected"]]

    corpus = load_calibration()
    scores = specialist_score_parts(corpus["causal"], corpus["promoted_head"], corpus)
    local = early_local_evidence(corpus["X"], corpus["names"], corpus["causal"])
    early = head.predict_groups(local, corpus)
    del local
    features, _ = verifier_features(corpus["X"], corpus["names"], scores, early, corpus)
    keep = verifier.verify(features, scores)
    del features
    decision = confirmed_decision(corpus, scores, keep, rule)

    events = {k: v for k, v in corpus["events"].items()
              if k in set(np.unique(corpus["scenario"]).tolist())}
    rows = trace(corpus, early, decision, events)
    leads = [r["warning_lead_over_confirmation_hours"] for r in rows
             if r["warning_lead_over_confirmation_hours"] is not None]

    result = {
        "campaign": "early_warning_multiseed_v1",
        "scope": "Modena calibration, 99 scenarios; the selected candidate rule. "
                 "This is a selection surface, not a confirmation.",
        "selected_candidate": selection["selected"],
        "declared_decision_latency_hours": DELTA,
        "clocks": {
            "warning": "measured against the true attack onset, evaluator metadata only",
            "confirmation": "the decision for hour t becomes available at t + 3",
        },
        "events_traced": len(rows),
        "per_event": rows,
        "summary": {
            "events_with_a_warning": sum(r["first_warning_hour"] is not None for r in rows),
            "events_with_a_confirmation": sum(
                r["first_confirmed_decision_hour"] is not None for r in rows),
            "events_warned_before_confirmation_was_available": sum(
                bool(r["warned_before_confirmation"]) for r in rows),
            "median_warning_lead_hours": float(np.median(leads)) if leads else None,
            "mean_warning_lead_hours": float(np.mean(leads)) if leads else None,
            "note": "lead is measured only on events that got both a warning and a "
                    "confirmation; the counts above say how many did not",
        },
        "episodes": episodes_and_cancellations(corpus, early, decision, events),
        "early_sensor_evidence": early_sensor_evidence(corpus, scores, events),
        "selection_sha256": sha(SCREEN / "selection_frozen.json"),
        "eval_evaluated": False, "test_evaluated": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(OUTPUT / "warning_to_confirmation.json", result)
    print(json.dumps({"summary": result["summary"], "episodes": result["episodes"],
                      "early_sensor_evidence": {
                          k: v["by_deadline"] for k, v in
                          result["early_sensor_evidence"].items() if k != "note"}},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
