"""Early-warning metrics on the *onset* clock.

Every number here is measured against the true attack onset, which is evaluator
metadata and never reaches the detector. This is a different clock from the
three-hour decision latency: three hours after a first warning is not three
hours after onset, and the two are never mixed in one figure.

Missed events stay in every denominator. An event with no warning before it
closes is a miss, not a dropped row, and its first-warning delay is censored
rather than deleted.
"""
from __future__ import annotations

import numpy as np

from wdn.early_warning import REPORT_CLASSES
from wdn.evidence_feedback import ROUTER_CLASSES

#: Deadlines measured in hours after the true onset, at hourly sampling.
DEADLINES = (0, 1, 2)

FAMILY_TO_CLASS = {"drift": 3, "noise": 4, "random": 1, "targeted": 1, "replay": 2}


def group_frame(result):
    """Graph-time keys, predictions and abstentions as flat arrays."""
    keys = result["keys"]
    return {"source": keys[:, 0], "scenario": keys[:, 1], "timestep": keys[:, 2],
            "prediction": result["prediction"], "abstain": result["abstain"],
            "confidence": result["confidence"], "probabilities": result["probabilities"]}


def _scenario_index(frame):
    index = {}
    order = np.lexsort((frame["timestep"], frame["scenario"]))
    scenario = frame["scenario"][order]
    boundaries = np.flatnonzero(np.r_[True, scenario[1:] != scenario[:-1]])
    ends = np.r_[boundaries[1:], len(order)]
    for start, end in zip(boundaries, ends):
        index[int(scenario[start])] = order[start:end]
    return index


def event_warnings(result, events, families=("drift", "noise"), deadlines=DEADLINES):
    """Per-event warning outcomes on the onset clock.

    ``events`` maps a global scenario id to its ledger entries. Only the named
    families are scored, but a warning of *any* family counts for the
    "warned at all" columns, so a confident wrong-family warning is never
    silently promoted to a correct one.
    """
    frame = group_frame(result)
    index = _scenario_index(frame)
    rows = []
    for scenario, entries in sorted(events.items()):
        positions = index.get(int(scenario))
        if positions is None:
            continue
        time_here = frame["timestep"][positions]
        for event in entries:
            if event["family"] not in families:
                continue
            onset, end = event["start"], event["start"] + event["steps"]
            active = positions[(time_here >= onset) & (time_here < end)]
            active_time = frame["timestep"][active]
            order = np.argsort(active_time)
            active, active_time = active[order], active_time[order]
            warned = (frame["prediction"][active] != 0) & ~frame["abstain"][active]
            correct = (frame["prediction"][active] == FAMILY_TO_CLASS[event["family"]]) \
                & ~frame["abstain"][active]
            row = {"scenario": int(scenario), "source": int(event["source"]),
                   "family": event["family"], "onset": onset, "steps": event["steps"],
                   "graph_hours_in_event": int(len(active))}
            for deadline in deadlines:
                inside = active_time <= onset + deadline
                row[f"warned_by_plus{deadline}"] = bool(np.any(warned & inside))
                row[f"correct_family_by_plus{deadline}"] = bool(np.any(correct & inside))
            first = np.flatnonzero(warned)
            row["first_warning_delay"] = (
                int(active_time[first[0]] - onset) if len(first) else None)
            first_correct = np.flatnonzero(correct)
            row["first_correct_delay"] = (
                int(active_time[first_correct[0]] - onset) if len(first_correct) else None)
            row["ever_warned"] = bool(warned.any())
            row["ever_correct"] = bool(correct.any())
            rows.append(row)
    return rows


def summarise_events(rows, deadlines=DEADLINES):
    """Fractions over *all* events, with misses kept in the denominator."""
    result = {}
    for family in sorted({row["family"] for row in rows}):
        selected = [row for row in rows if row["family"] == family]
        entry = {"events": len(selected)}
        for deadline in deadlines:
            entry[f"warned_by_plus{deadline}"] = float(np.mean(
                [row[f"warned_by_plus{deadline}"] for row in selected]))
            entry[f"correct_family_by_plus{deadline}"] = float(np.mean(
                [row[f"correct_family_by_plus{deadline}"] for row in selected]))
        entry["ever_warned"] = float(np.mean([row["ever_warned"] for row in selected]))
        entry["ever_correct_family"] = float(np.mean([row["ever_correct"] for row in selected]))
        delays = [row["first_warning_delay"] for row in selected]
        observed = [d for d in delays if d is not None]
        entry["first_warning_delay_hours"] = {
            "events": len(delays), "warned": len(observed),
            "censored_missed_events": len(delays) - len(observed),
            "median": float(np.median(observed)) if observed else None,
            "p90": float(np.percentile(observed, 90)) if observed else None,
            "mean": float(np.mean(observed)) if observed else None,
            "note": "quantiles are over warned events only; the censored count is "
                    "reported next to them and must not be dropped",
        }
        correct_delays = [row["first_correct_delay"] for row in selected]
        observed_correct = [d for d in correct_delays if d is not None]
        entry["first_correct_family_delay_hours"] = {
            "warned_with_correct_family": len(observed_correct),
            "censored": len(correct_delays) - len(observed_correct),
            "median": float(np.median(observed_correct)) if observed_correct else None,
        }
        result[family] = entry
    return result


def confusion(result, families, starts):
    """Graph-time confusion matrix, abstentions included as their own column."""
    from wdn.early_warning import CLASS_TO_REPORT, group_family_state
    truth = group_family_state(families, starts)
    frame = group_frame(result)
    predicted = np.array([CLASS_TO_REPORT[ROUTER_CLASSES[c]] for c in frame["prediction"]],
                         dtype=object)
    predicted[frame["abstain"]] = "abstain"
    columns = list(REPORT_CLASSES) + ["abstain"]
    matrix = {}
    for code, name in enumerate(ROUTER_CLASSES):
        selected = truth == code
        matrix[name] = {"graph_hours": int(selected.sum())}
        for column in columns:
            matrix[name][column] = int(np.sum(selected & (predicted == column)))
    return {"rows_are_true_state": True, "columns": columns, "matrix": matrix,
            "total_graph_hours": int(len(truth))}


def warning_episodes(result):
    """Maximal runs of consecutive graph-time hours carrying a non-normal warning."""
    frame = group_frame(result)
    active = (frame["prediction"] != 0) & ~frame["abstain"]
    index = np.flatnonzero(active)
    if not len(index):
        return {"episodes": 0, "warned_graph_hours": 0}
    key = frame["source"].astype(np.int64) * 10**9 + frame["scenario"].astype(np.int64)
    order = index[np.lexsort((frame["timestep"][index], key[index]))]
    same = key[order[1:]] == key[order[:-1]]
    adjacent = (frame["timestep"][order[1:]] - frame["timestep"][order[:-1]]) == 1
    episodes = int(1 + np.sum(~(same & adjacent)))
    return {"episodes": episodes, "warned_graph_hours": int(active.sum()),
            "rule": "maximal run of consecutive graph-time hours within one "
                    "(source, scenario) with a non-normal, non-abstaining warning"}


def clean_hour_false_alarms(result, families, starts, events, span_hours=0):
    """Warning false alarms over genuinely clean network-hours.

    A graph-time hour is clean when no event of any family is open in that
    scenario at that hour. ``span_hours`` extends the exclusion after an event
    closes, so a decaying tail is not counted as a fresh clean-hour false alarm
    unless that is what is being measured.
    """
    frame = group_frame(result)
    open_event = np.zeros(len(frame["timestep"]), dtype=bool)
    index = _scenario_index(frame)
    for scenario, entries in events.items():
        positions = index.get(int(scenario))
        if positions is None:
            continue
        time_here = frame["timestep"][positions]
        for event in entries:
            end = event["start"] + event["steps"] + int(span_hours)
            block = np.zeros(len(open_event), dtype=bool)
            block[positions] = (time_here >= event["start"]) & (time_here < end)
            open_event |= block
    clean = ~open_event
    warned = (frame["prediction"] != 0) & ~frame["abstain"]
    drift = (frame["prediction"] == 3) & ~frame["abstain"]
    noise = (frame["prediction"] == 4) & ~frame["abstain"]
    return {
        "clean_graph_hours": int(clean.sum()),
        "false_warning_hours": int(np.sum(warned & clean)),
        "rate": float(np.mean(warned[clean])) if clean.any() else None,
        "drift_false_warning_hours": int(np.sum(drift & clean)),
        "noise_false_warning_hours": int(np.sum(noise & clean)),
        "abstain_hours_on_clean": int(np.sum(frame["abstain"] & clean)),
        "post_event_span_hours_excluded": int(span_hours),
    }
