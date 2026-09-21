"""TRAIN-only causal-latency and retrospective trajectory diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from sklearn.metrics import average_precision_score

from wdn.run_expert_redesign import read_json, sha, write_json


RUN = Path("runs/operational/family_specific_experts_v1")
DATA = Path("data/thesis_v2/operational_modena_seed811")
OUTPUT = Path("thesis_v2/outputs/family_sequence_diagnosis.json")
REPORT = Path("thesis_v2/FAMILY_SEQUENCE_DIAGNOSIS.md")


def best_f1(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels) > 0
    order = np.argsort(-scores, kind="stable")
    score, y = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(score[:-1] != score[1:]), len(score)-1]
    predicted, tp = ends+1, np.cumsum(y)[ends]
    f1 = 2*tp/np.maximum(predicted+labels.sum(), 1)
    selected = int(np.argmax(f1))
    return {"f1": float(f1[selected]),
        "precision": float(tp[selected]/predicted[selected]),
        "recall": float(tp[selected]/labels.sum()),
        "threshold": float(score[ends[selected]])}


def metrics(scores, labels, scenario):
    by_scenario = {str(int(sid)): float(average_precision_score(
        labels[scenario == sid], scores[scenario == sid])) for sid in np.unique(scenario)}
    return {**best_f1(scores, labels),
        "macro_ap": float(np.mean(list(by_scenario.values()))),
        "pooled_ap": float(average_precision_score(labels, scores)),
        "by_scenario_ap": by_scenario, "points": int(len(labels)),
        "positives": int(np.sum(labels > 0))}


def trajectory_score(score, selected, event, node, timestep, mode, causal):
    """Known-event aggregation used only as a nondeployable diagnostic."""
    output = np.zeros(len(score), dtype=float)
    for event_id in np.unique(event[selected]):
        event_rows = selected & (event == event_id)
        for sensor in np.unique(node[event_rows]):
            rows = np.flatnonzero(event_rows & (node == sensor))
            rows = rows[np.argsort(timestep[rows], kind="stable")]
            values = score[rows]
            if causal:
                if mode == "mean":
                    aggregate = np.cumsum(values)/np.arange(1, len(values)+1)
                elif mode == "max":
                    aggregate = np.maximum.accumulate(values)
                elif mode == "top2":
                    aggregate = np.asarray([np.mean(np.sort(values[:end])[-min(end, 2):])
                                            for end in range(1, len(values)+1)])
                elif mode == "logit":
                    clipped = np.clip(values, 1e-6, 1-1e-6)
                    logits = np.log(clipped/(1-clipped))
                    aggregate = np.cumsum(logits)/np.sqrt(np.arange(1, len(values)+1))
                else:
                    raise ValueError("Unknown aggregation")
            else:
                if mode == "mean":
                    value = np.mean(values)
                elif mode == "max":
                    value = np.max(values)
                elif mode == "top2":
                    value = np.mean(np.sort(values)[-min(len(values), 2):])
                elif mode == "logit":
                    clipped = np.clip(values, 1e-6, 1-1e-6)
                    value = np.log(clipped/(1-clipped)).sum()/np.sqrt(len(values))
                else:
                    raise ValueError("Unknown aggregation")
                aggregate = np.full(len(values), value)
            output[rows] = aggregate
    return output


def run():
    summary = read_json(RUN/"summary.json")
    audit = read_json(RUN/"independent_audit.json")
    if (summary["survivors"] or not audit["saved_metrics_and_gates_reproduced_exact"]
            or any(summary[key] for key in
                   ("calibration_evaluated", "validation_evaluated", "test_evaluated"))):
        raise ValueError("Family-specific screen state is not the frozen failed TRAIN screen")
    arrays = dict(np.load(RUN/"oof_predictions.npz"))
    events = json.loads((DATA/"events.json").read_text())
    starts = {event_id: event["start_timestep"] for event_id, event in enumerate(events)}
    age = np.asarray([int(t)-starts[int(event_id)]+1 if event_id >= 0 else -1
                      for t, event_id in zip(arrays["timestep"], arrays["event"])])
    result = {"scope": "TRAIN-only post-hoc sequence diagnosis",
        "deployable_model": False, "calibration_evaluated": False,
        "validation_evaluated": False, "test_evaluated": False,
        "event_boundaries_used_for_diagnosis_only": True,
        "future_observations_used_only_in_retrospective_probe": True,
        "oof_sha256": sha(RUN/"oof_predictions.npz"), "families": {}}
    for family, family_id, column, heads in (
            ("drift", 3, 0, (0, 1)), ("noise", 4, 1, (2, 3))):
        selected = arrays["families"] == family_id
        labels, scenario = arrays["labels"][selected], arrays["scenario"][selected]
        sources = {"legacy": arrays["scores_L"][:, column],
            "separate": arrays["scores_E"][:, column],
            "fast": arrays["head_scores"][:, heads[0]],
            "persistent": arrays["head_scores"][:, heads[1]]}
        base = {name: metrics(score[selected], labels, scenario)
                for name, score in sources.items()}
        by_age = {}
        for name, low, high in (("hours_1_3", 1, 3), ("hours_4_6", 4, 6),
                                ("hours_7_plus", 7, 10_000), ("hours_4_plus", 4, 10_000)):
            rows = selected & (age >= low) & (age <= high)
            by_age[name] = metrics(arrays["scores_E"][rows, column], arrays["labels"][rows],
                                   arrays["scenario"][rows])
        aggregates = {"known_onset_causal": {}, "full_event_retrospective": {}}
        for source_name, score in sources.items():
            for mode in ("mean", "max", "top2", "logit"):
                for causal, group in ((True, "known_onset_causal"),
                                      (False, "full_event_retrospective")):
                    transformed = trajectory_score(score, selected, arrays["event"],
                        arrays["node"], arrays["timestep"], mode, causal)
                    aggregates[group][f"{source_name}_{mode}"] = metrics(
                        transformed[selected], labels, scenario)
        best_aggregates = {group: {"candidate": max(rows,
            key=lambda key: rows[key]["f1"]), "metrics": max(rows.values(),
            key=lambda row: row["f1"])} for group, rows in aggregates.items()}
        perfect_after_age = {}
        positives = (arrays["labels"] > 0) & selected
        total = int(positives.sum())
        for first_age in range(1, 8):
            tp = int((positives & (age >= first_age)).sum())
            perfect_after_age[str(first_age)] = {"tp": tp, "fn": total-tp,
                "f1_with_zero_fp": float(2*tp/max(total+tp, 1))}
        result["families"][family] = {"base_scores": base, "separate_by_event_age": by_age,
            "trajectory_aggregates": aggregates, "best_trajectory_aggregates": best_aggregates,
            "perfect_detection_from_clock_age": perfect_after_age}
    train = set(read_json(RUN/"signature.json")["splits"]["train"])
    thresholds = np.linspace(0., 8., 200_001)
    single_reading = []
    for event_id, event in enumerate(events):
        if event["scenario_id"] not in train or event["family"] != "noise":
            continue
        prevalence = len(event["targets"]["pressure"])/272
        ratio = np.sqrt(1+event["noise_factor"]**2)
        recall = 2*(1-ndtr(thresholds/ratio))
        false_positive_rate = 2*(1-ndtr(thresholds))
        f1 = (2*prevalence*recall
              /(prevalence*(1+recall)+(1-prevalence)*false_positive_rate))
        best = int(np.argmax(f1))
        precision = (prevalence*recall[best]
            /(prevalence*recall[best]+(1-prevalence)*false_positive_rate[best]))
        single_reading.append({"event_id": event_id, "scenario": event["scenario_id"],
            "noise_factor": event["noise_factor"], "variance_ratio": float(ratio),
            "ideal_gaussian_one_reading_max_f1": float(f1[best]),
            "precision": float(precision), "recall": float(recall[best]),
            "absolute_z_threshold": float(thresholds[best])})
    result["ideal_gaussian_single_reading_noise_reference"] = {
        "assumptions": "perfect known normal mean/scale, Gaussian null and injection, known event noise factor, one reading only",
        "not_a_sequence_upper_bound": True, "train_events": single_reading}
    write_json(OUTPUT, result)

    lines = ["# Drift/noise sequence gap: TRAIN-only diagnosis", "",
        "This is a post-hoc diagnostic over scenario-OOF TRAIN scores. It does not fit or select a",
        "deployable model and does not read calibration, validation or test. Known event boundaries",
        "and, in the retrospective probe, future observations are deliberately used only to locate",
        "where the remaining information lives.", "",
        "## Causal score by event age", "",
        "Each row uses a hindsight best threshold within that age stratum, so it is optimistic.", "",
        "| Family | 1-3 h F1 | 4-6 h F1 | 7+ h F1 | 4+ h F1 |",
        "|---|---:|---:|---:|---:|"]
    for family in ("drift", "noise"):
        rows = result["families"][family]["separate_by_event_age"]
        lines.append(f"| {family} | {rows['hours_1_3']['f1']:.4f} | "
            f"{rows['hours_4_6']['f1']:.4f} | {rows['hours_7_plus']['f1']:.4f} | "
            f"{rows['hours_4_plus']['f1']:.4f} |")
    lines += ["", "## Trajectory identity probes", "",
        "Known-onset causal aggregation sees only the current and earlier observations but is still",
        "nondeployable because it receives the true event reset. Full-event retrospective aggregation",
        "also sees the future and can only support an offline forensic interpretation.", "",
        "| Family | Best known-onset causal F1 | Candidate | Best retrospective F1 | Candidate |",
        "|---|---:|---|---:|---|"]
    for family in ("drift", "noise"):
        rows = result["families"][family]["best_trajectory_aggregates"]
        lines.append(f"| {family} | {rows['known_onset_causal']['metrics']['f1']:.4f} | "
            f"{rows['known_onset_causal']['candidate']} | "
            f"{rows['full_event_retrospective']['metrics']['f1']:.4f} | "
            f"{rows['full_event_retrospective']['candidate']} |")
    lines += ["", "The retrospective probe crosses 0.80 for both families, while even a true-onset",
        "causal aggregation does not. The current target therefore conflicts with early real-time",
        "localisation: information accumulates across the complete attacked-sensor trajectory, but",
        "the pointwise metric charges the detector for weak first observations before that identity is",
        "reliable. This is evidence about the current scores, not a mathematical impossibility proof.", "",
        "With zero false positives and otherwise perfect detection, drift must detect every positive",
        "from clock-hour 4 onward to reach 0.8017; noise must do so from hour 5 onward to reach 0.8477",
        "(waiting until hour 6 gives 0.7966). A realistic next model must therefore report detection",
        "delay explicitly. If offline smoothing is allowed, it must be named retrospective and must not",
        "be presented as an online detector.", "", "## Why one noise reading is often insufficient", "",
        "Under an idealised perfect-mean Gaussian calculation, the best single-reading F1 for the four",
        "TRAIN noise factors is " + ", ".join(
            f"{row['ideal_gaussian_one_reading_max_f1']:.3f} (factor {row['noise_factor']:.2f})"
            for row in single_reading) + ". Three of the four remain below 0.80 even with the normal",
        "mean, scale and event noise factor known. This is not an upper bound for a sequential detector;",
        "it explains why target-identity accumulation and detection-delay reporting are necessary.", ""]
    REPORT.write_text("\n".join(lines))
    print(f"Wrote {OUTPUT} and {REPORT}")


if __name__ == "__main__":
    run()
