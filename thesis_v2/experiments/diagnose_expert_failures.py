"""Read-only model/data diagnostics for the frozen development experiments.

Event/target metadata are used solely for evaluation and a clearly labelled
oracle reference ablation, never deployable inference features.
No models are trained, no thresholds are fitted on validation, and no test
examples are extracted from the shared dataset containers.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle

import joblib
import numpy as np
from sklearn.metrics import average_precision_score

from wdn.train_operational_moe import FAMILY_NAMES, _binary_counts


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/"runs/operational/family_balance_v1"
DATA = ROOT/"data/thesis_v2/operational_modena_seed811"
MECHANISMS = ("general", "abrupt", "replay", "drift", "noise")
OWNER = {1: 1, 2: 2, 3: 3, 4: 4, 5: 1}


def read(path):
    return json.loads(path.read_text())


def quantiles(values):
    return np.quantile(values, [.1, .5, .9]).tolist() if len(values) else None


def metadata(snapshots, corrupted, scenario_ids, cached):
    pieces = {k: [] for k in ("scenario", "timestep", "node", "label", "family")}
    for sid in sorted(scenario_ids):
        indices = sorted((i for i, s in enumerate(snapshots) if s.scenario_id == sid),
                         key=lambda i: snapshots[i].timestep)[15:]
        for i in indices:
            c = corrupted[i]
            observed = c.pressure_mask.numpy() > 0
            nodes = np.flatnonzero(observed)
            n = len(nodes)
            pieces["scenario"].append(np.full(n, sid))
            pieces["timestep"].append(np.full(n, snapshots[i].timestep))
            pieces["node"].append(nodes)
            pieces["label"].append(c.pressure_anomaly.numpy()[observed])
            pieces["family"].append(np.full(n, c.attack_type_id))
    result = {k: np.concatenate(v) for k, v in pieces.items()}
    np.testing.assert_array_equal(result["label"], cached["labels"])
    np.testing.assert_array_equal(result["family"], cached["families"])
    np.testing.assert_array_equal(result["scenario"], cached["scenario"])
    return result


def reference_contamination_probe(snapshots, corrupted, event, meta, features, false_positives):
    """Oracle diagnostic: remove known attack inputs, not a proposed detector."""
    sid, start = event["scenario_id"], event["start_timestep"]
    stop = start+event["actual_steps"]
    indices = sorted((i for i, s in enumerate(snapshots) if s.scenario_id == sid),
                     key=lambda i: snapshots[i].timestep)
    values = np.stack([corrupted[i].pressure_obs.numpy() for i in indices])
    observed = np.stack([corrupted[i].pressure_mask.numpy() > 0 for i in indices])
    times = np.array([snapshots[i].timestep for i in indices])
    reference = joblib.load(ROOT/"runs/operational/blind_reference_probe_rank16/reference.joblib")
    original, _ = reference.predict(values, observed)
    oracle_mask = observed.copy()
    oracle_mask[np.ix_((times >= start) & (times < stop), event["targets"]["pressure"])] = False
    excluded, _ = reference.predict(values, oracle_mask)
    selected = false_positives & (meta["scenario"] == sid) & (meta["timestep"] >= start) & (meta["timestep"] < stop)
    selected &= ~np.isin(meta["node"], event["targets"]["pressure"])
    time_index = np.searchsorted(times, meta["timestep"][selected])
    node_index = meta["node"][selected]
    before = ((values-original)/reference.noise_scale_)[time_index, node_index]
    after = ((values-excluded)/reference.noise_scale_)[time_index, node_index]
    np.testing.assert_allclose(before, features["X"][selected, 0], rtol=1e-6, atol=1e-6)
    return {"scenario": sid, "start": start, "family": event["family"],
        "selection": "validation event with most false positives on unattacked sensors",
        "oracle": True, "deployable": False, "detector_scored": False,
        "intervention": "remove true attacked pressure inputs from reference during this event; observed values unchanged",
        "normal_false_positive_readings": int(selected.sum()),
        "absolute_standardized_residual_before": quantiles(np.abs(before)),
        "absolute_standardized_residual_after": quantiles(np.abs(after)),
        "above_3_before": int((np.abs(before) > 3).sum()),
        "above_3_after": int((np.abs(after) > 3).sum()),
        "interpretation": "Supports contaminated reference as a cause in this event; no deployable F1 improvement or bound is established"}


def main():
    splits = read(ROOT/"runs/operational/blind_reference_probe_rank16/splits.json")
    events = read(DATA/"events.json")
    with (DATA/"snapshots.pkl").open("rb") as stream:
        snapshots = pickle.load(stream)
    with (DATA/"corrupted.pkl").open("rb") as stream:
        corrupted = pickle.load(stream)
    names = read(RUN/"feature_names.json")
    summary = read(RUN/"summary.json")
    audit = read(RUN/"expert_audit.json")
    old_summary = read(ROOT/"runs/operational/blind_residual_experts_v1/summary.json")
    thresholds = {name: summary["dynamic"][f"mixture_{objective}"]["selection"]["threshold"]
                  for name, objective in (("overall", "legacy"), ("balanced", "macro_f1"))}
    expert_thresholds = np.array([audit["expert_matrix"][name]["threshold"] for name in MECHANISMS])
    report = {"test_evaluated": False, "models_trained": False,
        "scope": "descriptive development diagnostics, not an independent evaluation",
        "metadata_use": "evaluation and labelled oracle reference ablation only; true events/targets never enter deployable features or routing",
        "quantiles_order": [0.1, 0.5, 0.9], "thresholds": thresholds, "splits": {}}
    for split in ("train", "calibration", "validation"):
        a = dict(np.load(RUN/f"features_{split}.npz"))
        meta = metadata(snapshots, corrupted, splits[split], a)
        selected_events = [e for e in events if e["scenario_id"] in splits[split]]
        entry = {"event_counts": dict(Counter(e["family"] for e in selected_events)),
                 "families": {}, "events": []}
        p = None if split == "train" else dict(np.load(RUN/f"predictions_{split}.npz"))
        old_p = None if split == "train" else dict(np.load(ROOT/f"runs/operational/blind_residual_experts_v1/predictions_{split}.npz"))
        positive = a["labels"] > 0
        if p is not None:
            detector = p["mixture"] > thresholds["balanced"]
            expert_detects = p["experts"] > expert_thresholds
            entry["any_expert_global_threshold_OR"] = _binary_counts(expert_detects.any(1).astype(float), a["labels"], .5)
            entry["any_expert_OR_warning"] = "Diagnostic only; individual thresholds do not control the combined false-alarm budget"
        for fid, family in enumerate(FAMILY_NAMES):
            family_mask = a["families"] == fid
            pos = family_mask & positive
            row = {"positive_readings": int(pos.sum()), "normal_readings": int((family_mask & ~positive).sum())}
            if fid:
                row["feature_quantiles_positive"] = {name: quantiles(a["X"][pos, names.index(name)]) for name in (
                    "abs_residual", "normal_error_scale", "last_gap", "std_4", "std_16", "dynamic_abs_innovation",
                    "dynamic_sigma", "dynamic_past_std_48", "dynamic_innovation_rms_8")}
                if p is not None:
                    owner = OWNER[fid]
                    row["mixture_balanced"] = _binary_counts(p["mixture"][family_mask], a["labels"][family_mask], thresholds["balanced"])
                    row["owner_expert"] = _binary_counts(p["experts"][family_mask, owner], a["labels"][family_mask], expert_thresholds[owner])
                    row["owner_auprc"] = float(average_precision_score(a["labels"][family_mask], p["experts"][family_mask, owner]))
                    row["router_argmax_on_positives"] = dict(Counter(MECHANISMS[i] for i in p["routing"][pos].argmax(1)))
                    row["mean_routing_on_positives"] = dict(zip(MECHANISMS, p["routing"][pos].mean(0).tolist()))
                    misses = pos & ~detector
                    row["misses"] = {"n": int(misses.sum()),
                        "caught_by_owner_at_own_global_calibration_threshold": int((misses & expert_detects[:, owner]).sum()),
                        "caught_by_any_expert_at_individual_global_calibration_thresholds": int((misses & expert_detects.any(1)).sum()),
                        "owner_score_quantiles": quantiles(p["experts"][misses, owner]),
                        "owner_gate_quantiles": quantiles(p["routing"][misses, owner]),
                        "abs_residual_quantiles": quantiles(a["X"][misses, names.index("abs_residual")])}
            else:
                row["normal_features"] = {name: quantiles(a["X"][family_mask, names.index(name)]) for name in (
                    "abs_residual", "normal_error_scale", "dynamic_abs_innovation", "dynamic_sigma")}
            entry["families"][family] = row

        for event in selected_events:
            sid, start, duration = event["scenario_id"], event["start_timestep"], event["actual_steps"]
            event_mask = (meta["scenario"] == sid) & (meta["timestep"] >= start) & (meta["timestep"] < start+duration)
            target = event_mask & np.isin(meta["node"], event["targets"]["pressure"])
            pos = target & positive
            npos = int(pos.sum())
            row = {key: event[key] for key in ("scenario_id", "family", "start_timestep", "actual_steps", "ramp_steps", "noise_factor")}
            row.update({"positive_readings": npos, "targets": len(event["targets"]["pressure"]),
                        "observed_positive_counts_per_target": [int((pos & (meta["node"] == node)).sum()) for node in event["targets"]["pressure"]],
                        "positive_abs_residual_quantiles": quantiles(a["X"][pos, names.index("abs_residual")]),
                        "positive_error_scale_quantiles_m": quantiles(a["X"][pos, names.index("normal_error_scale")])})
            if event["family"] == "stealthy":
                amplitudes = []
                for node, magnitude in zip(event["targets"]["pressure"], event["magnitudes"]["pressure"]):
                    age = meta["timestep"][pos & (meta["node"] == node)]-start+1
                    amplitudes.extend((magnitude*np.minimum(age/event["ramp_steps"], 1.)).tolist())
                row["configured_drift_amplitude_quantiles_m"] = quantiles(amplitudes)
            if p is not None:
                row["detectors"] = {}
                for method, decision in (("previous", old_p["mixture"] > old_summary["results"]["mixture"]["threshold"]),
                    ("overall", p["mixture"] > thresholds["overall"]), ("balanced", detector)):
                    hits = pos & decision
                    delays = []
                    for node in event["targets"]["pressure"]:
                        node_hits = hits & (meta["node"] == node)
                        delays.append(int(meta["timestep"][node_hits].min()-start) if node_hits.any() else None)
                    row["detectors"][method] = {"true_positive_readings": int(hits.sum()),
                        "false_negative_readings": int((pos & ~decision).sum()),
                        "false_positives_on_unattacked_nodes": int((event_mask & ~target & decision).sum()),
                        "target_sensors_detected_at_least_once": sum(d is not None for d in delays),
                        "per_target_first_detection_delay_hours": delays,
                        "event_detected": bool(hits.any()),
                        "by_age": [{"hour_since_start": age, "positive_readings": int((pos & (meta["timestep"] == start+age)).sum()),
                                     "tp": int((hits & (meta["timestep"] == start+age)).sum())} for age in range(duration)]}
                    post = (meta["scenario"] == sid) & (meta["timestep"] >= start+duration) & (meta["timestep"] < start+duration+16) & ~positive
                    post_target = post & np.isin(meta["node"], event["targets"]["pressure"])
                    row["detectors"][method]["post_event_16h_false_positives"] = {
                        "formerly_targeted_sensors": int((post_target & decision).sum()),
                        "other_sensors": int((post & ~post_target & decision).sum()),
                        "observed_readings": int(post.sum())}
            entry["events"].append(row)
        assert sum(e["positive_readings"] for e in entry["events"]) == int(positive.sum())
        if split == "validation":
            for method, objective in (("overall", "legacy"), ("balanced", "macro_f1")):
                measured = _binary_counts(p["mixture"], a["labels"], thresholds[method])
                expected = summary["dynamic"][f"mixture_{objective}"]["validation"]["overall"]
                assert all(measured[k] == expected[k] for k in measured)
        if split == "validation":
            worst = max(entry["events"], key=lambda e: e["detectors"]["balanced"]["false_positives_on_unattacked_nodes"])
            event = next(e for e in selected_events if e["scenario_id"] == worst["scenario_id"] and e["start_timestep"] == worst["start_timestep"])
            report["reference_contamination_oracle"] = reference_contamination_probe(
                snapshots, corrupted, event, meta, a, ~positive & detector)
        report["splits"][split] = entry
    report["input_hashes"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (RUN/"summary.json", RUN/"expert_audit.json", RUN/"predictions_validation.npz", DATA/"events.json")}
    report["diagnostic_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = ROOT/"thesis_v2/outputs/expert_failure_diagnosis.json"
    output.write_text(json.dumps(report, indent=2))
    print(output)
    for split in ("train", "calibration", "validation"):
        print(split, report["splits"][split]["event_counts"])
    for family in FAMILY_NAMES[1:]:
        row = report["splits"]["validation"]["families"][family]
        print(family, "routing", row["router_argmax_on_positives"], "misses", row["misses"])
    for event in report["splits"]["validation"]["events"]:
        print("EVENT", event["scenario_id"], event["family"], "positives", event["positive_readings"],
              "detected sensors", event["detectors"]["balanced"]["target_sensors_detected_at_least_once"],
              "of", event["targets"], "age", event["detectors"]["balanced"]["by_age"])


if __name__ == "__main__":
    main()
