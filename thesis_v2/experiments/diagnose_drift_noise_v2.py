"""Diagnose frozen drift/noise predictions; no fitting or threshold changes.

Only cached TRAIN/calibration/validation examples are inspected. Event metadata
are evaluation annotations, never inference inputs. Conditional bookkeeping
bounds below are not achievable model scores or theoretical detection limits.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np

from wdn.train_operational_moe import summarise


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/operational/mechanism_redesign_v1"
FAMILIES = {3: "stealthy", 4: "noise"}
EXPERTS = ("general", "abrupt", "replay", "drift", "noise")


def read(path):
    return json.loads(path.read_text())


def main():
    summary = read(RUN / "summary.json")
    frozen = read(RUN / "selection_frozen.json")
    assert summary["status"] == "completed" and not summary["test_evaluated"]
    assert frozen == summary["selection"]
    signatures = summary["source_and_inputs"]
    for name, expected in signatures["source"].items():
        assert hashlib.sha256((ROOT / "src/wdn" / name).read_bytes()).hexdigest() == expected
    splits = signatures["splits"]
    events = read(ROOT / "data/thesis_v2/operational_modena_seed811/events.json")
    assert hashlib.sha256((ROOT / "data/thesis_v2/operational_modena_seed811/events.json").read_bytes()).hexdigest() == signatures["data_files"]["events.json"]
    folder = RUN / f"trial_{frozen['trial']:04d}"
    model = joblib.load(folder / "model.joblib")
    audit = read(folder / "expert_audit.json")["expert_matrix"]
    names = model["experts"].names
    threshold = frozen["selection"]["threshold"]
    fusion = model["fusion"]
    fusion_names = list(EXPERTS) + [names[i] for i in fusion.columns]
    report = {
        "scope": "read-only development diagnostics, no new model or threshold",
        "selected_trial": frozen["trial"], "test_extracted": False,
        "event_annotations_used_for_evaluation_only": True,
        "source_hashes_unchanged": True,
        "fusion_standardised_coefficients": dict(zip(fusion_names, fusion.model.coef_[0].tolist())),
        "events": [], "split_support": {},
    }
    for split in ("train", "calibration", "validation"):
        a = dict(np.load(RUN / f"full/features_{split}.npz"))
        assert set(a["scenario"]) == set(splits[split])
        assert set(a["scenario"]).isdisjoint(splits["test"])
        report["split_support"][split] = {}
        predictions = None if split == "train" else dict(np.load(folder / f"predictions_{split}.npz"))
        if split == "validation":
            reproduced = summarise(predictions["mixture"], a["labels"], a["families"], threshold)
            assert reproduced == summary["trials"][str(frozen["trial"])]["validation"]
        for fid, family in FAMILIES.items():
            relevant = [e for e in events if e["family"] == family and e["scenario_id"] in splits[split]]
            report["split_support"][split][family] = {
                "events": len(relevant), "independent_scenarios": len({e["scenario_id"] for e in relevant}),
                "positives": int(((a["families"] == fid) & (a["labels"] > 0)).sum()),
            }
            for event in relevant:
                sid, start, duration = event["scenario_id"], event["start_timestep"], event["actual_steps"]
                active = (a["scenario"] == sid) & (a["timestep"] >= start) & (a["timestep"] < start + duration)
                positive = active & (a["labels"] > 0)
                assert np.all(a["families"][positive] == fid)
                early = positive & (a["timestep"] < start + 3)
                row = {"split": split, "family": family, "scenario": sid, "start": start,
                       "duration_h": duration, "ramp_h": event["ramp_steps"],
                       "noise_factor": event["noise_factor"], "positives": int(positive.sum()),
                       "early_3h_positives": int(early.sum())}
                if family == "stealthy":
                    offsets = dict(zip(event["targets"]["pressure"], event["magnitudes"]["pressure"]))
                    amplitudes = [offsets[int(node)] * min((int(t) - start + 1) / event["ramp_steps"], 1.) for node, t in zip(a["node"][positive], a["timestep"][positive])]
                    row["configured_amplitude_m_q10_q50_q90"] = np.quantile(amplitudes, [.1, .5, .9]).tolist()
                if predictions is not None:
                    decision = predictions["mixture"] > threshold
                    missed = positive & ~decision
                    owner = 3 if family == "stealthy" else 4
                    owner_hit = predictions["experts"][:, owner] > audit[EXPERTS[owner]]["threshold"]
                    per_sensor = []
                    for node in event["targets"]["pressure"]:
                        ix = np.flatnonzero(positive & (a["node"] == node))
                        hit_ix = ix[decision[ix]]
                        first_hit = int(a["timestep"][hit_ix].min()) if len(hit_ix) else None
                        after = int((~decision[ix] & (a["timestep"][ix] > first_hit)).sum()) if first_hit is not None else 0
                        per_sensor.append({"node": node, "positives": len(ix), "tp": len(hit_ix),
                                           "misses_after_first_same_event_hit": after})
                    row.update(tp=int((positive & decision).sum()), fp=int((active & ~positive & decision).sum()),
                               missed=int(missed.sum()), early_3h_tp=int((early & decision).sum()),
                               owner_recovers_at_frozen_global_threshold=int((missed & owner_hit).sum()),
                               misses_after_earlier_same_sensor_hit=sum(x["misses_after_first_same_event_hit"] for x in per_sensor),
                               sensors_never_hit=sum(x["tp"] == 0 for x in per_sensor), per_sensor=per_sensor)
                    row["missed_abs_residual_q10_q50_q90"] = np.quantile(a["X"][missed, names.index("abs_residual")], [.1, .5, .9]).tolist() if missed.any() else None
                    row["tp_needed_if_zero_fp"] = {str(target): math.ceil(target * row["positives"] / (2 - target)) for target in (.8, .9)}
                    if family == "stealthy":
                        hypothetical_tp = row["early_3h_tp"] + row["positives"] - row["early_3h_positives"]
                        row["conditional_f1_if_early_hits_unchanged_late_perfect_and_zero_fp"] = 2 * hypothetical_tp / (row["positives"] + hypothetical_tp)
                report["events"].append(row)
    output = ROOT / "thesis_v2/outputs/drift_noise_diagnosis_v2.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "validation": [e for e in report["events"] if e["split"] == "validation"],
                      "drift_fusion_coefficient": report["fusion_standardised_coefficients"]["drift"]}, indent=2))


if __name__ == "__main__":
    main()
