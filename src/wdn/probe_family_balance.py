"""Frozen-reference drift/noise features and family-balanced operating points."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import joblib
import numpy as np

from wdn.dynamic_residual_features import dynamic_residual_features
from wdn.operational_calibration import calibrate_threshold, family_summary
from wdn.probe_residual_experts import ResidualExpertMixture
from wdn.train_operational_moe import summarise


def augment(reference_run, output):
    source = Path(reference_run)
    config = json.loads((source/"config.json").read_text())
    data = Path(config["data_dir"])
    splits = json.loads((source/"splits.json").read_text())
    reference = joblib.load(source/"reference.joblib")
    with (data/"snapshots.pkl").open("rb") as f: snapshots = pickle.load(f)
    with (data/"corrupted.pkl").open("rb") as f: corrupted = pickle.load(f)
    base_names = json.loads((source/"feature_names.json").read_text())
    result = {}
    for split in ("train", "calibration", "validation"):
        cached = dict(np.load(source/f"features_{split}.npz"))
        added, labels, families, scenario_ids = [], [], [], []
        for sid in sorted(splits[split]):
            indices = sorted((i for i, s in enumerate(snapshots) if s.scenario_id == sid),
                             key=lambda i: snapshots[i].timestep)
            values = np.stack([corrupted[i].pressure_obs.numpy() for i in indices])
            mask = np.stack([corrupted[i].pressure_mask.numpy() > 0 for i in indices])
            pred, _ = reference.predict(values, mask)
            feats, names = dynamic_residual_features(values, mask, pred, reference.noise_scale_)
            endpoint = indices[15:]
            y = np.stack([corrupted[i].pressure_anomaly.numpy() for i in endpoint])
            family = np.array([corrupted[i].attack_type_id for i in endpoint])
            added.append(feats[mask[15:]])
            labels.append(y[mask[15:]])
            families.append(np.broadcast_to(family[:, None], y.shape)[mask[15:]])
            scenario_ids.append(np.full(int(mask[15:].sum()), sid))
        if not np.array_equal(np.concatenate(labels), cached["labels"]) or not np.array_equal(np.concatenate(families), cached["families"]):
            raise ValueError("Feature endpoints do not match the frozen baseline")
        cached["X"] = np.column_stack([cached["X"], np.concatenate(added)])
        cached["scenario"] = np.concatenate(scenario_ids)
        result[split] = cached
        np.savez_compressed(output/f"features_{split}.npz", **cached)
    return result, base_names+names


def evaluate(c, v, cal_labels, val_labels, max_fpr=.005):
    reports = {}
    for name in ("general", "mixture", "uniform"):
        for objective in ("legacy", "macro_f1"):
            key = f"{name}_{objective}"
            point = calibrate_threshold(c[name], cal_labels["labels"], cal_labels["families"],
                objective=objective, max_fpr=1. if objective == "legacy" else max_fpr,
                min_replay_f1=0. if objective == "legacy" else .5)
            r = summarise(v[name], val_labels["labels"], val_labels["families"], point["threshold"])
            reports[key] = {"selection": point, "validation": r, **family_summary(r)}
            print(key, "F1", round(r["overall"]["f1"],3), "families",
                  [round(r["per_family"][f]["f1"],3) for f in ("random", "replay", "stealthy", "noise", "targeted")],
                  "FPR", round(r["overall"]["fpr"],5), flush=True)
    return reports


def probe(reference_run, original_run, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    arrays, names = augment(reference_run, output)
    (output/"feature_names.json").write_text(json.dumps(names, indent=2))
    model = ResidualExpertMixture(names)
    # Dedicated drift/noise heads retain their original mechanism features
    # and gain the causal adaptation branch. Old saved models are unchanged.
    advanced = [i for i, name in enumerate(names) if name.startswith("dynamic_")]
    for expert in (1, 3, 4):
        model.profiles[expert] = np.unique(np.r_[model.profiles[expert], advanced]).astype(int)
    a = arrays["train"]
    model.fit(a["X"], a["labels"], a["families"])
    joblib.dump(model, output/"model.joblib")
    pred = {s: model.predict(arrays[s]["X"]) for s in ("calibration", "validation")}
    for split, values in pred.items():
        np.savez_compressed(output/f"predictions_{split}.npz", **values)
    original = {s: dict(np.load(Path(original_run)/f"predictions_{s}.npz")) for s in pred}
    # Calibration guard is a provisional engineering constraint, not a field
    # false-alarm SLA. Also report exact legacy selection as a control.
    report = {"test_evaluated": False, "reference_run": reference_run,
        "new_features": names, "max_calibration_fpr": .005,
        "selection_objective": "equal-weight mean F1 across five attack families",
        "source_sha256": {n: hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest()
            for n in ("probe_family_balance.py", "dynamic_residual_features.py", "operational_calibration.py", "probe_residual_experts.py")}}
    print("FROZEN ORIGINAL", flush=True)
    report["original"] = evaluate(original["calibration"], original["validation"], arrays["calibration"], arrays["validation"])
    print("DYNAMIC CONTEXT", flush=True)
    report["dynamic"] = evaluate(pred["calibration"], pred["validation"], arrays["calibration"], arrays["validation"])
    (output/"summary.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference_run", default="runs/operational/blind_reference_probe_rank16")
    p.add_argument("--original_run", default="runs/operational/blind_residual_experts_v1")
    p.add_argument("--output_dir", required=True)
    probe(**vars(p.parse_args()))
