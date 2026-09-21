"""Validation-only diagnostic: can a blind normal reference expose attacks?

This is NOT a completed MoE. It isolates representation quality using one
fixed tree detector, plus a simple absolute-residual control. No clean target
is used to fit the reference or form features, and test examples are excluded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from wdn.models.blind_reference import BlindPressureReference
from wdn.train_operational_moe import select_threshold, summarise


def residual_features(values, observed, predicted, support, scale, window=16):
    """Causal sensor features; caller must process one scenario at a time."""
    values = np.asarray(values)
    observed = np.asarray(observed, dtype=bool)
    safe = np.where(observed, values, 0.)
    residual = np.where(observed, (safe-predicted)/scale, 0.)
    rows = []
    names = ["residual", "abs_residual", "reference_support", "normal_error_scale"]
    for size in (4, 8, 16):
        names += [f"{n}_{size}" for n in ("coverage", "mean", "std", "rms", "max_abs", "slope", "sign_consistency")]
    names += ["lag_advantage", "lag_support", "last_gap", "residual_rate"]
    for t in range(window-1, len(values)):
        cols = [residual[t], np.abs(residual[t]), support[t], np.broadcast_to(scale, residual[t].shape)]
        for size in (4, 8, 16):
            start = max(0, t-size+1)
            r, w = residual[start:t+1], observed[start:t+1].astype(float)
            n = w.sum(0).clip(min=1)
            mean = r.sum(0)/n
            std = np.sqrt(((r-mean)**2*w).sum(0)/n)
            time_index = np.arange(len(r))[:, None]
            centered_time = time_index-(time_index*w).sum(0)/n
            slope = (centered_time*r).sum(0)/np.maximum((centered_time**2*w).sum(0), 1.)
            cols.extend([w.mean(0), mean, std, np.sqrt((r*r).sum(0)/n),
                         np.abs(r).max(0), slope, np.abs(np.sign(r).sum(0))/n])
        improvements, counts = [], []
        for lag in range(1, 7):
            # Same endpoint mask in current and delayed comparisons.
            start = max(lag, t-window+1)
            w = observed[start:t+1] & (support[start:t+1] > 0) & (support[start-lag:t+1-lag] > 0)
            n = w.sum(0)
            current = np.abs(safe[start:t+1]-predicted[start:t+1])/scale
            delayed = np.abs(safe[start:t+1]-predicted[start-lag:t+1-lag])/scale
            advantage = ((current-delayed)*w).sum(0)/n.clip(min=1)
            improvements.append(np.where(n >= 2, advantage, -np.inf))
            counts.append(n/len(w))
        improvements = np.asarray(improvements)
        best = improvements.argmax(0)
        adv = np.take_along_axis(improvements, best[None], axis=0)[0]
        lag_count = np.take_along_axis(np.asarray(counts), best[None], axis=0)[0]
        supported = np.isfinite(adv)
        prior = observed[max(0, t-window+1):t]
        times = np.arange(max(0, t-window+1), t)[:, None]
        previous = np.where(prior, times, -1).max(0)
        pair = observed[t] & (previous >= 0)
        gap = np.where(pair, t-previous, 0.)
        previous_residual = residual[np.maximum(previous, 0), np.arange(len(previous))]
        rate = np.where(pair, (residual[t]-previous_residual)/np.maximum(gap, 1.), 0.)
        cols += [np.where(supported, adv, 0.), np.where(supported, lag_count, 0.), gap, rate]
        rows.append(np.column_stack(cols))
    return np.asarray(rows, dtype=np.float32), names


def probe(data_dir, baseline_run, output_dir, rank=4, seed=601):
    started = time.monotonic()
    data_dir, baseline_run, output = map(Path, (data_dir, baseline_run, output_dir))
    output.mkdir(parents=True, exist_ok=False)
    config = {"data_dir": str(data_dir), "baseline_run": str(baseline_run), "rank": rank,
              "seed": seed, "window": 16, "test_evaluated": False,
              "reference_training": "observed pressure in clean TRAIN episodes only",
              "detector": "fixed HistGradientBoosting, diagnostic not MoE"}
    (output/"config.json").write_text(json.dumps(config, indent=2))
    sources = ("probe_blind_reference.py", "models/blind_reference.py", "train_operational_moe.py")
    (output/"source_sha256.json").write_text(json.dumps({p: hashlib.sha256(
        (Path(__file__).parent/p).read_bytes()).hexdigest() for p in sources}, indent=2))
    splits = json.loads((baseline_run/"splits.json").read_text())
    (output/"splits.json").write_text(json.dumps(splits, indent=2))
    import yaml
    data_config = yaml.safe_load((data_dir/"generate_config.yaml").read_text())
    if any(data_config[f"missing_rate_{name}"] != .5 for name in ("pressure", "flow")):
        raise ValueError("This probe requires the unchanged 50% missing benchmark")
    with (data_dir/"snapshots.pkl").open("rb") as f:
        snapshots = pickle.load(f)
    with (data_dir/"corrupted.pkl").open("rb") as f:
        corrupted = pickle.load(f)
    # Extract only observed pressure, masks, labels and scenario metadata.
    # No pressure_true, flow_true or held-out test arrays enter this probe.
    arrays = {}
    for name in ("train", "calibration", "validation"):
        indices = [i for i, s in enumerate(snapshots) if s.scenario_id in splits[name]]
        arrays[name] = {
            "values": np.stack([corrupted[i].pressure_obs.numpy() for i in indices]),
            "mask": np.stack([corrupted[i].pressure_mask.numpy() > 0 for i in indices]),
            "labels": np.stack([corrupted[i].pressure_anomaly.numpy() for i in indices]),
            "family": np.array([corrupted[i].attack_type_id for i in indices]),
            "scenario": np.array([snapshots[i].scenario_id for i in indices]),
            "timestep": np.array([snapshots[i].timestep for i in indices]),
        }
    train = arrays["train"]
    normal = train["family"] == 0
    reference = BlindPressureReference(rank=rank).fit(train["values"][normal], train["mask"][normal])
    joblib.dump(reference, output/"reference.joblib")
    print("Reference fitted from noisy normal training observations", flush=True)
    features = {}
    for name, a in arrays.items():
        X, y, family, normal_abs_errors = [], [], [], []
        for scenario in np.unique(a["scenario"]):
            selected = np.flatnonzero(a["scenario"] == scenario)
            selected = selected[np.argsort(a["timestep"][selected])]
            p, m = a["values"][selected], a["mask"][selected]
            pred, support = reference.predict(p, m)
            x, names = residual_features(p, m, pred, support, reference.noise_scale_, window=16)
            endpoint = selected[15:]
            mask = a["mask"][endpoint]
            X.append(x[mask]); y.append(a["labels"][endpoint][mask])
            family.append(np.broadcast_to(a["family"][endpoint, None], mask.shape)[mask])
            normal_mask = mask & (a["family"][endpoint, None] == 0)
            normal_abs_errors.extend(np.abs(p[15:]-pred[15:])[normal_mask].tolist())
        features[name] = {"X": np.concatenate(X), "labels": np.concatenate(y), "families": np.concatenate(family)}
        features[name]["normal_observed_reference_mae_m"] = float(np.mean(normal_abs_errors))
        np.savez_compressed(output/f"features_{name}.npz", **features[name])
        print(name, features[name]["X"].shape, "reference normal MAE", np.mean(normal_abs_errors), flush=True)
    (output/"feature_names.json").write_text(json.dumps(names, indent=2))
    # Keep all train positives and a reproducible sample of negatives. Correct
    # the negative sampling weights; evaluation prevalence is never sampled.
    train = features["train"]
    positives = np.flatnonzero(train["labels"] > .5)
    negatives = np.flatnonzero(train["labels"] <= .5)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(negatives, min(len(negatives), 60000), replace=False)
    subset = np.concatenate([positives, chosen])
    weights = np.r_[np.ones(len(positives)), np.full(len(chosen), len(negatives)/len(chosen))]
    model = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15,
        min_samples_leaf=30, learning_rate=.08, l2_regularization=10.,
        early_stopping=False, random_state=seed)
    model.fit(train["X"][subset], train["labels"][subset], sample_weight=weights)
    joblib.dump(model, output/"detector.joblib")
    cal, val = features["calibration"], features["validation"]
    report = {"status": "representation_probe_not_MoE", "test_evaluated": False,
              "rank": rank, "clean_truth_used": False, "results": {}}
    for detector in ("absolute_residual", "residual_tree"):
        if detector == "absolute_residual":
            cs, vs = cal["X"][:, 1], val["X"][:, 1]
            cs, vs = cs/(1+cs), vs/(1+vs)
        else:
            cs, vs = model.predict_proba(cal["X"])[:, 1], model.predict_proba(val["X"])[:, 1]
        threshold = select_threshold(cs, cal["labels"], cal["families"])
        result = summarise(vs, val["labels"], val["families"], threshold)
        report["results"][detector] = result
        np.savez_compressed(output/f"scores_{detector}.npz", calibration=cs, validation=vs)
        print(detector, "F1", result["overall"]["f1"], "replay", result["per_family"]["replay"]["f1"], flush=True)
    report["seconds"] = time.monotonic()-started
    report["normal_reference_mae_m"] = {name: a["normal_observed_reference_mae_m"] for name, a in features.items()}
    (output/"summary.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="data/thesis_v2/operational_modena_seed811")
    parser.add_argument("--baseline_run", default="runs/operational/mechanism_experts_v2")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--rank", type=int, default=4)
    args = parser.parse_args()
    probe(**vars(args))
