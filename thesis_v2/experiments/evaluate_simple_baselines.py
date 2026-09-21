"""Leakage-safe label-free baselines for thesis-v2 sensor attacks.

The detector scores do not use attack labels. Labels are used only to select a
decision threshold on validation data and to report test metrics, matching the
neural-model protocol.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


V2 = Path(__file__).resolve().parents[1]
ROOT = V2.parent
OUT = V2 / "outputs" / "baselines"
EVALUATION_MIN_TIMESTEP = 5


def scenario_split(snapshots: list, seed: int = 42):
    scenarios = sorted({int(s.scenario_id) for s in snapshots})
    rng = np.random.default_rng(seed)
    rng.shuffle(scenarios)
    n_train = int(len(scenarios) * 0.70)
    n_val = int(len(scenarios) * 0.15)
    return (
        set(scenarios[:n_train]),
        set(scenarios[n_train:n_train + n_val]),
        set(scenarios[n_train + n_val:]),
    )


def robust_location_scale(values: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray([x for row in values for x in row], dtype=np.float64)
    global_median = float(np.median(flat)) if flat.size else 0.0
    global_scale = (
        float(1.4826 * np.median(np.abs(flat - global_median)))
        if flat.size else 1.0
    )
    global_scale = max(global_scale, 1e-6)
    location = np.full(len(values), global_median, dtype=np.float64)
    scale = np.full(len(values), global_scale, dtype=np.float64)
    for node, row in enumerate(values):
        if len(row) < 10:
            continue
        arr = np.asarray(row, dtype=np.float64)
        median = float(np.median(arr))
        mad = float(1.4826 * np.median(np.abs(arr - median)))
        location[node] = median
        scale[node] = max(mad, 1e-6)
    return location, scale


def fit_profile(snapshots: list, corrupted: list, train_scenarios: set[int]):
    n_nodes = int(snapshots[0].pressure_true.shape[0])
    by_slot: dict[int, list[list[float]]] = defaultdict(
        lambda: [[] for _ in range(n_nodes)]
    )
    by_node = [[] for _ in range(n_nodes)]
    for snap, corr in zip(snapshots, corrupted):
        if int(snap.scenario_id) not in train_scenarios:
            continue
        observed = np.asarray(corr.pressure_mask).astype(bool)
        values = np.asarray(corr.pressure_obs, dtype=np.float64)
        slot = int(snap.timestep)
        for node in np.flatnonzero(observed):
            value = float(values[node])
            by_slot[slot][int(node)].append(value)
            by_node[int(node)].append(value)

    global_values = [value for row in by_node for value in row]
    global_median = float(np.median(global_values)) if global_values else 0.0
    node_median = np.asarray(
        [np.median(row) if row else global_median for row in by_node],
        dtype=np.float64,
    )
    profile = {
        slot: np.asarray(
            [np.median(row) if row else node_median[node] for node, row in enumerate(rows)],
            dtype=np.float64,
        )
        for slot, rows in by_slot.items()
    }

    residuals = [[] for _ in range(n_nodes)]
    for snap, corr in zip(snapshots, corrupted):
        if int(snap.scenario_id) not in train_scenarios:
            continue
        observed = np.asarray(corr.pressure_mask).astype(bool)
        values = np.asarray(corr.pressure_obs, dtype=np.float64)
        expected = profile.get(int(snap.timestep), node_median)
        for node in np.flatnonzero(observed):
            residuals[int(node)].append(float(values[node] - expected[node]))
    residual_location, residual_scale = robust_location_scale(residuals)
    return profile, node_median, residual_location, residual_scale


def fit_persistence_scale(snapshots: list, corrupted: list, train_scenarios: set[int]):
    n_nodes = int(snapshots[0].pressure_true.shape[0])
    differences = [[] for _ in range(n_nodes)]
    last: dict[int, np.ndarray] = {}
    last_mask: dict[int, np.ndarray] = {}
    for snap, corr in zip(snapshots, corrupted):
        scenario = int(snap.scenario_id)
        if scenario not in train_scenarios:
            continue
        values = np.asarray(corr.pressure_obs, dtype=np.float64)
        observed = np.asarray(corr.pressure_mask).astype(bool)
        if scenario in last:
            valid = observed & last_mask[scenario]
            for node in np.flatnonzero(valid):
                differences[int(node)].append(float(values[node] - last[scenario][node]))
        last[scenario] = values.copy()
        last_mask[scenario] = observed.copy()
    return robust_location_scale(differences)


def collect_profile_scores(
    snapshots: list,
    corrupted: list,
    scenarios: set[int],
    profile: dict[int, np.ndarray],
    fallback: np.ndarray,
    location: np.ndarray,
    scale: np.ndarray,
    min_timestep: int = EVALUATION_MIN_TIMESTEP,
):
    scores, labels = [], []
    for snap, corr in zip(snapshots, corrupted):
        if int(snap.scenario_id) not in scenarios:
            continue
        if int(snap.timestep) < min_timestep:
            continue
        observed = np.asarray(corr.pressure_mask).astype(bool)
        values = np.asarray(corr.pressure_obs, dtype=np.float64)
        expected = profile.get(int(snap.timestep), fallback)
        residual = values - expected
        scores.append(np.abs((residual[observed] - location[observed]) / scale[observed]))
        labels.append(np.asarray(corr.pressure_anomaly)[observed].astype(int))
    return np.concatenate(scores), np.concatenate(labels)


def collect_persistence_scores(
    snapshots: list,
    corrupted: list,
    scenarios: set[int],
    location: np.ndarray,
    scale: np.ndarray,
    min_timestep: int = EVALUATION_MIN_TIMESTEP,
):
    scores, labels = [], []
    last: dict[int, np.ndarray] = {}
    last_mask: dict[int, np.ndarray] = {}
    for snap, corr in zip(snapshots, corrupted):
        scenario = int(snap.scenario_id)
        if scenario not in scenarios:
            continue
        values = np.asarray(corr.pressure_obs, dtype=np.float64)
        observed = np.asarray(corr.pressure_mask).astype(bool)
        if scenario in last and int(snap.timestep) >= min_timestep:
            valid = observed & last_mask[scenario]
            if valid.any():
                residual = values - last[scenario]
                scores.append(np.abs((residual[valid] - location[valid]) / scale[valid]))
                labels.append(np.asarray(corr.pressure_anomaly)[valid].astype(int))
        last[scenario] = values.copy()
        last_mask[scenario] = observed.copy()
    return np.concatenate(scores), np.concatenate(labels)


def select_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 501)))
    best_threshold, best_f1 = float(candidates[0]), -1.0
    for threshold in candidates:
        value = f1_score(labels, scores >= threshold, zero_division=0)
        if value > best_f1:
            best_threshold, best_f1 = float(threshold), float(value)
    return best_threshold, best_f1


def metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    predictions = scores >= threshold
    return {
        "n": int(labels.size),
        "prevalence": float(labels.mean()),
        "threshold": float(threshold),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


def evaluate(data_dir: Path) -> dict:
    with (data_dir / "snapshots.pkl").open("rb") as stream:
        snapshots = pickle.load(stream)
    with (data_dir / "corrupted.pkl").open("rb") as stream:
        corrupted = pickle.load(stream)
    train_scenarios, val_scenarios, test_scenarios = scenario_split(snapshots)
    if not train_scenarios or not val_scenarios or not test_scenarios:
        raise ValueError("dataset needs enough scenarios for non-empty train/val/test splits")

    profile, fallback, profile_location, profile_scale = fit_profile(
        snapshots, corrupted, train_scenarios
    )
    persistence_location, persistence_scale = fit_persistence_scale(
        snapshots, corrupted, train_scenarios
    )

    methods = {}
    collectors = {
        "historical_profile": lambda split: collect_profile_scores(
            snapshots, corrupted, split, profile, fallback,
            profile_location, profile_scale,
        ),
        "persistence": lambda split: collect_persistence_scores(
            snapshots, corrupted, split, persistence_location, persistence_scale,
        ),
    }
    for name, collect in collectors.items():
        val_scores, val_labels = collect(val_scenarios)
        threshold, val_f1 = select_threshold(val_scores, val_labels)
        test_scores, test_labels = collect(test_scenarios)
        methods[name] = {
            "validation_f1": val_f1,
            "test": metrics(test_scores, test_labels, threshold),
        }

    return {
        "dataset": str(data_dir.relative_to(ROOT)),
        "split": {
            "seed": 42,
            "evaluation_min_timestep": EVALUATION_MIN_TIMESTEP,
            "train_scenarios": sorted(train_scenarios),
            "validation_scenarios": sorted(val_scenarios),
            "test_scenarios": sorted(test_scenarios),
        },
        "methods": methods,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for data_dir in args.data_dirs:
        resolved = data_dir if data_dir.is_absolute() else ROOT / data_dir
        result = evaluate(resolved)
        safe_name = result["dataset"].replace("/", "__")
        path = OUT / f"{safe_name}.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\n{result['dataset']}")
        for name, report in result["methods"].items():
            test = report["test"]
            print(
                f"  {name:20s} F1={test['f1']:.3f} "
                f"AUPRC={test['auprc']:.3f} AUROC={test['auroc']:.3f} "
                f"P={test['precision']:.3f} R={test['recall']:.3f}"
            )
        print(f"  wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
