"""Stage 4a: refit the whole detector on all 110 TRAIN scenarios.

Stage 4 found that the frozen detector is much weaker on a fresh generator seed
than on the three reused calibration scenarios, and that the weakest components
are the strong-family experts: the old mixture was fitted on 14 original TRAIN
scenarios, with a rank-16 reference and a 29-column bank, while the drift and
noise specialists already had 62 scenarios, a better reference and 116 columns.

This refits every classifier on the same footing:

* TRAIN is the 62 scenarios already used plus the 48 of seeds 10811 and 11811,
* the blind normal reference stays frozen at the 62-scenario fit, so the
  promoted specialists remain comparable,
* the feature bank is the 109-column expanded bank plus the seven seasonal
  features for every expert, strong families included,
* architectures, hyperparameters and blend weights are unchanged.

TRAIN only. Calibration, validation, the locked test and the locked EVAL seeds
are not read.

    python3 thesis_v2/experiments/fit_expanded_detector.py
"""
from __future__ import annotations

import fcntl
import gc
import json
import time
from pathlib import Path

import joblib
import lightgbm  # noqa: F401  loaded before scikit-learn's OpenMP
import numpy as np

from wdn.latency_deployment import specialist_bank
from wdn.models.seasonal_family import SeasonalFamilyExperts
from wdn.models.tuned_family_tree import TunedFamilyTreeConfig, TunedFamilyTrees
from wdn.probe_residual_experts import ResidualExpertMixture
from wdn.run_expert_redesign import CampaignData, load_arrays, read_json, sha, write_json
from wdn.train_weak_families import add_early

DATA = Path("data/thesis_v2")
DEPLOY = Path("runs/operational/seasonal_family_deployment_v2/full")
REFERENCE = DEPLOY / "reference.joblib"
NEW_TRAIN_SEEDS = (10811, 11811)
OUTPUT = Path("runs/operational/expanded_detector_v1")


def atomic_npz(path, **arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / "campaign.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()

    def status(phase, **details):
        write_json(OUTPUT / "status.json", {"phase": phase,
            "elapsed_seconds": time.monotonic() - started,
            "calibration_evaluated": False, "validation_evaluated": False,
            "test_evaluated": False, "locked_eval_evaluated": False, **details})
        print(phase, details, flush=True)

    if (OUTPUT / "bundle.joblib").exists():
        print("Expanded detector already fitted; no refit", flush=True)
        return

    reference = joblib.load(REFERENCE)
    deployment = Path("runs/operational/seasonal_family_deployment_v2")
    names = read_json(deployment / "feature_names.json")
    seasonal_names = read_json(deployment / "seasonal_feature_names.json")
    bank = list(names) + list(seasonal_names)

    parts = []
    existing = load_arrays(DEPLOY / "features_train.npz")
    existing["X"] = np.column_stack((existing["X"],
        load_arrays(DEPLOY / "seasonal_train.npz")["X"])).astype(np.float32)
    parts.append(existing)
    status("loaded frozen 62-scenario TRAIN", rows=len(existing["labels"]))

    for seed in NEW_TRAIN_SEEDS:
        cache = OUTPUT / f"features_seed{seed}.npz"
        if not cache.exists():
            status("building TRAIN features for a new seed", seed=seed)
            data = CampaignData(DATA / f"operational_train_expansion2_seed{seed}")
            arrays, found = specialist_bank(data, list(range(24)), reference)
            if found != bank:
                raise ValueError("Feature schema differs on the new TRAIN seed")
            arrays = add_early(arrays, data.events)
            arrays["scenario"] = np.asarray(arrays["scenario"]) + seed * 1000
            atomic_npz(cache, **arrays)
            del data, arrays
            gc.collect()
        parts.append(load_arrays(cache))

    train = {}
    for key in ("X", "labels", "families", "event", "scenario", "timestep", "node", "early"):
        pieces = []
        offset = 0
        for part in parts:
            value = np.asarray(part[key])
            if key == "event":
                value = np.where(value >= 0, value + offset, -1)
                offset += int(value.max()) + 1 if value.size else 0
            pieces.append(value)
        train[key] = np.concatenate(pieces, axis=0)
    del parts
    gc.collect()
    status("assembled full TRAIN", rows=len(train["labels"]),
           scenarios=int(len(np.unique(train["scenario"]))),
           positives=int((train["labels"] > 0).sum()))

    status("fitting strong-family mixture on the full bank")
    mixture = ResidualExpertMixture(bank, seed=601).fit(
        train["X"], train["labels"], train["families"])
    status("fitting the frozen-recipe drift auxiliary")
    config = TunedFamilyTreeConfig(max_leaf_nodes=23, min_samples_leaf=50,
        learning_rate=.1, l2_regularization=5., max_iter=150, seed=2621)
    tuned = TunedFamilyTrees(names, config).fit(
        {**train, "X": train["X"][:, :len(names)]})
    status("fitting the seasonal drift/noise specialists")
    seasonal = SeasonalFamilyExperts(bank, seed=4100).fit(train)

    joblib.dump({"mixture": mixture, "seasonal": seasonal, "tuned_drift": tuned,
                 "names": bank, "base_feature_count": len(names),
                 "score_blends": {"drift_seasonal": .90, "noise_fast": .95}},
                OUTPUT / "bundle.joblib")
    summary = {"status": "completed", "scope": "TRAIN-only refit on 110 scenarios",
        "train_scenarios": int(len(np.unique(train["scenario"]))),
        "train_rows": int(len(train["labels"])),
        "train_positives": int((train["labels"] > 0).sum()),
        "reference": "frozen 62-scenario expanded TRAIN blind reference",
        "reference_sha256": sha(REFERENCE), "feature_count": len(bank),
        "new_train_seeds": list(NEW_TRAIN_SEEDS),
        "architecture_unchanged": True, "hyperparameters_unchanged": True,
        "calibration_evaluated": False, "validation_evaluated": False,
        "test_evaluated": False, "locked_eval_evaluated": False,
        "bundle_sha256": sha(OUTPUT / "bundle.joblib"),
        "elapsed_seconds": time.monotonic() - started}
    write_json(OUTPUT / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
