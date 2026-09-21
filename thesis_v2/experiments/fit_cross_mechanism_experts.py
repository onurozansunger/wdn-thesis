"""Refit the weak-family experts to stay quiet inside other mechanisms' events.

EVAL-1 diagnosis: the whole detector's replay F1 fell to 0.7886 not because the
mixture missed replay — the mixture branch alone reaches 0.9046 there — but
because the noise specialist fired on 752 replay-event rows that carry no
attack. The confusion is intrinsic, not an artefact of the decision latency:
at zero latency the same branch produces 795 such false positives.

Replay is the family this hurts most: it has 37 negatives per positive inside
its own scope against about 18 for every other family, so a shared false-alarm
budget costs replay twice as much F1 as anyone else.

The fix is the one a mixture of experts is supposed to make: train each
specialist to discriminate against the *other mechanisms*, not only against
clean water. `CATEGORY_MASS` moves weight from clean rows to other-family rows
and the other-family sample is doubled. Architecture, features, latency and
hyperparameters are unchanged, and the strong-family mixture is untouched.

TRAIN only, 110 scenarios. Calibration, EVAL-1 and EVAL-2 are not read.

    python3 thesis_v2/experiments/fit_cross_mechanism_experts.py
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

from wdn.delayed_decision_features import delayed_decision_features
from wdn.models.delayed_decision import DelayedDecisionExperts
from wdn.models.seasonal_family import SeasonalFamilyExperts
from wdn.models.tuned_family_tree import TunedFamilyTreeConfig, TunedFamilyTrees
from wdn.run_expert_redesign import load_arrays, read_json, sha, write_json

DEPLOY = Path("runs/operational/seasonal_family_deployment_v2/full")
EXPANDED = Path("runs/operational/expanded_detector_v1")
NEW_TRAIN_SEEDS = (10811, 11811)
OUTPUT = Path("runs/operational/cross_mechanism_experts_v1")
DELTA = 3
CROSS_MASS = {"positive": .40, "hard": .25, "clean": .10, "other": .25}
OTHER_SAMPLE = 60000


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / "campaign.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()

    def status(phase, **details):
        write_json(OUTPUT / "status.json", {"phase": phase,
            "elapsed_seconds": time.monotonic() - started,
            "calibration_evaluated": False, "locked_eval_evaluated": False,
            "test_evaluated": False, **details})
        print(phase, details, flush=True)

    if (OUTPUT / "bundle.joblib").exists():
        print("Cross-mechanism experts already fitted; no refit", flush=True)
        return

    reference_bundle = joblib.load(EXPANDED / "bundle.joblib")
    bank = reference_bundle["names"]
    base_count = reference_bundle["base_feature_count"]
    del reference_bundle
    gc.collect()

    parts = []
    existing = load_arrays(DEPLOY / "features_train.npz")
    existing["X"] = np.column_stack((existing["X"],
        load_arrays(DEPLOY / "seasonal_train.npz")["X"])).astype(np.float32)
    existing["source"] = np.where(np.asarray(existing["scenario"]) >= 1000,
                                  np.asarray(existing["scenario"]) // 1000, 811)
    parts.append(existing)
    for seed in NEW_TRAIN_SEEDS:
        part = load_arrays(EXPANDED / f"features_seed{seed}.npz")
        part["source"] = np.full(len(part["labels"]), seed)
        parts.append(part)

    train = {}
    for key in ("X", "labels", "families", "event", "scenario", "timestep", "node", "early", "source"):
        pieces, offset = [], 0
        for part in parts:
            value = np.asarray(part[key])
            if key == "event":
                value = np.where(value >= 0, value + offset, -1)
                offset += int(value.max()) + 1 if value.size else 0
            pieces.append(value)
        train[key] = np.concatenate(pieces, axis=0)
    del parts, existing
    gc.collect()
    status("assembled full TRAIN", rows=len(train["labels"]),
           scenarios=int(len(np.unique(train["scenario"]))))

    status("fitting cross-mechanism seasonal specialists", mass=CROSS_MASS)
    seasonal = SeasonalFamilyExperts(bank, seed=4100, category_mass=CROSS_MASS,
                                     other_sample=OTHER_SAMPLE).fit(train)
    status("fitting the frozen-recipe drift auxiliary")
    config = TunedFamilyTreeConfig(max_leaf_nodes=23, min_samples_leaf=50,
        learning_rate=.1, l2_regularization=5., max_iter=150, seed=2621)
    tuned = TunedFamilyTrees(bank[:base_count], config).fit(
        {**train, "X": train["X"][:, :base_count]})

    status("building bounded forward-window evidence", delta=DELTA)
    forward, forward_names = delayed_decision_features(train, bank, DELTA)
    train["X"] = np.column_stack((train["X"], forward)).astype(np.float32)
    del forward
    gc.collect()
    names = list(bank) + list(forward_names)
    status("fitting the cross-mechanism delayed head", features=len(names))
    head = DelayedDecisionExperts(names, DELTA, seed=5100, category_mass=CROSS_MASS,
                                  other_sample=OTHER_SAMPLE).fit(train)

    joblib.dump({"seasonal": seasonal, "tuned_drift": tuned, "delayed": head,
                 "names": list(bank), "head_names": names,
                 "forward_names": list(forward_names), "delta": DELTA,
                 "base_feature_count": base_count, "category_mass": CROSS_MASS,
                 "score_blends": {"drift_seasonal": .90, "noise_fast": .95}},
                OUTPUT / "bundle.joblib")
    summary = {"status": "completed", "scope": "TRAIN-only refit on 110 scenarios",
        "change": "cross-mechanism training mass for the weak-family experts",
        "category_mass": CROSS_MASS, "other_sample": OTHER_SAMPLE,
        "declared_delta_hours": DELTA, "feature_count": len(names),
        "architecture_unchanged": True, "hyperparameters_unchanged": True,
        "strong_family_mixture_unchanged": True,
        "train_rows": int(len(train["labels"])),
        "train_scenarios": int(len(np.unique(train["scenario"]))),
        "bundle_sha256": sha(OUTPUT / "bundle.joblib"),
        "calibration_evaluated": False, "locked_eval_evaluated": False,
        "test_evaluated": False, "elapsed_seconds": time.monotonic() - started}
    write_json(OUTPUT / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
