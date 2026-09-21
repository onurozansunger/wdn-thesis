"""Fit the Stage 1 delayed-decision head on all 110 TRAIN scenarios.

Stage 1 showed on generator-held OOF that blending the delayed-decision head
with the forward maximum gives the best weak-family ranking of the campaign
(drift 0.8124, noise 0.8390 against 0.8106 and 0.8121 for the forward maximum
alone), at the cost of a longer post-event false-positive tail. Stage 4 on a
fresh calibration seed lands below target for both weak families, so the head
is fitted on the full TRAIN corpus and offered to Stage 4 as a second,
predeclared specialist variant. Selection between the two happens on
calibration; the locked EVAL seeds are not read.

Same architecture, same fixed recipe, no Optuna, declared latency 3 hours.

    python3 thesis_v2/experiments/fit_delayed_head_full.py
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
from wdn.run_expert_redesign import load_arrays, read_json, sha, write_json

DEPLOY = Path("runs/operational/seasonal_family_deployment_v2/full")
EXPANDED = Path("runs/operational/expanded_detector_v1")
NEW_TRAIN_SEEDS = (10811, 11811)
OUTPUT = Path("runs/operational/delayed_head_full_v1")
DELTA = 3


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
        print("Delayed head already fitted; no refit", flush=True)
        return

    bank = joblib.load(EXPANDED / "bundle.joblib")["names"]
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

    status("building bounded forward-window evidence", delta=DELTA)
    forward, forward_names = delayed_decision_features(train, bank, DELTA)
    train["X"] = np.column_stack((train["X"], forward)).astype(np.float32)
    del forward
    gc.collect()
    names = list(bank) + list(forward_names)

    status("fitting the delayed-decision experts", features=len(names))
    model = DelayedDecisionExperts(names, DELTA, seed=5100).fit(train)
    joblib.dump({"delayed": model, "names": names, "bank": list(bank),
                 "forward_names": list(forward_names), "delta": DELTA},
                OUTPUT / "bundle.joblib")
    summary = {"status": "completed", "scope": "TRAIN-only fit on 110 scenarios",
        "declared_delta_hours": DELTA, "feature_count": len(names),
        "metadata": model.metadata(), "train_rows": int(len(train["labels"])),
        "train_scenarios": int(len(np.unique(train["scenario"]))),
        "detector_bundle_sha256": sha(EXPANDED / "bundle.joblib"),
        "bundle_sha256": sha(OUTPUT / "bundle.joblib"),
        "calibration_evaluated": False, "locked_eval_evaluated": False,
        "test_evaluated": False, "elapsed_seconds": time.monotonic() - started}
    write_json(OUTPUT / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
