"""Stage 1: does a declared decision latency close the weak-family gap?

Expanded-TRAIN generator-held OOF only. Four candidates are scored on exactly
the same rows, all predeclared before the run:

    frozen         the promoted seasonal expert score, delta = 0
    maxpool        forward maximum of the frozen score over [t, t + delta]
    delayed        a new head over the seasonal bank plus forward-window
                   evidence, same fixed LightGBM recipe, no Optuna
    delayed_blend  50/50 logit blend of `delayed` and `maxpool`

Stage 1 passes for a family when a candidate reaches oracle F1 >= 0.80 with the
control's clean-FPR allowance and post-event false-positive budget held. Ties
are broken in the order listed above, so a cheaper candidate wins a tie.

The base feature bank, the seasonal features and the frozen fold predictions
are all reused from the completed runs; no reference is refitted and no data is
generated. Calibration, validation, the locked test and the new locked EVAL
seeds are not read.

    python3 thesis_v2/experiments/screen_delayed_decision_head.py
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np

from wdn.delayed_decision_features import delayed_decision_features
from wdn.expanded_train_data import ExpandedTrainData
from wdn.models.delayed_decision import DelayedDecisionExperts
from wdn.run_expert_redesign import load_arrays, read_json, sha, write_json
from wdn.screen_family_specific import _best_f1
from wdn.screen_recovery import expert_diagnostics

BASE = Path("runs/operational/expanded_train_weak_experts_v1")
SEASONAL = Path("runs/operational/seasonal_family_experts_v3")
PLAN = Path("thesis_v2/ALL_FAMILIES_080_PLAN.md")
ORIGINAL = Path("data/thesis_v2/operational_modena_seed811")
EXPANSIONS = tuple(Path(f"data/thesis_v2/operational_train_expansion_seed{seed}")
                   for seed in (1811, 2811, 3811))
SPLITS = Path("runs/operational/blind_reference_probe_rank16/splits.json")
META_KEYS = ("labels", "families", "event", "scenario", "source", "timestep", "node", "early")
DELTA = 3
CANDIDATES = ("frozen", "maxpool", "delayed", "delayed_blend")
TARGET_F1 = .80


def atomic_npz(path, **arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def logit(value):
    value = np.clip(value, 1e-6, 1 - 1e-6)
    return np.log(value / (1 - value))


def logit_blend(first, second, first_weight):
    linear = first_weight * logit(first) + (1 - first_weight) * logit(second)
    return 1 / (1 + np.exp(-np.clip(linear, -30, 30)))


def forward_max(score, arrays, delta):
    """max(score_t .. score_{t+delta}) inside each sensor series."""
    from wdn.delayed_decision_features import forward_index
    index = forward_index(arrays, delta)
    pooled = score.copy()
    for offset in range(delta):
        present = index[:, offset] >= 0
        pooled[present] = np.maximum(pooled[present], score[index[present, offset]])
    return pooled


def oracle(score, arrays, family_id):
    selected = arrays["families"] == family_id
    return _best_f1(score[selected], arrays["labels"][selected])


def run(output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "campaign.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Delayed-decision screen is already running") from error

    signature = {"plan_sha256": sha(PLAN), "declared_delta_hours": DELTA,
        "candidates": list(CANDIDATES), "target_family_f1": TARGET_F1,
        "source_sha256": {path: sha(Path(path)) for path in (
            "src/wdn/delayed_decision_features.py", "src/wdn/models/delayed_decision.py",
            "thesis_v2/experiments/screen_delayed_decision_head.py")},
        "base_signature_sha256": sha(BASE / "signature.json"),
        "seasonal_oof_sha256": sha(SEASONAL / "oof_predictions.npz"),
        "splits_sha256": sha(SPLITS), "calibration_evaluated": False,
        "validation_evaluated": False, "test_evaluated": False,
        "locked_eval_evaluated": False}
    if (output / "signature.json").exists() and read_json(output / "signature.json") != signature:
        raise ValueError("Delayed-decision screen source/input signature changed")
    write_json(output / "signature.json", signature)
    if (output / "summary.json").exists():
        print("Delayed-decision screen already complete; no refit", flush=True)
        return
    started = time.monotonic()

    def status(phase, **details):
        write_json(output / "status.json", {"phase": phase,
            "elapsed_seconds": time.monotonic() - started,
            "calibration_evaluated": False, "validation_evaluated": False,
            "test_evaluated": False, **details})
        print(phase, details, flush=True)

    splits = read_json(SPLITS)
    data = ExpandedTrainData(ORIGINAL, EXPANSIONS, splits)
    base_names = read_json(BASE / "feature_names.json")
    seasonal_names = read_json(SEASONAL / "seasonal_feature_names.json")

    held_parts, fold_scores = [], []
    for outer in range(len(data.source_folds)):
        folder = output / f"fold_{outer}"
        folder.mkdir(exist_ok=True)
        prediction_path = folder / "predictions.npz"
        held_meta = load_arrays(BASE / f"fold_{outer}/features_held_out.npz")
        held_parts.append({key: held_meta[key] for key in META_KEYS})
        if prediction_path.exists():
            fold_scores.append(load_arrays(prediction_path))
            del held_meta
            gc.collect()
            continue

        status("assembling banks", fold=outer, delta=DELTA)
        train = load_arrays(BASE / f"fold_{outer}/features_train.npz")
        train["X"] = np.column_stack((train["X"],
            load_arrays(SEASONAL / f"fold_{outer}/seasonal_train.npz")["X"])).astype(np.float32)
        held = dict(held_meta)
        held["X"] = np.column_stack((held["X"],
            load_arrays(SEASONAL / f"fold_{outer}/seasonal_held_out.npz")["X"])).astype(np.float32)
        bank = list(base_names) + list(seasonal_names)

        status("building bounded forward-window evidence", fold=outer)
        train_forward, forward_names = delayed_decision_features(train, bank, DELTA)
        held_forward, held_forward_names = delayed_decision_features(held, bank, DELTA)
        if forward_names != held_forward_names:
            raise ValueError("Forward feature schemas differ between train and held out")
        train["X"] = np.column_stack((train["X"], train_forward)).astype(np.float32)
        held["X"] = np.column_stack((held["X"], held_forward)).astype(np.float32)
        del train_forward, held_forward
        gc.collect()
        names = bank + forward_names
        write_json(output / "feature_names.json", names)

        status("fitting delayed-decision experts", fold=outer, features=len(names))
        model = DelayedDecisionExperts(names, DELTA, seed=5100 + outer).fit(train)
        predictions = model.predict(held["X"])
        joblib.dump(model, folder / "bundle.joblib")
        write_json(folder / "model_metadata.json", model.metadata())
        atomic_npz(prediction_path, **predictions)
        fold_scores.append(load_arrays(prediction_path))
        del train, held, held_meta
        gc.collect()

    oof = {key: np.concatenate([part[key] for part in held_parts]) for key in META_KEYS}
    frozen_run = load_arrays(SEASONAL / "oof_predictions.npz")
    for key in ("labels", "families", "scenario", "timestep", "node", "source"):
        if not np.array_equal(frozen_run[key], oof[key]):
            raise ValueError(f"Row alignment with the frozen seasonal OOF broke on {key}")

    delayed = np.column_stack([np.concatenate([part[name] for part in fold_scores])
                               for name in ("drift", "noise")])
    frozen = frozen_run["scores"]
    pooled = np.column_stack([forward_max(frozen[:, column], oof, DELTA) for column in (0, 1)])
    blend = np.column_stack([logit_blend(delayed[:, c], pooled[:, c], .5) for c in (0, 1)])
    scored = {"frozen": frozen, "maxpool": pooled, "delayed": delayed, "delayed_blend": blend}

    control = read_json(SEASONAL / "summary.json")
    results, gates = {}, {}
    for name in CANDIDATES:
        scores = scored[name]
        diagnostics = expert_diagnostics(oof, scores, data.events)
        family_f1 = {"drift": oracle(scores[:, 0], oof, 3),
                     "noise": oracle(scores[:, 1], oof, 4)}
        entry = {}
        for family in ("drift", "noise"):
            row = diagnostics[family]
            reference = control["diagnostics"][family]
            checks = {
                "oracle_f1_at_least_0_80": family_f1[family]["f1"] >= TARGET_F1,
                "clean_fpr_within_allowance": row["curve_clean_fpr"] <= reference["curve_clean_fpr"] + .00025,
                "post_event_fp_within_budget": row["curve_post_event_fp"] <= reference["curve_post_event_fp"],
                "worst_ap_preserved": row["worst_ap"] >= reference["worst_ap"] - .03}
            entry[family] = {"passed": bool(all(checks.values())), "checks": checks,
                "oracle_f1": family_f1[family]["f1"],
                "oracle_f1_gain_over_frozen": family_f1[family]["f1"]
                    - control["family_oracle_f1"][family]["f1"],
                "macro_ap": row["macro_ap"], "worst_ap": row["worst_ap"],
                "early_macro_recall": row["early_macro_recall"],
                "clean_fpr": row["curve_clean_fpr"],
                "post_event_fp": row["curve_post_event_fp"]}
        results[name] = {"diagnostics": diagnostics, "family_oracle_f1": family_f1}
        gates[name] = entry

    winner = {}
    for family in ("drift", "noise"):
        passing = [name for name in CANDIDATES if gates[name][family]["passed"]]
        best = max(CANDIDATES, key=lambda name: (gates[name][family]["oracle_f1"],
                                                 -CANDIDATES.index(name)))
        winner[family] = {"passed_candidates": passing,
            "selected": passing[0] if passing else None,
            "best_f1_candidate": best,
            "best_f1": gates[best][family]["oracle_f1"]}

    atomic_npz(output / "oof_predictions.npz", delayed=delayed, maxpool=pooled,
               delayed_blend=blend, frozen=frozen, **oof)
    both = all(winner[family]["selected"] for family in ("drift", "noise"))
    partial = all(gates[winner[family]["best_f1_candidate"]][family]["oracle_f1"] >= .78
                  for family in ("drift", "noise"))
    summary = {"status": "completed", "stage": "1",
        "scope": "expanded TRAIN-only four-generator-held OOF; declared decision latency",
        "declared_delta_hours": DELTA,
        "control_family_oracle_f1": {f: control["family_oracle_f1"][f]["f1"] for f in ("drift", "noise")},
        "candidates": results, "gates": gates, "winner": winner,
        "stage1_passed": bool(both),
        "next_action": ("stage_4_joint_calibration" if both else
                        "stage_2_drift_onset_regression" if partial else
                        "stop_and_report_ceiling"),
        "selection_limit": "candidates were predeclared; oracle F1 is a hindsight "
                           "optimum over the score ordering and is not a deployable threshold",
        "calibration_evaluated": False, "validation_evaluated": False,
        "test_evaluated": False, "locked_eval_evaluated": False,
        "oof_sha256": sha(output / "oof_predictions.npz"),
        "elapsed_seconds": time.monotonic() - started}
    write_json(output / "summary.json", summary)
    print(json.dumps({"winner": winner, "next_action": summary["next_action"],
                      "f1": {name: {f: round(gates[name][f]["oracle_f1"], 4)
                                    for f in ("drift", "noise")} for name in CANDIDATES}},
                     indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/operational/delayed_decision_head_v1")
    run(parser.parse_args().output_dir)
