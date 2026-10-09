"""Four-trial frozen-data campaign with scenario-OOF fusion and late validation."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import pickle
import time

import joblib
import numpy as np
import optuna
from sklearn.metrics import average_precision_score

from wdn.audit_operational_experts import best_f1_threshold
from wdn.dynamic_residual_features import dynamic_residual_features
from wdn.models.blind_reference import BlindPressureReference
from wdn.models.residual_hybrid import EvidenceFusion, MECHANISMS, ResidualHybrid
from wdn.models.robust_reference import RobustBlindReference
from wdn.operational_calibration import calibrate_threshold, family_summary
from wdn.probe_blind_reference import residual_features
from wdn.sequential_evidence import sequential_evidence
from wdn.train_operational_moe import FAMILY_NAMES, _binary_counts, summarise


SOURCE_FILES = ("run_expert_redesign.py", "models/robust_reference.py", "sequential_evidence.py",
                "models/residual_hybrid.py", "models/blind_reference.py", "probe_blind_reference.py",
                "dynamic_residual_features.py", "operational_calibration.py")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_folds(scenarios, events, seed=821):
    rng = np.random.default_rng(seed)
    for _ in range(10000):
        groups = [sorted(g.tolist()) for g in np.array_split(rng.permutation(scenarios), 3)]
        if all({e["family"] for e in events if e["scenario_id"] in group} == set(FAMILY_NAMES[1:]) for group in groups):
            return groups
    raise ValueError("Cannot form three whole-scenario folds with all families")


class CampaignData:
    def __init__(self, data_dir):
        self.directory = Path(data_dir)
        with (self.directory/"snapshots.pkl").open("rb") as stream:
            self.snapshots = pickle.load(stream)
        with (self.directory/"corrupted.pkl").open("rb") as stream:
            self.corrupted = pickle.load(stream)
        self.events = read_json(self.directory/"events.json")

    def scenario(self, sid):
        indices = sorted((i for i, s in enumerate(self.snapshots) if s.scenario_id == sid),
                         key=lambda i: self.snapshots[i].timestep)
        # Never extract clean hydraulic values, flow truth or other scenarios.
        return {"values": np.stack([self.corrupted[i].pressure_obs.numpy() for i in indices]),
                "mask": np.stack([self.corrupted[i].pressure_mask.numpy() > 0 for i in indices]),
                "labels": np.stack([self.corrupted[i].pressure_anomaly.numpy() for i in indices]),
                "families": np.array([self.corrupted[i].attack_type_id for i in indices]),
                "timestep": np.array([self.snapshots[i].timestep for i in indices])}

    def reference(self, scenarios, frozen=None):
        values, masks = [], []
        for sid in scenarios:
            a = self.scenario(sid)
            normal = a["families"] == 0
            values.append(a["values"][normal]); masks.append(a["mask"][normal])
        values, masks = np.concatenate(values), np.concatenate(masks)
        base = joblib.load(frozen) if frozen else BlindPressureReference(rank=16).fit(values, masks)
        return RobustBlindReference(base).calibrate_scale(values, masks)

    def features(self, scenarios, reference):
        collected = {key: [] for key in ("X", "labels", "families", "event", "scenario", "timestep", "node")}
        errors = []
        for sid in sorted(scenarios):
            a = self.scenario(sid)
            values, mask = a["values"], a["mask"]
            prediction, support, disagreement = reference.predict_details(values, mask)
            scale = reference.noise_scale_
            base, base_names = residual_features(values, mask, prediction, support, scale)
            dynamic, dynamic_names = dynamic_residual_features(values, mask, prediction, scale)
            sequential, sequential_names = sequential_evidence(values, mask, prediction, scale)
            features = np.concatenate([base, dynamic, (disagreement[15:]/scale)[..., None], sequential], axis=-1)
            names = base_names+dynamic_names+["reference_disagreement"]+sequential_names
            endpoint_mask = mask[15:]
            labels = a["labels"][15:]
            times = a["timestep"][15:]
            family = np.broadcast_to(a["families"][15:, None], endpoint_mask.shape)
            events = np.full(labels.shape, -1, dtype=int)
            for event_id, e in enumerate(self.events):
                if e["scenario_id"] != sid:
                    continue
                active = (times >= e["start_timestep"]) & (times < e["start_timestep"]+e["actual_steps"])
                events[active] = event_id
            if np.any(events[labels > 0] < 0):
                raise ValueError("Positive label without training/evaluation event")
            collected["X"].append(features[endpoint_mask])
            collected["labels"].append(labels[endpoint_mask])
            collected["families"].append(family[endpoint_mask])
            collected["event"].append(events[endpoint_mask])
            collected["scenario"].append(np.full(int(endpoint_mask.sum()), sid))
            collected["timestep"].append(np.broadcast_to(times[:, None], endpoint_mask.shape)[endpoint_mask])
            collected["node"].append(np.broadcast_to(np.arange(labels.shape[1]), labels.shape)[endpoint_mask])
            normal = endpoint_mask & (family == 0)
            errors.extend(np.abs(values[15:]-prediction[15:])[normal].tolist())
        result = {key: np.concatenate(value) for key, value in collected.items()}
        if not np.isfinite(result["X"]).all():
            raise ValueError("Nonfinite features")
        result["normal_reference_mae_m"] = np.array(np.mean(errors))
        return result, names


def load_arrays(path):
    return dict(np.load(path))


def concat_arrays(parts):
    return {key: np.concatenate([p[key] for p in parts]) for key in ("X", "labels", "families", "event", "scenario", "timestep", "node")}


def operating_point(scores, arrays):
    return calibrate_threshold(scores, arrays["labels"], arrays["families"],
                               objective="worst_f1", max_fpr=.005, min_replay_f1=.5)


def score_report(scores, arrays, point):
    result = summarise(scores, arrays["labels"], arrays["families"], point["threshold"])
    return {"selection": point, "validation": result, **family_summary(result)}


def independent_audit(cal_scores, val_scores, cal, val):
    rows = {}
    for i, name in enumerate(MECHANISMS):
        threshold = best_f1_threshold(cal_scores[:, i], cal["labels"])
        row = {"threshold": threshold, "per_family": {}}
        for fid, family in enumerate(FAMILY_NAMES):
            selected = val["families"] == fid
            y, score = val["labels"][selected], val_scores[selected, i]
            row["per_family"][family] = {**_binary_counts(score, y, threshold),
                "auprc": float(average_precision_score(y, score)) if y.any() else None}
        rows[name] = row
    return {"test_evaluated": False, "threshold_protocol": "independent global calibration F1 per expert", "expert_matrix": rows}


def run(output_dir, data_dir="data/thesis_v2/operational_modena_seed811"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/"campaign.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("This campaign is already running; do not duplicate it")
    source = Path(__file__).parent
    baseline = Path("runs/operational/blind_reference_probe_rank16")
    splits = read_json(baseline/"splits.json")
    signature = {"source": {p: sha(source/p) for p in SOURCE_FILES},
                 "data_config": read_json(Path(data_dir)/"manifest.json")["config_sha256"],
                 "reference": sha(baseline/"reference.joblib"), "splits": splits, "trial_budget": 4,
                 "data_files": {name: sha(Path(data_dir)/name) for name in ("snapshots.pkl", "corrupted.pkl", "events.json")}}
    if signature["data_config"] != "b407f63deba884db27e763cad1c16a1e42d422e546b942741320cb36012b8dea":
        raise ValueError("Unexpected dataset configuration")
    if (output/"signature.json").exists() and read_json(output/"signature.json") != signature:
        raise ValueError("Campaign source/input signature changed; do not silently resume")
    write_json(output/"signature.json", signature)
    if (output/"summary.json").exists():
        print("Campaign already complete", output/"summary.json", flush=True)
        return
    started = time.monotonic()
    def status(phase, **extra):
        write_json(output/"status.json", {"phase": phase, "elapsed_seconds": time.monotonic()-started,
                                         "test_evaluated": False, **extra})
        print(phase, extra, flush=True)
    data = CampaignData(data_dir)
    folds = group_folds(splits["train"], data.events)
    write_json(output/"inner_folds.json", folds)
    full = output/"full"
    full.mkdir(exist_ok=True)
    if not (full/"reference.joblib").exists():
        status("building full TRAIN normal reference")
        reference = data.reference(splits["train"], frozen=baseline/"reference.joblib")
        joblib.dump(reference, full/"reference.joblib")
    reference = joblib.load(full/"reference.joblib")
    for split in ("train", "calibration"):
        if not (full/f"features_{split}.npz").exists():
            status("building full reference features", split=split)
            a, names = data.features(splits[split], reference)
            cached = load_arrays(baseline/f"features_{split}.npz")
            for key in ("labels", "families"):
                np.testing.assert_array_equal(a[key], cached[key])
            np.savez_compressed(full/f"features_{split}.npz", **a)
            write_json(output/"feature_names.json", names)
    names = read_json(output/"feature_names.json")
    for i, held_out in enumerate(folds):
        folder = output/f"fold_{i}"
        folder.mkdir(exist_ok=True)
        train_sids = sorted(set(splits["train"])-set(held_out))
        if not (folder/"reference.joblib").exists():
            status("fitting inner normal reference", fold=i, train_scenarios=train_sids, held_out_scenarios=held_out)
            joblib.dump(data.reference(train_sids), folder/"reference.joblib")
        fold_reference = joblib.load(folder/"reference.joblib")
        for split, sids in (("train", train_sids), ("held_out", held_out)):
            path = folder/f"features_{split}.npz"
            if not path.exists():
                status("building inner features", fold=i, split=split)
                a, found_names = data.features(sids, fold_reference)
                assert found_names == names
                np.savez_compressed(path, **a)
    train = load_arrays(full/"features_train.npz")
    cal = load_arrays(full/"features_calibration.npz")
    status("features complete; calibration only search", features=len(names),
           normal_train_mae=float(train["normal_reference_mae_m"]), normal_calibration_mae=float(cal["normal_reference_mae_m"]))
    study = optuna.create_study(study_name="mechanism_redesign_v1", storage=f"sqlite:///{output/'study.sqlite3'}",
        load_if_exists=True, direction="maximize", sampler=optuna.samplers.TPESampler(seed=821, n_startup_trials=2))
    if not study.trials:
        study.enqueue_trial({"C": .1})
        study.enqueue_trial({"C": .1})
    def objective(trial):
        kind = "trees" if trial.number == 0 else "mechanism"
        C = trial.suggest_float("C", .001, 1., log=True)
        trial.set_user_attr("kind", kind)
        folder = output/f"trial_{trial.number:04d}"
        folder.mkdir(exist_ok=False)
        oof_predictions, held_arrays = [], []
        for i in range(len(folds)):
            status("training OOF experts", trial=trial.number, kind=kind, fold=i)
            fit_arrays = load_arrays(output/f"fold_{i}/features_train.npz")
            held = load_arrays(output/f"fold_{i}/features_held_out.npz")
            assert set(fit_arrays["scenario"]).isdisjoint(set(held["scenario"]))
            model = ResidualHybrid(names, kind, C).fit(fit_arrays)
            oof_predictions.append(model.predict(held["X"]))
            held_arrays.append(held)
        oof = concat_arrays(held_arrays)
        oof_scores = np.concatenate(oof_predictions)
        fusion = EvidenceFusion(names, C=C).fit(oof_scores, oof)
        np.savez_compressed(folder/"oof_predictions.npz", experts=oof_scores,
                            labels=oof["labels"], families=oof["families"], scenario=oof["scenario"])
        status("training full experts", trial=trial.number, kind=kind)
        experts = ResidualHybrid(names, kind, C).fit(train)
        cal_experts = experts.predict(cal["X"])
        cal_score = fusion.predict(cal_experts, cal["X"])
        joblib.dump({"experts": experts, "fusion": fusion}, folder/"model.joblib")
        np.savez_compressed(folder/"predictions_calibration.npz", experts=cal_experts, mixture=cal_score)
        try:
            point = operating_point(cal_score, cal)
        except ValueError as error:
            if str(error) != "No feasible calibration operating point":
                raise
            trial.set_user_attr("infeasible", str(error))
            write_json(folder/"infeasible.json", {"reason": str(error), "validation_evaluated": False})
            raise optuna.TrialPruned(str(error))
        write_json(folder/"calibration.json", point)
        trial.set_user_attr("selection", point)
        status("trial complete", trial=trial.number, calibration=point["calibration"])
        return point["calibration"]["worst_family_f1"]
    consumed = sum(t.state != optuna.trial.TrialState.WAITING for t in study.trials)
    study.optimize(objective, n_trials=max(0, 4-consumed))
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("No completed calibration candidate")
    winner = max(completed, key=lambda t: tuple(t.user_attrs["selection"]["calibration"][k]
                 for k in ("worst_family_f1", "macro_f1", "overall_f1")))
    selection = {"trial": winner.number, "kind": winner.user_attrs["kind"], "params": winner.params,
                 "selected_before_validation_features": True, "selection": winner.user_attrs["selection"],
                 "completed_trials": len(completed)}
    write_json(output/"selection_frozen.json", selection)
    # This is the first extraction of validation examples in this campaign.
    status("selection frozen; building validation features", selected_trial=winner.number)
    val, found_names = data.features(splits["validation"], reference)
    assert found_names == names
    old_val = load_arrays(baseline/"features_validation.npz")
    for key in ("labels", "families"):
        np.testing.assert_array_equal(val[key], old_val[key])
    np.savez_compressed(full/"features_validation.npz", **val)
    reports = {}
    for trial in completed:
        folder = output/f"trial_{trial.number:04d}"
        bundle = joblib.load(folder/"model.joblib")
        val_experts = bundle["experts"].predict(val["X"])
        val_score = bundle["fusion"].predict(val_experts, val["X"])
        cal_scores = load_arrays(folder/"predictions_calibration.npz")
        # Verify the saved estimator reproduces its original calibration outputs.
        fresh = bundle["experts"].predict(cal["X"])
        np.testing.assert_array_equal(fresh, cal_scores["experts"])
        np.testing.assert_array_equal(bundle["fusion"].predict(fresh, cal["X"]), cal_scores["mixture"])
        point = trial.user_attrs["selection"]
        assert point == operating_point(cal_scores["mixture"], cal)
        result = {"trial": trial.number, "kind": trial.user_attrs["kind"], "params": trial.params,
                  **score_report(val_score, val, point)}
        write_json(folder/"summary.json", result)
        np.savez_compressed(folder/"predictions_validation.npz", experts=val_experts, mixture=val_score)
        audit = independent_audit(cal_scores["experts"], val_experts, cal, val)
        write_json(folder/"expert_audit.json", audit)
        reports[str(trial.number)] = result
        status("validation audit complete", trial=trial.number,
               overall_f1=result["validation"]["overall"]["f1"], worst_f1=result["worst_family_f1"])
    result = {"status": "completed", "test_evaluated": False, "selection": selection,
        "trials": reports, "normal_reference_mae_m": {"train": float(train["normal_reference_mae_m"]),
            "calibration": float(cal["normal_reference_mae_m"]), "validation": float(val["normal_reference_mae_m"])},
        "source_and_inputs": signature, "inner_folds": folds,
        "scope": "pressure-only heterogeneous experts; validation is reused development data",
        "seconds_this_invocation": time.monotonic()-started}
    write_json(output/"summary.json", result)
    status("completed", selected_trial=winner.number)
    print(json.dumps({"selection": selection, "trials": reports}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="runs/operational/mechanism_redesign_v1")
    parser.add_argument("--data_dir", default="data/thesis_v2/operational_modena_seed811")
    run(**vars(parser.parse_args()))
