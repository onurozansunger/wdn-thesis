"""Four local candidates, frozen data, scenario-OOF fitting, late validation."""
from __future__ import annotations

import argparse
import fcntl
from pathlib import Path
import time

import joblib
import numpy as np
import optuna

from wdn.dynamic_residual_features import dynamic_residual_features
from wdn.models.blind_reference import BlindPressureReference
from wdn.models.robust_reference import RobustBlindReference
from wdn.models.weak_family import MonotoneFusion, WeakExperts
from wdn.probe_blind_reference import residual_features
from wdn.run_expert_redesign import (CampaignData, independent_audit, load_arrays,
    operating_point, read_json, score_report, sha, write_json)
from wdn.sequential_evidence import sequential_evidence
from wdn.weak_family_features import conditional_features


BASE = Path("runs/operational/mechanism_redesign_v1")
STRONG = BASE / "trial_0000"
DATA = Path("data/thesis_v2/operational_modena_seed811")
SOURCE = ("src/wdn/train_weak_families.py", "src/wdn/weak_family_features.py",
          "src/wdn/models/weak_family.py", "thesis_v2/WEAK_FAMILY_TRAINING_PROTOCOL.md")


def concatenate(parts):
    return {key: np.concatenate([p[key] for p in parts]) for key in
            ("X", "labels", "families", "scenario", "timestep", "node", "event", "early")}


def add_early(a, events):
    early = np.zeros(len(a["labels"]), dtype=bool)
    for event in events:
        if event["scenario_id"] not in set(a["scenario"]):
            continue
        early |= ((a["scenario"] == event["scenario_id"]) &
                  (a["timestep"] >= event["start_timestep"]) &
                  (a["timestep"] < event["start_timestep"]+3))
    return {**a, "early": early}


class FeatureStore:
    def __init__(self, output, splits, folds, status):
        self.output, self.splits, self.folds, self.status = output, splits, folds, status
        self.data = CampaignData(DATA)
        self.names = read_json(BASE / "feature_names.json")
        self.allowed = set(splits["train"]+splits["calibration"]+splits["validation"])

    def observations(self, sid, joint):
        if sid not in self.allowed or sid in self.splits["test"]:
            raise ValueError("Test/unknown scenario extraction is forbidden")
        a = self.data.scenario(sid)
        if not joint:
            return a
        indices = sorted((i for i, s in enumerate(self.data.snapshots) if s.scenario_id == sid),
                         key=lambda i: self.data.snapshots[i].timestep)
        flow = np.stack([self.data.corrupted[i].flow_obs.numpy() for i in indices])
        mask = np.stack([self.data.corrupted[i].flow_mask.numpy() > 0 for i in indices])
        a["pressure_count"] = a["values"].shape[1]
        a["values"] = np.column_stack([a["values"], flow])
        a["mask"] = np.column_stack([a["mask"], mask])
        return a

    def scenario_ids(self, folder, split):
        if folder == "full":
            return self.splits[split]
        held = self.folds[int(folder.split("_")[1])]
        return held if split == "held_out" else sorted(set(self.splits["train"])-set(held))

    def get_reference(self, mode, folder):
        if mode == "context":
            return joblib.load(BASE / folder / "reference.joblib")
        path = self.output / mode / folder / "reference.joblib"
        if not path.exists():
            sids = self.scenario_ids(folder, "train")
            self.status("fitting joint normal TRAIN reference", folder=folder, scenarios=sids)
            values, masks = [], []
            for sid in sids:
                a = self.observations(sid, joint=True)
                normal = a["families"] == 0
                values.append(a["values"][normal]); masks.append(a["mask"][normal])
            values, masks = np.concatenate(values), np.concatenate(masks)
            base = BlindPressureReference(rank=16).fit(values, masks)
            reference = RobustBlindReference(base).calibrate_scale(values, masks)
            joblib.dump(reference, path)
        return joblib.load(path)

    def arrays(self, mode, folder, split):
        if split == "validation" and not (self.output / "selection_frozen.json").exists():
            raise ValueError("Validation extraction before selection is forbidden")
        base = load_arrays(BASE / folder / f"features_{split}.npz")
        assert set(base["scenario"]) == set(self.scenario_ids(folder, split))
        assert set(base["scenario"]).isdisjoint(self.splits["test"])
        if mode == "local":
            return add_early(base, self.data.events), self.names
        directory = self.output / mode / folder
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"features_{split}.npz"
        if not path.exists():
            reference = self.get_reference(mode, folder)
            pieces, extra_names = [], None
            self.status("extracting causal context features", who=mode, folder=folder, split=split)
            for sid in sorted(self.scenario_ids(folder, split)):
                a = self.observations(sid, joint=mode == "joint")
                count = a["pressure_count"] if mode == "joint" else a["values"].shape[1]
                prediction, support, spread, context, context_names = conditional_features(reference, a["values"], a["mask"], count)
                endpoint = a["mask"][15:, :count]
                valid_base = base["scenario"] == sid
                np.testing.assert_array_equal(a["labels"][15:][endpoint], base["labels"][valid_base])
                if mode == "context":
                    extra, found_names = context[15:][endpoint], context_names
                else:
                    p, mask = a["values"][:, :count], a["mask"][:, :count]
                    scale = reference.noise_scale_[:count]
                    basic, basic_names = residual_features(p, mask, prediction, support, scale)
                    dynamic, dynamic_names = dynamic_residual_features(p, mask, prediction, scale)
                    sequential, seq_names = sequential_evidence(p, mask, prediction, scale)
                    local = np.concatenate([basic, dynamic, (spread[15:]/scale)[..., None], sequential], axis=-1)
                    local_names = basic_names+dynamic_names+["reference_disagreement"]+seq_names
                    assert local_names == self.names
                    extra = np.concatenate([local, context[15:]], axis=-1)[endpoint]
                    found_names = ["joint_"+name for name in local_names]+context_names
                assert len(extra) == int(valid_base.sum())
                pieces.append(extra)
                if extra_names is not None:
                    assert extra_names == found_names
                extra_names = found_names
            extra = np.concatenate(pieces)
            assert np.isfinite(extra).all()
            np.savez_compressed(path, extra=extra)
            write_json(directory / "feature_names.json", self.names+extra_names)
        extra = np.load(path)["extra"]
        assert len(extra) == len(base["labels"])
        names = read_json(directory / "feature_names.json")
        return add_early({**base, "X": np.column_stack([base["X"], extra])}, self.data.events), names


def predict(bundle, a, base_scores=None):
    strong = bundle["strong"].predict(a["X"][:, :bundle["base_width"]])[:, :3] if base_scores is None else base_scores[:, :3]
    scores = np.column_stack([strong, bundle["weak"].predict(a["X"])])
    return {"experts": scores, "mixture": bundle["fusion"].predict(scores, a)}


def run(output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "campaign.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Campaign already running; refusing duplicate")
    old = read_json(BASE / "summary.json")
    old_signature = old["source_and_inputs"]
    for name, expected in old_signature["source"].items():
        assert sha(Path("src/wdn") / name) == expected
    for name, expected in old_signature["data_files"].items():
        assert sha(DATA / name) == expected
    signature = {"sources": {p: sha(Path(p)) for p in SOURCE},
                 "base_signature": old_signature, "strong_model_sha": sha(STRONG / "model.joblib"),
                 "strong_oof_sha": sha(STRONG / "oof_predictions.npz"),
                 "base_features": {str(p): sha(p) for p in BASE.glob("*/features_*.npz")
                                   if "validation" not in p.name}, "budget": 4}
    previous = output / "signature.json"
    if previous.exists() and read_json(previous) != signature:
        raise ValueError("Campaign inputs/source changed; do not resume silently")
    write_json(previous, signature)
    if (output / "summary.json").exists():
        print("Already completed", output, flush=True)
        return
    started = time.monotonic()
    def status(phase, **details):
        record = {"phase": phase, "elapsed_seconds": time.monotonic()-started, "test_evaluated": False, **details}
        write_json(output / "status.json", record)
        print(phase, details, flush=True)
    splits, folds = old_signature["splits"], old["inner_folds"]
    write_json(output / "splits.json", splits)
    write_json(output / "inner_folds.json", folds)
    store = FeatureStore(output, splits, folds, status)
    strong_model = joblib.load(STRONG / "model.joblib")["experts"]
    strong_oof = load_arrays(STRONG / "oof_predictions.npz")
    strong_cal = load_arrays(STRONG / "predictions_calibration.npz")["experts"]
    study = optuna.create_study(study_name="weak_family_v1", storage=f"sqlite:///{output/'study.sqlite3'}",
        load_if_exists=True, direction="maximize", sampler=optuna.samplers.TPESampler(seed=831, n_startup_trials=2))
    if not study.trials:
        for _ in range(3):
            study.enqueue_trial({"leaves": 15, "regularisation": 10., "early_weight": 1., "C": .1}, skip_if_exists=False)
    def objective(trial):
        if trial.number < 3:
            mode = ("local", "context", "joint")[trial.number]
        else:
            eligible = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if not eligible:
                raise ValueError("No feasible representation to refine")
            winner = max(eligible, key=lambda t: tuple(t.user_attrs["selection"]["calibration"][key]
                         for key in ("worst_family_f1", "macro_f1", "overall_f1")))
            mode = winner.user_attrs["mode"]
        trial.set_user_attr("mode", mode)
        leaves = trial.suggest_categorical("leaves", [7, 15, 31])
        regularisation = trial.suggest_float("regularisation", 3., 50., log=True)
        early_weight = trial.suggest_float("early_weight", 1., 4.)
        C = trial.suggest_float("C", .03, .5, log=True)
        folder = output / f"trial_{trial.number:04d}"
        folder.mkdir(exist_ok=False)
        scores, held_parts, cursor = [], [], 0
        for index in range(3):
            training, names = store.arrays(mode, f"fold_{index}", "train")
            held, held_names = store.arrays(mode, f"fold_{index}", "held_out")
            assert names == held_names and set(training["scenario"]).isdisjoint(held["scenario"])
            status("training weak experts", trial=trial.number, mode=mode, fold=index, features=len(names))
            weak = WeakExperts(names, leaves, regularisation, early_weight).fit(training)
            end = cursor+len(held["labels"])
            np.testing.assert_array_equal(held["labels"], strong_oof["labels"][cursor:end])
            np.testing.assert_array_equal(held["scenario"], strong_oof["scenario"][cursor:end])
            scores.append(np.column_stack([strong_oof["experts"][cursor:end, :3], weak.predict(held["X"])]))
            cursor = end
            held_parts.append(held)
            del training
        assert cursor == len(strong_oof["labels"])
        oof, scores = concatenate(held_parts), np.concatenate(scores)
        status("fitting monotone OOF fusion", trial=trial.number)
        fusion = MonotoneFusion(names, C=C).fit(scores, oof)
        np.savez_compressed(folder / "oof_predictions.npz", experts=scores, labels=oof["labels"],
                            families=oof["families"], scenario=oof["scenario"])
        del held_parts, oof, scores
        train, names = store.arrays(mode, "full", "train")
        cal, cal_names = store.arrays(mode, "full", "calibration")
        assert names == cal_names
        status("training full weak experts", trial=trial.number, mode=mode)
        weak = WeakExperts(names, leaves, regularisation, early_weight).fit(train)
        bundle = {"strong": strong_model, "weak": weak, "fusion": fusion, "base_width": len(store.names), "mode": mode}
        predictions = predict(bundle, cal, strong_cal)
        joblib.dump(bundle, folder / "model.joblib")
        np.savez_compressed(folder / "predictions_calibration.npz", **predictions)
        try:
            point = operating_point(predictions["mixture"], cal)
        except ValueError as error:
            if str(error) != "No feasible calibration operating point":
                raise
            write_json(folder / "infeasible.json", {"reason": str(error), "validation_evaluated": False})
            raise optuna.TrialPruned(str(error))
        write_json(folder / "calibration.json", point)
        trial.set_user_attr("selection", point)
        status("trial calibrated", trial=trial.number, mode=mode, calibration=point["calibration"])
        return point["calibration"]["worst_family_f1"]
    used = sum(t.state != optuna.trial.TrialState.WAITING for t in study.trials)
    study.optimize(objective, n_trials=max(0, 4-used))
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("No feasible candidate")
    winner = max(completed, key=lambda t: tuple(t.user_attrs["selection"]["calibration"][key]
                 for key in ("worst_family_f1", "macro_f1", "overall_f1")))
    selection = {"trial": winner.number, "mode": winner.user_attrs["mode"], "params": winner.params,
                 "point": winner.user_attrs["selection"], "selected_before_new_validation": True}
    write_json(output / "selection_frozen.json", selection)
    reports = {}
    status("selection frozen; starting validation", selected=winner.number, mode=selection["mode"])
    for trial in completed:
        mode = trial.user_attrs["mode"]
        folder = output / f"trial_{trial.number:04d}"
        bundle = joblib.load(folder / "model.joblib")
        cal, _ = store.arrays(mode, "full", "calibration")
        val, _ = store.arrays(mode, "full", "validation")
        stored_cal = load_arrays(folder / "predictions_calibration.npz")
        fresh_cal = predict(bundle, cal)
        for name in fresh_cal:
            np.testing.assert_array_equal(stored_cal[name], fresh_cal[name])
        point = trial.user_attrs["selection"]
        assert operating_point(fresh_cal["mixture"], cal) == point
        predictions = predict(bundle, val)
        np.savez_compressed(folder / "predictions_validation.npz", **predictions)
        audit = independent_audit(fresh_cal["experts"], predictions["experts"], cal, val)
        write_json(folder / "expert_audit.json", audit)
        result = {"trial": trial.number, "mode": mode, "params": trial.params,
                  **score_report(predictions["mixture"], val, point)}
        write_json(folder / "summary.json", result)
        reports[str(trial.number)] = result
        status("validation candidate complete", trial=trial.number, mode=mode,
               overall=result["validation"]["overall"]["f1"],
               drift=result["validation"]["per_family"]["stealthy"]["f1"],
               noise=result["validation"]["per_family"]["noise"]["f1"])
    for path, expected in signature["sources"].items():
        assert sha(Path(path)) == expected
    for name, expected in old_signature["data_files"].items():
        assert sha(DATA / name) == expected
    selected = reports[str(winner.number)]["validation"]
    success = all(selected["per_family"][family]["f1"] >= .8 for family in ("stealthy", "noise"))
    summary = {"status": "completed", "test_evaluated": False, "selection": selection,
               "trials": reports, "target_0_80_both_achieved": success, "source_and_inputs": signature,
               "verification": {"calibration_reload_exact": True, "calibration_selection_reproduced": True,
                                "source_data_hashes_unchanged": True, "test_extracted": False},
               "scope": "pressure detection using pressure/flow; heterogeneous stack, reused development validation",
               "seconds": time.monotonic()-started}
    write_json(output / "summary.json", summary)
    status("completed", selected=winner.number, both_at_least_0_80=success)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="runs/operational/weak_family_v1")
    run(**vars(parser.parse_args()))
