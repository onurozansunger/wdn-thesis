"""Two prespecified recovery mechanisms, TRAIN-only scenario screening.

No calibration/validation/test extraction and no Optuna happen in this stage.
Passing this screen authorises the next planned stage, not a success claim.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
from pathlib import Path
import pickle
import time

import joblib
import numpy as np
from sklearn.metrics import average_precision_score
import wntr
import yaml

from wdn.dynamic_residual_features import dynamic_residual_features
from wdn.marked_change import ChangeConfig, MarkedChangeFilter
from wdn.models.physics_reference import HazenWilliamsReference, PhysicsConfig
from wdn.models.weak_family import WeakExperts
from wdn.probe_blind_reference import residual_features
from wdn.run_expert_redesign import CampaignData, load_arrays, read_json, sha, write_json
from wdn.sequential_evidence import sequential_evidence
from wdn.train_weak_families import add_early


BASE = Path("runs/operational/mechanism_redesign_v1")
CONTROL = Path("runs/operational/weak_family_v1")
DATA = Path("data/thesis_v2/operational_modena_seed811")
SOURCE = ("src/wdn/screen_recovery.py", "src/wdn/marked_change.py",
          "src/wdn/models/physics_reference.py", "thesis_v2/DRIFT_NOISE_RECOVERY_PLAN.md")
PROTOCOL = {"version": 1, "mechanisms": ["physics", "state"],
    "physics": asdict(PhysicsConfig()), "state": asdict(ChangeConfig()),
    "expert": {"leaves": 15, "regularisation": 10., "early_weight": 1., "seed": 831},
    "gates": {"macro_ap_gain": .05, "improved_scenarios": 3, "max_worst_ap_drop": .05,
              "diagnostic_fpr": .001, "normal_supported_mae_ratio": .90,
              "allow_all_normal_p95_increase": False,
              "allow_clean_fpr_increase": False, "allow_post_event_fp_increase": False},
    "scope": "TRAIN-only component screen; whole-scenario held-out experts",
    "max_new_configurations_if_pass": 4, "calibration_evaluated": False,
    "validation_evaluated": False, "test_evaluated": False,
    "curve_thresholds": "Per held-TRAIN scenario normal quantile; diagnostic, not deployment thresholds"}


class TrainOnlyData:
    def __init__(self, directory, splits):
        self.allowed = set(splits["train"])
        if self.allowed & set(splits["calibration"]+splits["validation"]+splits["test"]):
            raise ValueError("Overlapping outer scenario splits")
        self.data = CampaignData(directory)
        self.events = [e for e in self.data.events if e["scenario_id"] in self.allowed]
        self.cache = {}

    def scenario(self, sid):
        if sid not in self.allowed:
            raise ValueError("Screening may extract TRAIN scenarios only")
        if sid not in self.cache:
            a = self.data.scenario(sid)
            indices = sorted((i for i, s in enumerate(self.data.snapshots) if s.scenario_id == sid),
                             key=lambda i: self.data.snapshots[i].timestep)
            a["flow"] = np.stack([self.data.corrupted[i].flow_obs.numpy() for i in indices])
            a["flow_mask"] = np.stack([self.data.corrupted[i].flow_mask.numpy() > 0 for i in indices])
            self.cache[sid] = a
        return self.cache[sid]


def local_features(a, prediction, support, spread, scale):
    p, mask = a["values"], a["mask"]
    base, bn = residual_features(p, mask, prediction, support, scale)
    dynamic, dn = dynamic_residual_features(p, mask, prediction, scale)
    sequential, sn = sequential_evidence(p, mask, prediction, scale)
    return np.concatenate([base, dynamic, (spread[15:]/scale)[..., None], sequential], axis=-1), bn+dn+["reference_disagreement"]+sn


def post_event_mask(a, events, family):
    result = np.zeros(len(a["labels"]), dtype=bool)
    for event in events:
        if event["family"] != family:
            continue
        end = event["start_timestep"]+event["actual_steps"]
        result |= ((a["scenario"] == event["scenario_id"]) & (a["timestep"] >= end)
            & (a["timestep"] < end+16) & np.isin(a["node"], event["targets"]["pressure"])
            & (a["labels"] == 0))
    return result


def expert_diagnostics(a, scores, events):
    """Threshold-free scenario AP + explicitly nondeployable FPR curve audit."""
    if scores.shape != (len(a["labels"]), 2) or not np.isfinite(scores).all():
        raise ValueError("Two finite specialist score columns required")
    results = {}
    for col, fid, name in ((0, 3, "drift"), (1, 4, "noise")):
        family = a["families"] == fid
        by_scenario, early_recalls, thresholds = {}, {}, {}
        decision = np.zeros(len(scores), dtype=bool)
        for sid in np.unique(a["scenario"]):
            scenario = a["scenario"] == sid
            negatives = scores[scenario & (a["labels"] == 0), col]
            threshold = float(np.quantile(negatives, 1-PROTOCOL["gates"]["diagnostic_fpr"], method="higher"))
            thresholds[str(int(sid))] = threshold
            decision[scenario] = scores[scenario, col] > threshold
            loc = scenario & family
            if not np.any(a["labels"][loc] > 0):
                continue
            by_scenario[str(int(sid))] = float(average_precision_score(a["labels"][loc], scores[loc, col]))
            early = loc & a["early"] & (a["labels"] > 0)
            if not early.any():
                raise ValueError("Missing early observations in a weak-family scenario")
            early_recalls[str(int(sid))] = float(decision[early].mean())
        clean = a["families"] == 0
        normal = a["labels"] == 0
        post = post_event_mask(a, events, "stealthy" if fid == 3 else "noise")
        values = list(by_scenario.values())
        results[name] = {"by_scenario_ap": by_scenario, "macro_ap": float(np.mean(values)),
            "worst_ap": min(values), "pooled_ap": float(average_precision_score(a["labels"][family], scores[family, col])),
            "early_recall_by_scenario": early_recalls, "early_macro_recall": float(np.mean(list(early_recalls.values()))),
            "curve_thresholds_not_deployable": thresholds,
            "curve_normal_fp": int(decision[normal].sum()), "curve_normal_fpr": float(decision[normal].mean()),
            "curve_clean_fp": int(decision[clean].sum()), "curve_clean_fpr": float(decision[clean].mean()),
            "curve_post_event_fp": int(decision[post].sum()), "curve_post_event_points": int(post.sum())}
    return results


def gate(candidate, control, reference_pass=True):
    if set(candidate["by_scenario_ap"]) != set(control["by_scenario_ap"]):
        raise ValueError("Paired scenario support differs")
    gains = {k: candidate["by_scenario_ap"][k]-v for k, v in control["by_scenario_ap"].items()}
    g = PROTOCOL["gates"]
    checks = {"macro_ap_gain": candidate["macro_ap"] >= control["macro_ap"]+g["macro_ap_gain"],
        "at_least_three_scenarios_improve": sum(v > 0 for v in gains.values()) >= g["improved_scenarios"],
        "worst_scenario_preserved": candidate["worst_ap"] >= control["worst_ap"]-g["max_worst_ap_drop"],
        "early_recall_improves": candidate["early_macro_recall"] > control["early_macro_recall"],
        "clean_fpr_not_increased": candidate["curve_clean_fpr"] <= control["curve_clean_fpr"],
        "post_event_fp_not_increased": candidate["curve_post_event_fp"] <= control["curve_post_event_fp"],
        "reference_pass": bool(reference_pass)}
    return {"passed": all(checks.values()), "checks": checks, "scenario_ap_gains": gains,
            "macro_ap_gain": candidate["macro_ap"]-control["macro_ap"]}


def prepare_fold(output, index, held, splits, data, graph, status):
    folder = output/f"fold_{index}"
    folder.mkdir(exist_ok=True)
    reference = joblib.load(BASE/f"fold_{index}/reference.joblib")
    physics = HazenWilliamsReference(graph, reference)
    training = sorted(set(splits["train"])-set(held))
    for sid in sorted(splits["train"]):
        path = folder/f"scenario_{sid:02d}.npz"
        if path.exists():
            continue
        status("extracting blind physics and marked-change evidence", fold=index, scenario=sid)
        a = data.scenario(sid)
        pred, support, spread, diagnostics = physics.predict_details(a["values"], a["mask"], a["flow"], a["flow_mask"])
        state, state_names = MarkedChangeFilter().transform(a["values"], a["mask"],
            diagnostics["fallback_prediction"], reference.noise_scale_)
        np.savez_compressed(path, prediction=pred, support=support, spread=spread,
                            state=state, **diagnostics)
        write_json(folder/"marked_names.json", state_names)
    errors = []
    for sid in training:
        a = data.scenario(sid)
        cached = load_arrays(folder/f"scenario_{sid:02d}.npz")
        normal = a["families"] == 0
        errors.append(np.where(a["mask"][normal], np.abs(a["values"][normal]-cached["prediction"][normal]), np.nan))
    scale = np.nanmedian(np.concatenate(errors), axis=0)/.67448975
    if not np.isfinite(scale).all():
        raise ValueError("Missing normal TRAIN scale support")
    physics.noise_scale_ = np.maximum(scale, max(float(np.median(scale))*.1, 1e-6))
    joblib.dump(physics, folder/"physics_reference.joblib")
    write_json(folder/"fit_scope.json", {"training_scenarios": training, "held_scenarios": held,
        "reference_source": str(BASE/f"fold_{index}/reference.joblib"),
        "normal_scale_fitted_only_on": training, "state_config": asdict(ChangeConfig())})
    for split, sids in (("train", training), ("held_out", held)):
        base = add_early(load_arrays(BASE/f"fold_{index}/features_{split}.npz"), data.events)
        if set(base["scenario"]) != set(sids):
            raise ValueError("Cached feature scenario scope differs")
        pieces, state_parts, reference_rows = [], [], []
        for sid in sorted(sids):
            a = data.scenario(sid)
            c = load_arrays(folder/f"scenario_{sid:02d}.npz")
            endpoint = a["mask"][15:]
            loc = base["scenario"] == sid
            for key, expected in (("labels", a["labels"][15:][endpoint]),
                    ("timestep", np.broadcast_to(a["timestep"][15:, None], endpoint.shape)[endpoint]),
                    ("node", np.broadcast_to(np.arange(endpoint.shape[1]), endpoint.shape)[endpoint])):
                np.testing.assert_array_equal(base[key][loc], expected)
            features, names = local_features(a, c["prediction"], c["support"], c["spread"], physics.noise_scale_)
            assert names == read_json(BASE/"feature_names.json")
            confidence = np.stack([c["physical_weight"], c["anchor_count"]/10,
                                  c["physical_sigma"]/reference.noise_scale_,
                                  c["jackknife"]/reference.noise_scale_], axis=-1)
            pieces.append(np.concatenate([features, confidence[15:]], axis=-1)[endpoint])
            state_parts.append(c["state"][endpoint])
            reference_rows.append(np.column_stack([
                (a["values"][15:]-c["fallback_prediction"][15:])[endpoint],
                (a["values"][15:]-c["prediction"][15:])[endpoint],
                c["physical_weight"][15:][endpoint], c["anchor_count"][15:][endpoint]]))
        physical_names = names+["seq_physical_weight", "seq_physical_anchors", "seq_physical_sigma", "seq_physical_jackknife"]
        state_names = names+read_json(folder/"marked_names.json")
        physical = np.concatenate(pieces)
        state = np.column_stack([base["X"], np.concatenate(state_parts)])
        for mode, X, found_names in (("physics", physical, physical_names), ("state", state, state_names)):
            if X.shape[1] != len(found_names) or not np.isfinite(X).all():
                raise ValueError("Nonfinite screen features")
            np.savez_compressed(folder/f"{mode}_{split}.npz", **{**base, "X": X})
            write_json(folder/f"{mode}_names.json", found_names)
        if split == "held_out":
            np.savez_compressed(folder/"reference_audit.npz", errors=np.concatenate(reference_rows),
                families=base["families"], scenario=base["scenario"], early=base["early"], labels=base["labels"],
                support=base["X"][:, names.index("reference_support")], coverage=base["X"][:, names.index("coverage_4")])
    return folder


def reference_audit(parts):
    a = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    normal = a["families"] == 0
    supported = normal & (a["errors"][:, 2] > 0)
    base, new = np.abs(a["errors"][:, 0]), np.abs(a["errors"][:, 1])
    ratio = float(new[supported].mean()/base[supported].mean()) if supported.any() else None
    old_p95, new_p95 = float(np.quantile(base[normal], .95)), float(np.quantile(new[normal], .95))
    by_scenario = {}
    for sid in np.unique(a["scenario"]):
        loc = normal & (a["scenario"] == sid)
        by_scenario[str(int(sid))] = {"base_bias_m": float(a["errors"][loc, 0].mean()),
            "base_mae_m": float(base[loc].mean()), "physics_mae_m": float(new[loc].mean()),
            "base_p95_p99_m": np.quantile(base[loc], [.95, .99]).tolist(),
            "physics_p95_p99_m": np.quantile(new[loc], [.95, .99]).tolist(),
            "physical_fraction": float((a["errors"][loc, 2] > 0).mean())}
    support_audit = {}
    for label, loc in (("coverage_4_at_most_half", normal & (a["coverage"] <= .5)),
                       ("coverage_4_above_half", normal & (a["coverage"] > .5))):
        support_audit[label] = {"n": int(loc.sum()), "base_mae_m": float(base[loc].mean()),
                               "physics_mae_m": float(new[loc].mean())}
    return {"passed": ratio is not None and ratio <= .90 and new_p95 <= old_p95,
        "normal_points": int(normal.sum()), "supported_normal_points": int(supported.sum()),
        "supported_mae_ratio": ratio, "base_all_normal_p95_m": old_p95,
        "physics_all_normal_p95_m": new_p95, "by_scenario": by_scenario,
        "coverage_diagnostics": support_audit}


def run(output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/"campaign.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Recovery screening is already running")
    old = read_json(CONTROL/"signature.json")
    for p, expected in old["sources"].items():
        if sha(Path(p)) != expected:
            raise ValueError(f"Prior experiment source changed: {p}")
    for p, expected in old["base_signature"]["source"].items():
        if sha(Path("src/wdn")/p) != expected:
            raise ValueError(f"Baseline source changed: {p}")
    for p, expected in old["base_signature"]["data_files"].items():
        if sha(DATA/p) != expected:
            raise ValueError(f"Frozen data changed: {p}")
    config = yaml.safe_load((DATA/"generate_config.yaml").read_text())
    if any(config[f"missing_rate_{c}"] != .5 for c in ("pressure", "flow")):
        raise ValueError("Both missing rates must remain 0.50")
    if (config["pressure_noise_sigma_m"] != PhysicsConfig().pressure_sigma or
            config["flow_noise_sigma_m3s"] != PhysicsConfig().flow_sigma):
        raise ValueError("Physical uncertainty must match the frozen measurement process")
    splits, folds = read_json(CONTROL/"splits.json"), read_json(CONTROL/"inner_folds.json")
    signature = {"source": {p: sha(Path(p)) for p in SOURCE}, "prior_signature": old,
        "control_oof_sha": sha(CONTROL/"trial_0000/oof_predictions.npz"),
        "graph_sha": sha(DATA/"graph.pkl"), "inp_sha": sha(Path("data/modena.inp")),
        "train_caches": {str(p): sha(p) for p in BASE.glob("fold_*/*") if p.suffix in (".npz", ".joblib")},
        "protocol": PROTOCOL, "splits": splits, "folds": folds}
    if (output/"signature.json").exists() and read_json(output/"signature.json") != signature:
        raise ValueError("Screen source/input signature changed; refusing silent resume")
    write_json(output/"signature.json", signature)
    write_json(output/"protocol.json", PROTOCOL)
    if (output/"screening.json").exists():
        print("Screen already completed; no refit", flush=True)
        return
    started = time.monotonic()
    def status(phase, **details):
        item = {"phase": phase, "elapsed_seconds": time.monotonic()-started,
                "validation_evaluated": False, "test_evaluated": False, **details}
        write_json(output/"status.json", item)
        print(phase, details, flush=True)
    status("preflight verified; TRAIN-only screen starting")
    wn = wntr.network.WaterNetworkModel("data/modena.inp")
    if (wn.options.hydraulic.headloss != "H-W" or wn.num_controls or
            any(p.minor_loss != 0 or str(p.initial_status) != "Open" for _, p in wn.pipes())
            or any(r.head_pattern_name for _, r in wn.reservoirs())):
        raise ValueError("Static Hazen-Williams network assumptions violated")
    with (DATA/"graph.pkl").open("rb") as stream:
        graph = pickle.load(stream)
    if graph.node_names != wn.node_name_list or graph.edge_names != wn.link_name_list:
        raise ValueError("Graph/INP ordering differs")
    data = TrainOnlyData(DATA, splits)
    control = load_arrays(CONTROL/"trial_0000/oof_predictions.npz")
    all_arrays, predictions, reference_parts, raw_state = [], {"physics": [], "state": []}, [], []
    cursor = 0
    for index, held in enumerate(folds):
        folder = output/f"fold_{index}"
        if not (folder/"completed.json").exists():
            folder = prepare_fold(output, index, held, splits, data, graph, status)
        for mode in ("physics", "state"):
            training = load_arrays(folder/f"{mode}_train.npz")
            test = load_arrays(folder/f"{mode}_held_out.npz")
            if set(training["scenario"]) & set(test["scenario"]):
                raise ValueError("Scenario overlap inside screen")
            model_path = folder/f"{mode}_expert.joblib"
            if model_path.exists():
                model = joblib.load(model_path)
            else:
                status("training isolated weak experts", mode=mode, fold=index)
                model = WeakExperts(read_json(folder/f"{mode}_names.json"), **PROTOCOL["expert"]).fit(training)
                joblib.dump(model, model_path)
            score = model.predict(test["X"])
            np.testing.assert_array_equal(score, joblib.load(model_path).predict(test["X"]))
            np.savez_compressed(folder/f"{mode}_predictions.npz", scores=score)
            predictions[mode].append(score)
            del training
        base = add_early(load_arrays(BASE/f"fold_{index}/features_held_out.npz"), data.events)
        end = cursor+len(base["labels"])
        for key in ("labels", "families", "scenario"):
            np.testing.assert_array_equal(control[key][cursor:end], base[key])
        cursor = end
        names = read_json(folder/"state_names.json")
        raw_state.append(test["X"][:, [names.index("drift_run_probability"), names.index("noise_run_probability")]])
        all_arrays.append({k: base[k] for k in ("labels", "families", "scenario", "event", "timestep", "node", "early")})
        reference_parts.append(load_arrays(folder/"reference_audit.npz"))
        write_json(folder/"completed.json", {"fold": index, "model_reload_exact": True,
            "held_scenarios": held, "new_expert_fit_calls": 2, "weak_heads": 4})
    assert cursor == len(control["labels"])
    a = {key: np.concatenate([part[key] for part in all_arrays]) for key in all_arrays[0]}
    results = {"control": expert_diagnostics(a, control["experts"][:, 3:5], data.events)}
    for mode in predictions:
        scores = np.concatenate(predictions[mode])
        results[mode] = expert_diagnostics(a, scores, data.events)
        np.savez_compressed(output/f"{mode}_oof_predictions.npz", scores=scores, **a)
    results["raw_state_posterior_diagnostic"] = expert_diagnostics(a, np.concatenate(raw_state), data.events)
    reference = reference_audit(reference_parts)
    gates = {mode: {family: gate(results[mode][family], results["control"][family],
                    reference["passed"] if mode == "physics" else True)
                   for family in ("drift", "noise")} for mode in predictions}
    survivors = [{"mechanism": mode, "family": family} for mode in gates
                 for family, decision in gates[mode].items() if decision["passed"]]
    for p, expected in signature["source"].items():
        if sha(Path(p)) != expected:
            raise ValueError("Screen source changed during execution")
    result = {"scope": PROTOCOL["scope"], "protocol": PROTOCOL, "reference": reference,
        "experts": results, "gates": gates, "survivors": survivors,
        "endpoints": len(a["labels"]), "calibration_evaluated": False,
        "validation_evaluated": False, "test_evaluated": False,
        "new_weak_expert_fit_calls": 6, "new_weak_heads": 12, "optuna_trials": 0,
        "elapsed_seconds": time.monotonic()-started,
        "next_action": "planned_nested_stage" if survivors else "stop_no_component_passed"}
    write_json(output/"screening.json", result)
    status("screen completed", survivors=survivors, next_action=result["next_action"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="runs/operational/drift_noise_recovery_v1")
    run(parser.parse_args().output_dir)
