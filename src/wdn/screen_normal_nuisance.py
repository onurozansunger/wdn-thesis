"""Scenario-nested TRAIN-only screen for conditional normal reference error."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import fcntl
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from scipy.special import ndtr
import yaml

from wdn.conditional_change import ConditionalChangeConfig, ConditionalChangeFilter
from wdn.marked_change import MarkedChangeFilter
from wdn.models.conditional_normal import ConditionalNormalConfig, ConditionalNormalError
from wdn.normal_context import NAMES as CONTEXT_NAMES, normal_context_features
from wdn.run_expert_redesign import CampaignData, load_arrays, read_json, sha, write_json
from wdn.screen_recovery import expert_diagnostics, gate, post_event_mask
from wdn.train_weak_families import add_early


BASE = Path("runs/operational/mechanism_redesign_v1")
CONTROL = Path("runs/operational/weak_family_v1")
RECOVERY = Path("runs/operational/drift_noise_recovery_v1")
DATA = Path("data/thesis_v2/operational_modena_seed811")
SOURCE = ("src/wdn/screen_normal_nuisance.py", "src/wdn/conditional_change.py",
          "src/wdn/models/conditional_normal.py", "src/wdn/normal_context.py",
          "thesis_v2/NORMAL_NUISANCE_STAGE_A_PROTOCOL.md")
NORMAL_CONFIG = ConditionalNormalConfig(minimum_scale=.10, conditional_scale=False)
SCALE_CONFIG = replace(NORMAL_CONFIG, fit_mean=False)
CHANGE_CONFIG = ConditionalChangeConfig()
PROTOCOL = {"version": 1, "stage": "normal_nuisance_A",
    "normal_model": asdict(NORMAL_CONFIG), "scale_control": asdict(SCALE_CONFIG),
    "change_filter": asdict(CHANGE_CONFIG),
    "normal_gate": {"macro_mae_ratio": .90, "improved_scenarios": 10,
        "macro_nll_improves": True, "no_outer_nll_worse": True,
        "every_outer_abs_z_gt3_decreases": True, "macro_mean_width95_ratio": 1.75},
    "evidence_gate": {"macro_ap_gain": .05, "improved_scenarios": 3,
        "max_worst_ap_drop": .05, "diagnostic_fpr": .001,
        "minimum_control_macro_ap_fraction": .75,
        "clean_and_post_event_fp_not_increased": True},
    "secondary_diagnostic_fpr": .005,
    "logical_inner_reference_fits": 6, "maximum_unique_reference_fits": 3,
    "nuisance_bundle_fits": 6, "weak_head_fits": 0, "optuna_trials": 0,
    "calibration_evaluated": False, "validation_evaluated": False,
    "test_evaluated": False}
PROTOCOL = json.loads(json.dumps(PROTOCOL))
EXPECTED_FOLDS = [[5, 12, 17, 21, 23], [7, 14, 15, 19, 22], [2, 9, 16, 18]]


def atomic_npz(path, **arrays):
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def atomic_joblib(value, path):
    temporary = path.with_suffix(path.suffix+".tmp")
    joblib.dump(value, temporary)
    temporary.replace(path)


class TrainOnlyCampaign(CampaignData):
    def __init__(self, directory, splits):
        self.allowed = set(splits["train"])
        forbidden = set(splits["calibration"]+splits["validation"]+splits["test"])
        if self.allowed & forbidden:
            raise ValueError("Outer splits overlap")
        self.accessed = set()
        super().__init__(directory)

    def scenario(self, sid):
        if sid not in self.allowed:
            raise ValueError("Stage A may extract TRAIN scenarios only")
        self.accessed.add(int(sid))
        return super().scenario(sid)


def scenario_context(data, reference, sid):
    a = data.scenario(sid)
    prediction, support, disagreement = reference.predict_details(a["values"], a["mask"])
    context, names = normal_context_features(
        prediction, support, disagreement, reference.noise_scale_, a["timestep"])
    if names != CONTEXT_NAMES:
        raise ValueError("Normal context schema changed")
    return a, prediction, support, disagreement, context


def normal_fit_rows(data, reference, scenarios):
    parts = []
    for sid in sorted(scenarios):
        a, prediction, _, _, context = scenario_context(data, reference, sid)
        eligible = a["mask"][15:] & (a["families"][15:, None] == 0)
        residual = a["values"][15:]-prediction[15:]
        timestep = np.broadcast_to(a["timestep"][15:, None], eligible.shape)
        node = np.broadcast_to(np.arange(eligible.shape[1]), eligible.shape)
        parts.append({"X": context[15:][eligible], "residual": residual[eligible],
            "scenario": np.full(int(eligible.sum()), sid, dtype=np.int16),
            "timestep": timestep[eligible].astype(np.int16),
            "node": node[eligible].astype(np.int16)})
    result = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    keys = np.column_stack([result["scenario"], result["timestep"], result["node"]])
    if len(np.unique(keys, axis=0)) != len(keys) or not np.isfinite(result["X"]).all():
        raise ValueError("Directed normal rows must have unique finite scenario/time/node keys")
    return result


def concatenate(parts):
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def endpoint_key_digest(a):
    keys = np.ascontiguousarray(np.column_stack(
        [a["scenario"], a["timestep"], a["node"]]).astype("<i8"))
    return hashlib.sha256(keys.view(np.uint8)).hexdigest()


def observation_metadata(a, events):
    endpoint = a["mask"][15:]
    shape = endpoint.shape
    timestep = np.broadcast_to(a["timestep"][15:, None], shape)
    node = np.broadcast_to(np.arange(shape[1]), shape)
    family = np.broadcast_to(a["families"][15:, None], shape)
    event = np.full(shape, -1, dtype=np.int16)
    early = np.zeros(shape, dtype=bool)
    for item in events:
        if item["scenario_id"] != int(a["scenario_id"] if "scenario_id" in a else -1):
            continue
        active = ((a["timestep"][15:] >= item["start_timestep"])
                  & (a["timestep"][15:] < item["start_timestep"]+item["actual_steps"]))
        event[active] = item["_event_id"]
        first = ((a["timestep"][15:] >= item["start_timestep"])
                 & (a["timestep"][15:] < item["start_timestep"]+3))
        early[first] = True
    return {"labels": a["labels"][15:][endpoint], "families": family[endpoint],
        "event": event[endpoint], "timestep": timestep[endpoint].astype(np.int16),
        "node": node[endpoint].astype(np.int16), "early": early[endpoint]}


def gaussian_metrics(error, mean, scale):
    z = (error-mean)/scale
    phi = np.exp(-.5*z*z)/np.sqrt(2*np.pi)
    width = 2*1.959963984540054*scale
    return {"points": int(len(z)), "signed_bias_m": float(np.mean(error-mean)),
        "mae_m": float(np.mean(np.abs(error-mean))),
        "rmse_m": float(np.sqrt(np.mean((error-mean)**2))),
        "nll": float(np.mean(.5*(np.log(2*np.pi*scale**2)+z*z))),
        "crps_m": float(np.mean(scale*(z*(2*ndtr(z)-1)+2*phi-1/np.sqrt(np.pi)))),
        "abs_z_p95": float(np.quantile(np.abs(z), .95)),
        "abs_z_p99": float(np.quantile(np.abs(z), .99)),
        "fraction_abs_z_gt2": float((np.abs(z) > 2).mean()),
        "fraction_abs_z_gt3": float((np.abs(z) > 3).mean()),
        "coverage_90": float((np.abs(z) <= 1.6448536269514722).mean()),
        "coverage_95": float((np.abs(z) <= 1.959963984540054).mean()),
        "mean_width95_m": float(np.mean(width)),
        "median_width95_m": float(np.median(width)),
        "p95_width95_m": float(np.quantile(width, .95))}


def grouped_metrics(error, mean, scale, selected, scenario, outer):
    if not selected.any():
        return {"points": 0, "by_scenario": {}, "by_outer": {},
                "scenario_macro": None, "pooled": None}
    by_scenario = {str(int(sid)): gaussian_metrics(error[loc], mean[loc], scale[loc])
        for sid in np.unique(scenario[selected])
        for loc in [selected & (scenario == sid)]}
    by_outer = {str(int(fold)): gaussian_metrics(error[loc], mean[loc], scale[loc])
        for fold in np.unique(outer[selected])
        for loc in [selected & (outer == fold)]}
    keys = [key for key in next(iter(by_scenario.values())) if key != "points"]
    macro = {key: float(np.mean([row[key] for row in by_scenario.values()])) for key in keys}
    macro["scenarios"] = len(by_scenario)
    return {"points": int(selected.sum()), "by_scenario": by_scenario,
            "by_outer": by_outer, "scenario_macro": macro,
            "pooled": gaussian_metrics(error[selected], mean[selected], scale[selected])}


def normal_diagnostics(a, events):
    post = (post_event_mask(a, events, "stealthy")
            | post_event_mask(a, events, "noise"))
    groups = {"clean": a["families"] == 0,
              "all_label_zero": a["labels"] == 0, "post_event": post}
    result = {}
    for arm in ("B", "S", "M"):
        result[arm] = {name: grouped_metrics(a["error"], a[f"mean_{arm}"],
            a[f"scale_{arm}"], selected, a["scenario"], a["outer_fold"])
            for name, selected in groups.items()}
    return result


def normal_gate(result):
    b, m = result["B"]["clean"], result["M"]["clean"]
    bmacro, mmacro = b["scenario_macro"], m["scenario_macro"]
    improved = sum(m["by_scenario"][sid]["mae_m"] < row["mae_m"]
                   for sid, row in b["by_scenario"].items())
    checks = {"macro_mae_ratio": mmacro["mae_m"] <= .90*bmacro["mae_m"],
        "at_least_ten_scenarios_improve_mae": improved >= 10,
        "macro_nll_improves": mmacro["nll"] < bmacro["nll"],
        "no_outer_nll_worse": all(m["by_outer"][key]["nll"] <= row["nll"]
                                  for key, row in b["by_outer"].items()),
        "every_outer_abs_z_gt3_decreases": all(
            m["by_outer"][key]["fraction_abs_z_gt3"] < row["fraction_abs_z_gt3"]
            for key, row in b["by_outer"].items()),
        "interval_width_bounded": mmacro["mean_width95_m"] <= 1.75*bmacro["mean_width95_m"]}
    return {"passed": all(checks.values()), "checks": checks,
        "macro_mae_ratio": mmacro["mae_m"]/bmacro["mae_m"],
        "macro_mean_width95_ratio": mmacro["mean_width95_m"]/bmacro["mean_width95_m"],
        "improved_scenarios": improved}


def secondary_curve(a, scores, events, fpr=.005):
    result = {}
    for col, fid, family, event_family in ((0, 3, "drift", "stealthy"),
                                           (1, 4, "noise", "noise")):
        decision = np.zeros(len(scores), dtype=bool)
        early = {}
        for sid in np.unique(a["scenario"]):
            scenario = a["scenario"] == sid
            normal = scores[scenario & (a["labels"] == 0), col]
            threshold = np.quantile(normal, 1-fpr, method="higher")
            decision[scenario] = scores[scenario, col] > threshold
            loc = scenario & (a["families"] == fid) & a["early"] & (a["labels"] > 0)
            if loc.any():
                early[str(int(sid))] = float(decision[loc].mean())
        clean = a["families"] == 0
        post = post_event_mask(a, events, event_family)
        result[family] = {"early_recall_by_scenario": early,
            "early_macro_recall": float(np.mean(list(early.values()))),
            "normal_fp": int(decision[a["labels"] == 0].sum()),
            "clean_fp": int(decision[clean].sum()),
            "post_event_fp": int(decision[post].sum()), "diagnostic_fpr": fpr}
    return result


def run(output_dir, preflight_only=False):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/"campaign.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Normal nuisance Stage A is already running") from error

    prior = read_json(RECOVERY/"signature.json")
    for path, expected in prior["source"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"Frozen recovery source changed: {path}")
    for path, expected in prior["prior_signature"]["sources"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"Frozen weak-family source changed: {path}")
    for path, expected in prior["prior_signature"]["base_signature"]["data_files"].items():
        if sha(DATA/path) != expected:
            raise ValueError(f"Frozen data changed: {path}")
    for path, expected in prior["prior_signature"]["base_signature"]["source"].items():
        if sha(Path("src/wdn")/path) != expected:
            raise ValueError(f"Frozen baseline source changed: {path}")
    for path, expected in prior["train_caches"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"Frozen TRAIN cache changed: {path}")
    if sha(CONTROL/"trial_0000/oof_predictions.npz") != prior["control_oof_sha"]:
        raise ValueError("Frozen local-control OOF predictions changed")
    config = yaml.safe_load((DATA/"generate_config.yaml").read_text())
    if config["missing_rate_pressure"] != .5 or config["missing_rate_flow"] != .5:
        raise ValueError("Pressure and flow missing rates must remain 0.50")
    splits, folds = prior["splits"], prior["folds"]
    if (folds != EXPECTED_FOLDS or folds != read_json(CONTROL/"inner_folds.json")
            or any(set(left) & set(right) for i, left in enumerate(folds) for right in folds[i+1:])
            or set().union(*map(set, folds)) != set(splits["train"])):
        raise ValueError("Frozen TRAIN folds changed")
    signature = {"sources": {path: sha(Path(path)) for path in SOURCE},
        "protocol": PROTOCOL, "prior_signature_sha": sha(RECOVERY/"signature.json"),
        "control_oof_sha": sha(CONTROL/"trial_0000/oof_predictions.npz"),
        "historical_recovery_sha": sha(RECOVERY/"screening.json"),
        "outer_references": {str(path): sha(path) for path in BASE.glob("fold_*/reference.joblib")},
        "recovery_state_caches": {str(path): sha(path) for path in sorted(RECOVERY.glob("fold_*/scenario_*.npz"))},
        "recovery_state_schemas": {str(path): sha(path) for path in sorted(RECOVERY.glob("fold_*/marked_names.json"))},
        "generate_config_sha": sha(DATA/"generate_config.yaml"),
        "data_config_sha": prior["prior_signature"]["base_signature"]["data_config"],
        "splits": splits, "folds": folds}
    if (output/"signature.json").exists() and read_json(output/"signature.json") != signature:
        raise ValueError("Stage A source/input signature changed; refusing silent resume")
    write_json(output/"signature.json", signature)
    write_json(output/"protocol.json", PROTOCOL)
    write_json(output/"outer_folds.json", folds)
    if preflight_only:
        print("Stage A preflight complete; no data extracted and no model fitted", flush=True)
        return
    if (output/"summary.json").exists():
        print("Normal nuisance Stage A already complete; no refit", flush=True)
        return
    started = time.monotonic()

    def status(phase, **details):
        row = {"phase": phase, "elapsed_seconds": time.monotonic()-started,
               "calibration_evaluated": False, "validation_evaluated": False,
               "test_evaluated": False, **details}
        write_json(output/"status.json", row)
        print(phase, details, flush=True)

    status("preflight verified; nested TRAIN-only fits starting")
    data = TrainOnlyCampaign(DATA, splits)
    events = [{**event, "_event_id": event_id} for event_id, event in enumerate(data.events)
              if event["scenario_id"] in data.allowed]
    shared = output/"shared_references"
    shared.mkdir(exist_ok=True)
    references = {}
    for index, training in enumerate(folds):
        folder = shared/f"train_F{index}"
        folder.mkdir(exist_ok=True)
        path = folder/"reference.joblib"
        completed = folder/"completed.json"
        if not completed.exists():
            status("fitting unique inner reference", inner_fold=index, scenarios=training)
            reference = data.reference(training)
            atomic_joblib(reference, path)
            write_json(folder/"fit_scope.json", {"normal_family_only": True,
                "fit_scenarios": training, "forbidden_scenarios": sorted(data.allowed-set(training))})
            write_json(completed, {"model_sha256": sha(path), "fit_calls": 1,
                "fit_scenarios": training, "reload_scale_exact": bool(np.array_equal(
                    reference.noise_scale_, joblib.load(path).noise_scale_)),
                "fit_scope_sha256": sha(folder/"fit_scope.json")})
        record = read_json(completed)
        if (sha(path) != record["model_sha256"] or record["fit_scenarios"] != training
                or sha(folder/"fit_scope.json") != record["fit_scope_sha256"]):
            raise ValueError("Inner reference artifact mismatch")
        references[index] = joblib.load(path)

    directed = output/"directed_predictions"
    directed.mkdir(exist_ok=True)
    for train_index, reference in references.items():
        for held_index, held in enumerate(folds):
            if train_index == held_index:
                continue
            folder = directed/f"train_F{train_index}__held_F{held_index}"
            folder.mkdir(exist_ok=True)
            path, completed = folder/"normal_rows.npz", folder/"completed.json"
            if not completed.exists():
                status("extracting directed inner-OOF normal residuals",
                       trained_fold=train_index, held_fold=held_index)
                rows = normal_fit_rows(data, reference, held)
                atomic_npz(path, **rows)
                write_json(completed, {"reference_sha256": sha(shared/f"train_F{train_index}/reference.joblib"),
                    "rows_sha256": sha(path), "fit_scenarios": folds[train_index],
                    "predicted_scenarios": held, "normal_rows": len(rows["residual"]),
                    "key_digest": endpoint_key_digest(rows)})
            record = read_json(completed)
            cached_rows = load_arrays(path)
            if (sha(path) != record["rows_sha256"]
                    or record["reference_sha256"] != sha(shared/f"train_F{train_index}/reference.joblib")
                    or record["fit_scenarios"] != folds[train_index]
                    or record["predicted_scenarios"] != held
                    or record["key_digest"] != endpoint_key_digest(cached_rows)
                    or record["normal_rows"] != len(cached_rows["residual"])
                    or set(record["fit_scenarios"]) & set(record["predicted_scenarios"])):
                raise ValueError("Directed OOF cache mismatch or overlap")
    write_json(output/"context_names.json", CONTEXT_NAMES)

    filter_model = ConditionalChangeFilter()
    arm_scores = {arm: [] for arm in ("C", "B", "S", "M")}
    all_parts = []
    control = load_arrays(CONTROL/"trial_0000/oof_predictions.npz")
    cursor = 0
    for outer, held in enumerate(folds):
        folder = output/f"outer_{outer}"
        folder.mkdir(exist_ok=True)
        completed = folder/"completed.json"
        if not completed.exists():
            inner = [index for index in range(len(folds)) if index != outer]
            rows = concatenate([load_arrays(directed/f"train_F{inner[0]}__held_F{inner[1]}/normal_rows.npz"),
                                load_arrays(directed/f"train_F{inner[1]}__held_F{inner[0]}/normal_rows.npz")])
            if set(rows["scenario"]) != set(folds[inner[0]]+folds[inner[1]]) or set(rows["scenario"]) & set(held):
                raise ValueError("Outer nuisance fit rows violate scenario exclusion")
            status("fitting frozen nuisance bundles", outer_fold=outer, held_scenarios=held)
            scale_model = ConditionalNormalError(CONTEXT_NAMES, SCALE_CONFIG).fit(
                rows["X"], rows["residual"], node=rows["node"], scenario=rows["scenario"])
            mean_model = ConditionalNormalError(CONTEXT_NAMES, NORMAL_CONFIG).fit(
                rows["X"], rows["residual"], node=rows["node"], scenario=rows["scenario"])
            atomic_joblib(scale_model, folder/"scale_control.joblib")
            atomic_joblib(mean_model, folder/"conditional_mean.joblib")
            predictions, parts = {arm: [] for arm in arm_scores}, []
            outer_reference = joblib.load(BASE/f"fold_{outer}/reference.joblib")
            state_names = read_json(RECOVERY/f"fold_{outer}/marked_names.json")
            for sid in sorted(held):
                a, prediction, _, _, context = scenario_context(data, outer_reference, sid)
                T, N = a["values"].shape
                flat, node = context.reshape(T*N, -1), np.tile(np.arange(N), T)
                mean_s, scale_s = scale_model.predict(flat, node=node)
                mean_m, scale_m = mean_model.predict(flat, node=node)
                mean_s, scale_s = mean_s.reshape(T, N), scale_s.reshape(T, N)
                mean_m, scale_m = mean_m.reshape(T, N), scale_m.reshape(T, N)
                mean_b, scale_b = np.zeros_like(prediction), np.broadcast_to(
                    outer_reference.noise_scale_, prediction.shape)
                change = {}
                for arm, mean, scale in (("B", mean_b, scale_b), ("S", mean_s, scale_s),
                                         ("M", mean_m, scale_m)):
                    change[arm], names = filter_model.transform(a["values"], a["mask"], prediction,
                        mean, scale, outer_reference.noise_scale_)
                    if names[:2] != ["conditional_drift_probability", "conditional_noise_probability"]:
                        raise ValueError("Conditional change schema differs")
                old = load_arrays(RECOVERY/f"fold_{outer}/scenario_{sid:02d}.npz")
                np.testing.assert_array_equal(old["fallback_prediction"], prediction)
                fresh, old_names = MarkedChangeFilter().transform(
                    a["values"], a["mask"], prediction, outer_reference.noise_scale_)
                np.testing.assert_array_equal(fresh, old["state"])
                if old_names != state_names:
                    raise ValueError("Historical state schema differs")
                endpoint = a["mask"][15:]
                meta = observation_metadata({**a, "scenario_id": sid}, events)
                base = add_early(load_arrays(BASE/f"fold_{outer}/features_held_out.npz"), events)
                loc = base["scenario"] == sid
                for key in ("labels", "families", "event", "timestep", "node", "early"):
                    np.testing.assert_array_equal(meta[key], base[key][loc])
                part = {**meta, "scenario": np.full(int(endpoint.sum()), sid, dtype=np.int16),
                    "outer_fold": np.full(int(endpoint.sum()), outer, dtype=np.int8),
                    "error": (a["values"][15:]-prediction[15:])[endpoint],
                    "mean_B": mean_b[15:][endpoint], "scale_B": scale_b[15:][endpoint],
                    "mean_S": mean_s[15:][endpoint], "scale_S": scale_s[15:][endpoint],
                    "mean_M": mean_m[15:][endpoint], "scale_M": scale_m[15:][endpoint]}
                parts.append(part)
                predictions["C"].append(old["state"][endpoint][:, [
                    state_names.index("drift_run_probability"),
                    state_names.index("noise_run_probability")]])
                for arm in ("B", "S", "M"):
                    predictions[arm].append(change[arm][endpoint][:, :2])
            held_arrays = concatenate(parts)
            held_scores = {arm: np.concatenate(value) for arm, value in predictions.items()}
            atomic_npz(folder/"held_predictions.npz", **held_arrays,
                       **{f"scores_{arm}": value for arm, value in held_scores.items()})
            reloaded_scale, reloaded_mean = joblib.load(folder/"scale_control.joblib"), joblib.load(folder/"conditional_mean.joblib")
            probe = rows["X"][:1000]
            probe_node = rows["node"][:1000]
            for before, after in zip(scale_model.predict(probe, node=probe_node),
                                     reloaded_scale.predict(probe, node=probe_node)):
                np.testing.assert_array_equal(before, after)
            for before, after in zip(mean_model.predict(probe, node=probe_node),
                                     reloaded_mean.predict(probe, node=probe_node)):
                np.testing.assert_array_equal(before, after)
            write_json(folder/"fit_scope.json", {"outer_fold": outer,
                "held_scenarios": held,
                "nuisance_fit_scenarios": sorted(int(sid) for sid in set(rows["scenario"])),
                "normal_rows": len(rows["residual"]), "family_zero_only": True})
            write_json(completed, {"predictions_sha256": sha(folder/"held_predictions.npz"),
                "scale_model_sha256": sha(folder/"scale_control.joblib"),
                "mean_model_sha256": sha(folder/"conditional_mean.joblib"),
                "nuisance_bundle_fit_calls": 2, "reload_prediction_exact": True,
                "held_scenarios": held, "fit_scope_sha256": sha(folder/"fit_scope.json")})
        record = read_json(completed)
        scope = read_json(folder/"fit_scope.json")
        expected_fit = sorted(set(splits["train"])-set(held))
        if (record["held_scenarios"] != held or scope["held_scenarios"] != held
                or scope["nuisance_fit_scenarios"] != expected_fit
                or sha(folder/"fit_scope.json") != record["fit_scope_sha256"]):
            raise ValueError("Completed outer fit scope mismatch")
        for name, expected in (("held_predictions.npz", record["predictions_sha256"]),
                ("scale_control.joblib", record["scale_model_sha256"]),
                ("conditional_mean.joblib", record["mean_model_sha256"])):
            if sha(folder/name) != expected:
                raise ValueError("Completed outer artifact hash mismatch")
        held_arrays = load_arrays(folder/"held_predictions.npz")
        for arm in arm_scores:
            arm_scores[arm].append(held_arrays.pop(f"scores_{arm}"))
        all_parts.append(held_arrays)
        end = cursor+len(held_arrays["labels"])
        for key in ("labels", "families", "scenario"):
            np.testing.assert_array_equal(control[key][cursor:end], held_arrays[key])
        cursor = end
    if cursor != len(control["labels"]):
        raise ValueError("Outer predictions do not cover frozen TRAIN OOF rows")
    joined = concatenate(all_parts)
    scores = {arm: np.concatenate(parts) for arm, parts in arm_scores.items()}
    diagnostics = {arm: expert_diagnostics(joined, score, events) for arm, score in scores.items()}
    diagnostics["local_control"] = expert_diagnostics(joined, control["experts"][:, 3:5], events)
    normal = normal_diagnostics(joined, events)
    normal_decision = normal_gate(normal)
    gates, survivors = {}, []
    for family in ("drift", "noise"):
        decision = gate(diagnostics["M"][family], diagnostics["B"][family], True)
        control_fraction = (diagnostics["M"][family]["macro_ap"]
                            / diagnostics["local_control"][family]["macro_ap"])
        decision["checks"]["minimum_control_macro_ap_fraction"] = control_fraction >= .75
        decision["checks"]["global_normal_gate"] = normal_decision["passed"]
        decision["passed"] = all(decision["checks"].values())
        decision["control_macro_ap_fraction"] = control_fraction
        decision["comparison"] = "M versus coherent B; global normal gate also required"
        gates[family] = decision
        if decision["passed"]:
            survivors.append(family)
    secondary = {arm: secondary_curve(joined, score, events, .005) for arm, score in scores.items()}
    atomic_npz(output/"oof_predictions.npz", **joined,
               **{f"scores_{arm}": score for arm, score in scores.items()})
    for path, expected in signature["sources"].items():
        if sha(Path(path)) != expected:
            raise ValueError("Stage A source changed during execution")
    result = {"scope": "TRAIN-only nested normal nuisance screen", "protocol": PROTOCOL,
        "normal_diagnostics": normal, "normal_gate": normal_decision,
        "evidence_diagnostics": diagnostics, "evidence_gates": gates,
        "secondary_fpr_005": secondary, "survivors": survivors,
        "endpoints": len(joined["labels"]), "accessed_scenarios": sorted(data.accessed),
        "actual_unique_reference_fit_calls": 3, "logical_inner_reference_fits": 6,
        "nuisance_bundle_fit_calls": 6, "weak_head_fit_calls": 0,
        "optuna_trials": 0, "calibration_evaluated": False,
        "validation_evaluated": False, "test_evaluated": False,
        "elapsed_seconds": time.monotonic()-started,
        "next_action": ("eligible_for_separate_specialist_screen" if survivors
                        else "stop_no_normal_nuisance_candidate_passed")}
    write_json(output/"summary.json", result)
    write_json(output/"verification.json", {"model_reload_exact": True,
        "old_state_recomputed_exact": True, "outer_metadata_matches_frozen_cache": True,
        "all_artifact_hashes_checked": True, "source_hashes_match": True,
        "accessed_train_scenarios_only": set(data.accessed) <= set(splits["train"]),
        "calibration_evaluated": False, "validation_evaluated": False,
        "test_evaluated": False})
    status("Stage A complete", survivors=survivors, next_action=result["next_action"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="runs/operational/normal_nuisance_stage_a_v1")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    run(args.output_dir, args.preflight_only)
