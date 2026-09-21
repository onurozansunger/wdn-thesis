"""TRAIN-only received-observation history pilot; no new expert or test access."""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score

from build_feature_cache import CAMPAIGN, ROOT, check_distribution, config_of, sha, write_json
from screen_shared_history import BUDGETS, load, predict_chunks, select, streams
from wdn.latency_deployment import FAMILY_NAMES, family_scores
from wdn.observed_history import DELTA_BIN_SIGMA, OBSERVED_LAGS, observed_history_features
from wdn.run_expert_redesign import CampaignData
from wdn.shared_history import SharedHistoryExpertMixture, fit_presampled

RESOLUTIONS = (0., .01, .05)
FOLDER = CAMPAIGN / "stage_e_ltown/folds/fold_0"
CONTROL = CAMPAIGN / "shared_history_pilot_v1/ltown/fold_0_seed_701"
OUTPUT = CAMPAIGN / "observed_history_pilot_v1/ltown/fold_0_seed_701"


def build_bank(arrays, names, manifest, resolutions):
    count = len(arrays["labels"])
    result = {r: np.empty((count, len(OBSERVED_LAGS) * 3), np.float32) for r in resolutions}
    filled = np.zeros(count, bool)
    scale_column = names.index("normal_error_scale")
    for piece in manifest["pieces"]:
        source_rows = np.flatnonzero(arrays["source"] == piece["seed"])
        if not len(source_rows):
            continue
        data = CampaignData(ROOT / "data/thesis_v2" / piece["directory"])
        for scenario in np.unique(arrays["scenario"][source_rows]):
            sid = int(scenario - piece["seed"] * 1000)
            if sid not in piece["scenarios"]:
                raise ValueError("Endpoint is not in the frozen TRAIN source/scenario manifest")
            rows = source_rows[arrays["scenario"][source_rows] == scenario]
            observed = data.scenario(sid)
            # Only received pressures/masks/times cross the feature boundary.
            # Clean hydraulic targets and event/label metadata are not supplied.
            for resolution in resolutions:
                result[resolution][rows], added_names = observed_history_features(
                    observed["values"], observed["mask"], observed["timestep"],
                    arrays["timestep"][rows], arrays["node"][rows],
                    arrays["X"][rows, scale_column], rounding_m=resolution)
            filled[rows] = True
        del data
        gc.collect()
    if not filled.all():
        raise ValueError("Some endpoints lack a source in the frozen TRAIN manifest")
    return result, added_names


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / "summary.json").exists():
        print("Already complete; no training restarted", flush=True)
        return
    start = time.time()
    def status(phase):
        print(f"[{time.time()-start:.0f}s] {phase}", flush=True)
        write_json(OUTPUT / "status.json", {"phase": phase, "elapsed_seconds": time.time()-start,
                                             "updated": time.strftime("%Y-%m-%dT%H:%M:%S")})

    manifest = json.loads((CAMPAIGN / "features/ltown_train/manifest.json").read_text())
    names = manifest["feature_names"]
    old_protocol = json.loads((CONTROL / "protocol_frozen.json").read_text())
    input_hashes = {p.name: sha(p) for p in FOLDER.glob("*.npz")}
    if input_hashes != old_protocol["input_hashes"]:
        raise RuntimeError("Existing control used different TRAIN fold inputs")
    for path, digest in old_protocol["source_hashes"].items():
        if sha(ROOT / path) != digest:
            raise RuntimeError("Existing control source has changed")
    config = config_of(ROOT / "data/thesis_v2" / manifest["pieces"][0]["directory"])
    raw_hashes = {}
    for piece in manifest["pieces"]:
        directory = ROOT / "data/thesis_v2" / piece["directory"]
        check_distribution(directory, config)
        raw_hashes[piece["directory"]] = {name: sha(directory / name)
            for name in ("snapshots.pkl", "corrupted.pkl", "generate_config.yaml")}
    protocol = {"network": "ltown", "fold": 0, "training_seed": 701,
        "input_hashes": input_hashes, "raw_hashes": raw_hashes,
        "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in
            (Path(__file__), ROOT / "src/wdn/observed_history.py",
             ROOT / "src/wdn/shared_history.py", ROOT / "src/wdn/probe_residual_experts.py",
             ROOT / "thesis_v2/experiments/early_warning/screen_shared_history.py",
             ROOT / "src/wdn/run_expert_redesign.py")},
        "control_artifact_hashes": {name: sha(CONTROL / name) for name in
            ("baseline.joblib", "baseline_predictions.npz", "summary.json", "thresholds_frozen.json")},
        "lags": OBSERVED_LAGS, "delta_bin_sigma": DELTA_BIN_SIGMA,
        "feature_rounding_sensitivity_m": RESOLUTIONS,
        "selection": "Even held-source TRAIN scenarios; maximise pooled F1, then worst-family F1, "
                     "then lower clean FPR, same frozen budget grid in both arms",
        "budget_grid": BUDGETS, "diagnostic": "Odd held-source TRAIN scenarios; exploratory, "
                     "already seen in earlier pilot, not independent confirmation",
        "scope": "Same five experts, same hyperparameters and sample weights, 116+42 features; "
                 "no deployed seasonal, feedback, verifier, or warning branch modified",
        "sensitivity_scope": "Only the additional feature path is recomputed after rounding; "
                "original 116 features and frozen model/thresholds unchanged. "
                "Not end-to-end sensor quantisation robustness or a changed primary benchmark.",
        "missing_pressure": .5, "missing_flow": .5, "test_evaluated": False,
        "locked_eval_evaluated": False}
    protocol = json.loads(json.dumps(protocol))
    path = OUTPUT / "protocol_frozen.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise RuntimeError("Pilot signature changed; refusing stale artifacts")
    write_json(path, protocol)
    if not (OUTPUT / "model.joblib").exists():
        status("loading source-excluded TRAIN; same sample as frozen control")
        train = load(FOLDER / "features_train.npz")
        positives = np.flatnonzero(train["labels"] > .5)
        negatives = np.flatnonzero(train["labels"] <= .5)
        chosen = np.random.default_rng(701).choice(negatives, min(60000, len(negatives)), replace=False)
        subset = np.r_[positives, chosen]
        weights = np.r_[np.ones(len(positives)), np.full(len(chosen), len(negatives)/len(chosen))]
        train = {k: train[k][subset] for k in train}
        gc.collect()
        status("building received-reading history for sampled training endpoints")
        banks, added_names = build_bank(train, names, manifest, (0.,))
        write_json(OUTPUT / "feature_names.json", names + added_names)
        model = SharedHistoryExpertMixture(names + added_names, seed=701)
        status("fitting same five experts and router with 42 common observation features")
        fit_presampled(model, np.column_stack((train["X"], banks[0.])), train["labels"],
                       train["families"], weights)
        joblib.dump(model, OUTPUT / "model.joblib.tmp")
        (OUTPUT / "model.joblib.tmp").replace(OUTPUT / "model.joblib")
        del train, banks, model, subset, positives, negatives
        gc.collect()

    status("building held-out TRAIN observation bank and predeclared sensitivity banks")
    held = load(FOLDER / "features_held_out.npz")
    banks, _ = build_bank(held, names, manifest, RESOLUTIONS)
    model = joblib.load(OUTPUT / "model.joblib")
    predictions = {}
    for resolution in RESOLUTIONS:
        arm = "observed_history" if resolution == 0 else f"rounding_{resolution}m"
        path = OUTPUT / f"{arm}_predictions.npz"
        if not path.exists():
            status(f"predicting {arm}; only TRAIN data")
            predicted = predict_chunks(model, held["X"], banks[resolution])
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **predicted)
            temporary.replace(path)
        predictions[arm] = load(path)
    del banks, model, held["X"]
    gc.collect()
    selection = held["scenario"] % 2 == 0
    diagnostic = ~selection
    thresholds = {name: select(score, held, selection)
                  for name, score in streams(predictions["observed_history"]).items()}
    write_json(OUTPUT / "thresholds_frozen.json", thresholds)
    status("primary thresholds frozen; auditing primary and sensitivity predictions")
    baseline = json.loads((CONTROL / "summary.json").read_text())["results"]["baseline"]
    report = {"protocol": protocol, "results": {"baseline": baseline},
        "threshold_scenarios": np.unique(held["scenario"][selection]).tolist(),
        "diagnostic_scenarios": np.unique(held["scenario"][diagnostic]).tolist()}
    scope = {k: held[k][diagnostic] for k in ("labels", "families")}
    for arm, prediction in predictions.items():
        report["results"][arm] = {}
        for name, score in streams(prediction).items():
            metrics = family_scores(score[diagnostic] > thresholds[name]["threshold"], scope)
            for code, family in FAMILY_NAMES.items():
                mask = diagnostic & np.isin(held["families"], (0, code))
                metrics[family]["auprc_family_plus_clean"] = float(
                    average_precision_score(held["labels"][mask], score[mask]))
            report["results"][arm][name] = metrics
    before, after = (report["results"][arm]["mixture"] for arm in ("baseline", "observed_history"))
    report["mixture_f1_deltas"] = {f: after[f]["f1"] - before[f]["f1"] for f in FAMILY_NAMES.values()}
    report["mixture_f1_deltas"]["pooled"] = after["_overall"]["f1"] - before["_overall"]["f1"]
    report["elapsed_seconds"] = time.time()-start
    write_json(OUTPUT / "summary.json", report)
    status("complete; exploratory TRAIN diagnostic, no deployment promotion")
    print(json.dumps(report["mixture_f1_deltas"], indent=2), flush=True)


if __name__ == "__main__":
    run()
