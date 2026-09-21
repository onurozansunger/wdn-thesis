"""Stage D: L-Town training-only pilot — correctness audit and resource estimate.

Nothing is selected here and no L-Town evaluation corpus is read. The pilot
answers four questions before any full run is launched:

1. Does the pipeline work at all on a 785-sensor network — masks, target-group
   exclusion, seasonal history, eligible endpoints?
2. Is the normal reference still *blind*? Perturbing a sensor's own reading must
   not move that sensor's own prediction, because its whole group is excluded.
3. What do wall time, peak resident memory and disk actually cost, measured
   rather than guessed?
4. What does that extrapolate to for the full L-Town budget?

    /opt/miniconda3/bin/python thesis_v2/experiments/early_warning/ltown_pilot.py
"""
from __future__ import annotations

import gc
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

from wdn.latency_deployment import specialist_bank
from wdn.models.blind_reference import BlindPressureReference
from wdn.models.robust_reference import RobustBlindReference
from wdn.run_expert_redesign import CampaignData

from build_feature_cache import ROOT, write_json

DATA = ROOT / "data/thesis_v2"
PILOT = DATA / "ew_ltown_ltown_pilot_seed91811"
OUTPUT = ROOT / "runs/operational/early_warning_multiseed_v1/ltown_pilot_v1"

#: Planned full L-Town budget, mirroring Modena's whole-corpus assignment rule.
FULL_TRAIN_SCENARIOS = 120     # 5 corpora x 24
FULL_CALIBRATION_SCENARIOS = 96  # 4 corpora x 24
FULL_EVAL_SCENARIOS = 144      # 6 corpora x 24

MODENA_TRAIN_SCENARIOS = 110
MODENA_CALIBRATION_SCENARIOS = 99


def peak_rss_gb():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Darwin reports ru_maxrss in bytes; Linux reports it in kilobytes.
    divisor = 1024**3 if sys.platform == "darwin" else 1024**2
    return usage / divisor


def sensor_roster(data, scenarios):
    """Mask, support and eligibility audit over the pilot's sensors."""
    observed = None
    normal_observed = None
    total_hours = 0
    for sid in scenarios:
        arrays = data.scenario(sid)
        mask = arrays["mask"]
        normal = arrays["families"] == 0
        observed = mask.sum(0) if observed is None else observed + mask.sum(0)
        block = mask[normal].sum(0)
        normal_observed = block if normal_observed is None else normal_observed + block
        total_hours += len(mask)
        if not np.array_equal(arrays["timestep"], np.arange(len(arrays["timestep"]))):
            raise ValueError(f"Scenario {sid} is not contiguous hourly time; the "
                             "seasonal features require it")
    sensors = len(observed)
    return {
        "sensors": int(sensors),
        "scenario_hours_total": int(total_hours),
        "possible_endpoints": int(total_hours * sensors),
        "observed_endpoints": int(observed.sum()),
        "realised_missing_rate": float(1 - observed.sum() / (total_hours * sensors)),
        "min_observations_per_sensor": int(observed.min()),
        "min_normal_observations_per_sensor": int(normal_observed.min()),
        "sensors_below_reference_minimum": int((normal_observed < 2).sum()),
        "reference_groups": 4,
        "group_sizes": np.bincount(np.arange(sensors) % 4).tolist(),
        "contiguous_hourly_time": True,
    }


def blindness_probe(reference, data, sid, rng):
    """Perturb one sensor's own readings; its own prediction must not move."""
    arrays = data.scenario(sid)
    values, mask = arrays["values"].copy(), arrays["mask"]
    base, _, _ = reference.predict_details(values, mask)
    sensor = int(rng.integers(values.shape[1]))
    disturbed = values.copy()
    disturbed[mask[:, sensor], sensor] += 25.0
    moved, _, _ = reference.predict_details(disturbed, mask)
    own = float(np.max(np.abs(moved[:, sensor] - base[:, sensor])))
    group = reference.reference.group_
    same_group = np.flatnonzero((group == group[sensor]) & (np.arange(len(group)) != sensor))
    other_group = np.flatnonzero(group != group[sensor])
    return {
        "probed_sensor": sensor,
        "max_change_in_own_prediction": own,
        "own_prediction_is_blind": bool(own < 1e-9),
        "max_change_in_same_group": float(np.max(np.abs(
            moved[:, same_group] - base[:, same_group]))) if len(same_group) else 0.,
        "max_change_in_other_groups": float(np.max(np.abs(
            moved[:, other_group] - base[:, other_group]))) if len(other_group) else 0.,
        "note": "a sensor's own group is excluded from its prediction, so its own "
                "column must be exactly unchanged; other groups are expected to move",
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"campaign": "early_warning_multiseed_v1", "stage": "D",
              "scope": "L-Town TRAIN-only pilot; no L-Town evaluation corpus was read",
              "pilot_corpus": PILOT.name}

    config = json.loads((PILOT / "manifest.json").read_text())
    report["generator_manifest"] = {
        "snapshots": config["snapshots"], "events": config["events"],
        "pressure_missing_rate": config["pressure"]["missing_rate"],
        "flow_missing_rate": config["flow"]["missing_rate"],
        "family_endpoints": config["family_endpoints"]}

    print("loading pilot corpus", flush=True)
    mark = time.monotonic()
    data = CampaignData(PILOT)
    scenarios = sorted({int(s.scenario_id) for s in data.snapshots})
    report["load_seconds"] = round(time.monotonic() - mark, 1)
    report["scenarios"] = len(scenarios)

    print("auditing sensor roster and masks", flush=True)
    mark = time.monotonic()
    report["roster"] = sensor_roster(data, scenarios)
    report["roster_seconds"] = round(time.monotonic() - mark, 1)
    if report["roster"]["sensors_below_reference_minimum"]:
        raise SystemExit("Some L-Town sensors lack the two normal observations the "
                         "blind reference requires; sizing must change before a full run")

    print("fitting the L-Town normal reference (network-specific, never Modena's)",
          flush=True)
    mark = time.monotonic()
    values, masks = [], []
    for sid in scenarios:
        arrays = data.scenario(sid)
        normal = arrays["families"] == 0
        values.append(arrays["values"][normal])
        masks.append(arrays["mask"][normal])
    values, masks = np.concatenate(values), np.concatenate(masks)
    base = BlindPressureReference(rank=16).fit(values, masks)
    reference = RobustBlindReference(base).calibrate_scale(values, masks)
    report["reference"] = {
        "seconds": round(time.monotonic() - mark, 1),
        "normal_rows_fitted": int(len(values)),
        "sensors": int(values.shape[1]),
        "rank": 16,
        "median_noise_scale_m": float(np.median(reference.noise_scale_)),
        "fitted_on": "L-Town TRAIN pilot only; Modena's reference is never loaded",
    }
    del values, masks
    gc.collect()

    print("probing target-group blindness", flush=True)
    report["blindness"] = blindness_probe(reference, data, scenarios[0],
                                          np.random.default_rng(4242))
    if not report["blindness"]["own_prediction_is_blind"]:
        raise SystemExit("Target-group exclusion is broken on L-Town; stop here")

    print("building the 116-column feature bank", flush=True)
    mark = time.monotonic()
    arrays, names = specialist_bank(data, scenarios, reference)
    feature_seconds = time.monotonic() - mark
    rows = len(arrays["labels"])
    report["features"] = {
        "seconds": round(feature_seconds, 1),
        "seconds_per_scenario": round(feature_seconds / len(scenarios), 2),
        "columns": len(names),
        "rows": int(rows),
        "rows_per_scenario": int(rows / len(scenarios)),
        "float32_bytes": int(rows * len(names) * 4),
        "positives": int((arrays["labels"] > 0).sum()),
        "seasonal_columns_present": bool(names[-1] == "seasonal_signed_agreement"),
    }

    cache = OUTPUT / "pilot_bank.npz"
    mark = time.monotonic()
    np.savez_compressed(cache, X=np.asarray(arrays["X"], np.float32),
                        labels=arrays["labels"], families=arrays["families"])
    report["features"]["compressed_bytes"] = int(cache.stat().st_size)
    report["features"]["save_seconds"] = round(time.monotonic() - mark, 1)
    report["peak_rss_gb"] = round(peak_rss_gb(), 2)

    per_scenario_rows = rows / len(scenarios)
    per_scenario_bytes = cache.stat().st_size / len(scenarios)
    modena_rows_per_scenario = 2303621 / MODENA_TRAIN_SCENARIOS
    report["extrapolation"] = {
        "basis": f"measured on {len(scenarios)} L-Town scenarios",
        "rows_per_scenario_ltown": round(per_scenario_rows),
        "rows_per_scenario_modena": round(modena_rows_per_scenario),
        "row_ratio_vs_modena": round(per_scenario_rows / modena_rows_per_scenario, 2),
        "planned_scenarios": {"train": FULL_TRAIN_SCENARIOS,
                              "calibration": FULL_CALIBRATION_SCENARIOS,
                              "evaluation": FULL_EVAL_SCENARIOS},
        "estimated_rows": {
            name: int(per_scenario_rows * count) for name, count in
            (("train", FULL_TRAIN_SCENARIOS), ("calibration", FULL_CALIBRATION_SCENARIOS),
             ("evaluation", FULL_EVAL_SCENARIOS))},
        "estimated_feature_cache_gb": {
            name: round(per_scenario_bytes * count / 1024**3, 2) for name, count in
            (("train", FULL_TRAIN_SCENARIOS), ("calibration", FULL_CALIBRATION_SCENARIOS),
             ("evaluation", FULL_EVAL_SCENARIOS))},
        "estimated_feature_build_minutes": {
            name: round(feature_seconds / len(scenarios) * count / 60, 1)
            for name, count in
            (("train", FULL_TRAIN_SCENARIOS), ("calibration", FULL_CALIBRATION_SCENARIOS),
             ("evaluation", FULL_EVAL_SCENARIOS))},
        "caveat": "the reference fit is superlinear in rows, so its cost is measured "
                  "again on the first full L-Town TRAIN corpus rather than scaled from here",
    }
    report["budget_comparability"] = {
        "modena_train": MODENA_TRAIN_SCENARIOS, "ltown_train": FULL_TRAIN_SCENARIOS,
        "modena_calibration": MODENA_CALIBRATION_SCENARIOS,
        "ltown_calibration": FULL_CALIBRATION_SCENARIOS,
        "rule": "whole corpora are assigned entirely to TRAIN or to calibration, as on "
                "Modena. L-Town uses five 24-scenario TRAIN corpora and four "
                "24-scenario calibration corpora, so the counts are comparable "
                "(120 vs 110, 96 vs 99) without fabricating a partial split.",
    }
    report["elapsed_seconds"] = round(time.monotonic() - started, 1)
    report["ltown_evaluation_read"] = False
    report["test_evaluated"] = False
    write_json(OUTPUT / "pilot_report.json", report)
    print(json.dumps({k: v for k, v in report.items()
                      if k in ("roster", "reference", "blindness", "features",
                               "peak_rss_gb", "extrapolation")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
