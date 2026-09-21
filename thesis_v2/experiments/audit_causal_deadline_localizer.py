"""Replay the causal checkpoint localizer and audit its claim boundary."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import yaml

from evaluate_causal_deadline_localizer import (BASE, DATA, MEMBERS, MODEL, NOISE_SELECTION,
    PROTOCOL, RECIPES, evaluate_split)
from evaluate_incident_window_localizer import state_arrays
from wdn.run_expert_redesign import sha, write_json

RUN = Path("runs/operational/causal_deadline_localizer_v1")


def main():
    summary = json.loads((RUN/"summary.json").read_text())
    saved = dict(np.load(RUN/"predictions_validation.npz"))
    names = json.loads((BASE/"feature_names.json").read_text())
    events = json.loads((DATA/"events.json").read_text())
    model = joblib.load(MODEL)
    bundles = [joblib.load(MEMBERS/f"outer_{i}/bundle.joblib") for i in range(3)]
    report, replay_groups = evaluate_split(state_arrays("validation", names, events),
        model, bundles, names)

    replay = {}
    for family, groups in replay_groups.items():
        for event, group in groups.items():
            sensors = group["sensors"]
            chosen = sorted([sensor for sensor in sensors if sensor["available"]],
                key=lambda sensor: (sensor["fused_rank"], -sensor["node"]), reverse=True)
            selected = {sensor["node"] for sensor in chosen[:RECIPES[family]["top_k"]]}
            prefix = f"{family}_{event}"
            replay[f"{prefix}_node"] = np.asarray([sensor["node"] for sensor in sensors])
            replay[f"{prefix}_score"] = np.asarray([sensor["fused_rank"] for sensor in sensors])
            replay[f"{prefix}_label"] = np.asarray([sensor["target"] for sensor in sensors])
            replay[f"{prefix}_decision"] = np.asarray([sensor["node"] in selected for sensor in sensors])
    predictions_exact = set(saved) == set(replay) and all(
        np.array_equal(saved[key], replay[key]) for key in saved)
    metrics_exact = all(report[family] == summary["reused_development_validation"][family]
                        for family in RECIPES)
    hashes = {"model": sha(MODEL), "noise_selection": sha(NOISE_SELECTION),
        "calibration": sha(BASE/"full/features_calibration.npz"),
        "validation": sha(BASE/"full/features_validation.npz"),
        **{str(MEMBERS/f"outer_{i}/bundle.joblib"):
            sha(MEMBERS/f"outer_{i}/bundle.joblib") for i in range(3)}}
    generation = yaml.safe_load((DATA/"generate_config.yaml").read_text())
    checks = {"validation_predictions_replayed_exact": bool(predictions_exact),
        "saved_validation_metrics_reproduced_exact": bool(metrics_exact),
        "input_hashes_match": hashes == summary["input_hashes"],
        "prediction_hash_matches": sha(RUN/"predictions_validation.npz") == summary["prediction_sha256"],
        "protocol_hash_matches": sha(PROTOCOL) == summary["selection"]["protocol_sha256"],
        "missing_rates_pressure_and_flow_equal_0_50":
            generation["missing_rate_pressure"] == generation["missing_rate_flow"] == .5 and
            summary["pressure_missing_probability"] == summary["flow_missing_probability"] == .5,
        "retrospective_localizer_unchanged": summary["retrospective_localizer_unchanged"],
        "closure_decisions_disclosed": all(item["decision_at_closure"] for family in report.values()
            for item in family["incidents"].values()),
        "test_not_evaluated": not summary["test_evaluated"]}
    audit = {**checks, "all_checks_pass": bool(all(checks.values()))}
    write_json(RUN/"independent_audit.json", audit)
    result = {"summary": summary, "independent_audit": audit,
        "all_checks_pass": audit["all_checks_pass"],
        "claim_limits": {"metric": "sensor-event F1 at tenth checkpoint or closure",
            "strict_pointwise_online_improved": False, "early_continuous_localizer": False,
            "external_alarm_and_family_required": True,
            "reused_development_validation": True, "test_evaluated": False}}
    write_json(Path("thesis_v2/outputs/causal_deadline_localizer_results.json"), result)
    if not audit["all_checks_pass"]:
        raise AssertionError(json.dumps(checks, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
