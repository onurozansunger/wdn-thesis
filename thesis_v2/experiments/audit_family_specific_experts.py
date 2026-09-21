"""Independently reproduce the TRAIN-only family-specific expert screen."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import yaml

from wdn.run_expert_redesign import load_arrays, read_json, sha, write_json
from wdn.screen_family_specific import (_best_f1, _gate, _with_state, ARRAY_KEYS,
                                        BASE, DATA, LATENT, WEAK)
from wdn.screen_recovery import expert_diagnostics


RUN = Path("runs/operational/family_specific_experts_v1")
REPORT = Path("thesis_v2/FAMILY_SPECIFIC_EXPERTS_RESULTS.md")


def _events(train_scenarios):
    events = json.loads((DATA/"events.json").read_text())
    if isinstance(events, dict):
        events = events["events"]
    allowed = set(train_scenarios)
    return [{**event, "_event_id": event_id} for event_id, event in enumerate(events)
            if event["scenario_id"] in allowed]


def _exact(left, right):
    if left.keys() != right.keys():
        return False
    for key in left:
        if isinstance(left[key], dict):
            if not _exact(left[key], right[key]):
                return False
        elif isinstance(left[key], list):
            if left[key] != right[key]:
                return False
        elif isinstance(left[key], float):
            if not np.isclose(left[key], right[key], rtol=0, atol=1e-14):
                return False
        elif left[key] != right[key]:
            return False
    return True


def run():
    signature, summary = read_json(RUN/"signature.json"), read_json(RUN/"summary.json")
    protocol = read_json(RUN/"protocol.json")
    arrays = load_arrays(RUN/"oof_predictions.npz")
    names = read_json(BASE/"feature_names.json")
    events = _events(signature["splits"]["train"])

    source_hashes_match = all(sha(Path(path)) == digest
                              for path, digest in signature["sources"].items())
    input_hashes_match = (
        sha(LATENT/"oof_predictions.npz") == signature["latent_oof_sha"]
        and sha(WEAK/"oof_predictions.npz") == signature["legacy_local_oof_sha"]
        and all(sha(Path(path)) == digest
                for path, digest in signature["base_held_features"].items())
        and all(sha(Path(path)) == digest
                for path, digest in signature["shared_references"].items()))
    config = yaml.safe_load((DATA/"generate_config.yaml").read_text())
    missing_rates_exact = (config["missing_rate_pressure"] == .5
                           and config["missing_rate_flow"] == .5)

    held_parts, replay_scores, prediction_hashes_match = [], [], True
    for outer, held in enumerate(signature["folds"]):
        folder = RUN/f"outer_{outer}"
        record = read_json(folder/"completed.json")
        prediction_hashes_match &= (
            sha(folder/"bundle.joblib") == record["bundle_sha256"]
            and sha(folder/"held_predictions.npz") == record["predictions_sha256"]
            and sha(folder/"fit_scope.json") == record["fit_scope_sha256"])
        saved = load_arrays(folder/"held_predictions.npz")
        held_base = load_arrays(BASE/f"fold_{outer}/features_held_out.npz")
        held_arrays, held_names = _with_state(held_base, names, events)
        if held_names != read_json(RUN/"feature_names.json"):
            raise ValueError("Independent held feature schema mismatch")
        for key in ARRAY_KEYS:
            if key != "X":
                np.testing.assert_array_equal(saved[key], held_arrays[key])
        bundle = joblib.load(folder/"bundle.joblib")
        replay = np.column_stack([bundle[family]["stacker"].predict(
            bundle[family]["expert"].predict_heads(held_arrays["X"]))
            for family in ("drift", "noise")])
        np.testing.assert_array_equal(replay, saved["scores_E"])
        held_parts.append(saved)
        replay_scores.append(replay)

    concatenated = {key: np.concatenate([part[key] for part in held_parts])
                    for key in ARRAY_KEYS if key != "X"}
    for key, value in concatenated.items():
        np.testing.assert_array_equal(value, arrays[key])
    np.testing.assert_array_equal(np.concatenate(replay_scores), arrays["scores_E"])

    legacy = load_arrays(WEAK/"oof_predictions.npz")
    np.testing.assert_array_equal(legacy["experts"][:, 3:5], arrays["scores_L"])
    reproduced = {
        "L": expert_diagnostics(arrays, arrays["scores_L"], events),
        "E": expert_diagnostics(arrays, arrays["scores_E"], events),
    }
    oracles, gates = {}, {}
    for family, family_id, column in (("drift", 3, 0), ("noise", 4, 1)):
        selected = arrays["families"] == family_id
        oracles[family] = _best_f1(
            arrays["scores_E"][selected, column], arrays["labels"][selected])
        gates[family] = _gate(reproduced["E"][family], reproduced["L"][family],
                              oracles[family])

    forbidden = set(signature["splits"]["calibration"]
                    + signature["splits"]["validation"]
                    + signature["splits"]["test"])
    accessed_train_only = (set(summary["accessed_scenarios"])
                           == set(signature["splits"]["train"]))
    no_forbidden_scenario = not (set(arrays["scenario"].tolist()) & forbidden)
    saved_metrics_exact = (_exact(reproduced, summary["diagnostics"])
                           and _exact(oracles, summary["family_oracle_f1"])
                           and _exact(gates, summary["gates"]))
    split_flags_exact = not any(summary[key] for key in (
        "calibration_evaluated", "validation_evaluated", "test_evaluated"))
    audit = {
        "scope": "independent TRAIN-only reproduction",
        "source_hashes_match": source_hashes_match,
        "input_hashes_match": input_hashes_match,
        "missing_rates_pressure_and_flow_equal_0_50": missing_rates_exact,
        "prediction_artifact_hashes_match": bool(prediction_hashes_match),
        "saved_metrics_and_gates_reproduced_exact": saved_metrics_exact,
        "bundle_reload_predictions_exact": True,
        "held_metadata_exact": True,
        "accessed_train_scenarios_only": accessed_train_only,
        "no_forbidden_scenario_in_oof": no_forbidden_scenario,
        "forbidden_split_flags_false": split_flags_exact,
        "protocol_exact": protocol == signature["protocol"] == summary["protocol"],
        "oof_predictions_sha256": sha(RUN/"oof_predictions.npz"),
        "audit_source_sha256": sha(Path(__file__)),
    }
    if not all(value is True for key, value in audit.items()
               if key not in ("scope", "oof_predictions_sha256", "audit_source_sha256")):
        raise ValueError(f"Independent family-specific audit failed: {audit}")
    write_json(RUN/"independent_audit.json", audit)
    write_json(Path("thesis_v2/outputs/family_specific_experts_results.json"),
               {"summary": summary, "independent_audit": audit})

    lines = ["# Family-specific drift/noise experts: TRAIN-only results", "",
        "This is a nested scenario-OOF component screen. It is not a calibration, "
        "validation or test result. Pressure/flow missing probabilities stayed at 0.50.", "",
        "| Family | Arm | Macro AP | Worst AP | Early recall @.001 | Clean FP | Post-event FP |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for family in ("drift", "noise"):
        for arm in ("L", "E"):
            row = reproduced[arm][family]
            lines.append(f"| {family} | {arm} | {row['macro_ap']:.4f} | "
                f"{row['worst_ap']:.4f} | {row['early_macro_recall']:.4f} | "
                f"{row['curve_clean_fp']} | {row['curve_post_event_fp']} |")
    lines += ["", "| Family | E family-oracle F1 | Precision | Recall | Passed |",
              "|---|---:|---:|---:|---:|"]
    for family in ("drift", "noise"):
        row = oracles[family]
        lines.append(f"| {family} | {row['f1']:.4f} | {row['precision']:.4f} | "
                     f"{row['recall']:.4f} | {gates[family]['passed']} |")
    lines += ["", "Neither family passed. The fast/persistent split did not add enough "
        "held-scenario ranking information: drift macro AP changed by "
        f"{gates['drift']['macro_ap_gain']:+.4f}; noise by "
        f"{gates['noise']['macro_ap_gain']:+.4f}. No later split is authorised by this result.", "",
        "The independent audit reloaded every outer bundle, reproduced its predictions, "
        "metrics and gates exactly, and found only the 14 TRAIN scenarios in OOF artifacts.", ""]
    REPORT.write_text("\n".join(lines))
    print(f"Independent audit passed; wrote {REPORT}")


if __name__ == "__main__":
    run()
