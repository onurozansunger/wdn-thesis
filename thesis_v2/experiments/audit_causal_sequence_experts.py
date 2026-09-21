"""Independent artifact and metric audit for the bounded causal GRU screen."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import average_precision_score

from wdn.run_expert_redesign import load_arrays, read_json, sha, write_json
from wdn.screen_family_specific import _best_f1

BASE = Path("runs/operational/mechanism_redesign_v1")
FAMILY = Path("runs/operational/family_specific_experts_v1")
RUN = Path("runs/operational/causal_sequence_experts_v1")
DATA = Path("data/thesis_v2/operational_modena_seed811")
PROTOCOL = Path("thesis_v2/CAUSAL_SEQUENCE_EXPERT_PROTOCOL.md")
FUSION_PROTOCOL = Path("thesis_v2/CAUSAL_SEQUENCE_FUSION_PROTOCOL.md")


def main():
    summary, fusion = read_json(RUN/"summary.json"), read_json(RUN/"fusion_screen.json")
    saved = load_arrays(RUN/"oof_predictions.npz")
    short = load_arrays(RUN/"short/oof_raw.npz")["scores"]
    long = load_arrays(RUN/"long/oof_raw.npz")["scores"]
    expected = np.column_stack((short[:, 0], long[:, 1]))
    base = load_arrays(FAMILY/"oof_predictions.npz")
    aligned = all(np.array_equal(saved[key], base[key]) for key in
                  ("labels", "families", "event", "scenario", "timestep", "node", "early"))
    generation = yaml.safe_load((DATA/"generate_config.yaml").read_text())
    fusion_scores = load_arrays(RUN/"fusion_predictions.npz")
    fusion_metrics_exact = True
    for family_id, family in ((3, "drift"), (4, "noise")):
        family_mask = saved["families"] == family_id
        for key, result in fusion["results"].items():
            if not key.startswith(family+"_"):
                continue
            score = fusion_scores[key]
            by = {str(int(sid)): float(average_precision_score(
                saved["labels"][family_mask & (saved["scenario"] == sid)],
                score[family_mask & (saved["scenario"] == sid)]))
                for sid in np.unique(saved["scenario"][family_mask])}
            oracle = _best_f1(score[family_mask], saved["labels"][family_mask])
            fusion_metrics_exact &= (by == result["by_scenario_ap"] and
                np.mean(list(by.values())) == result["macro_ap"] and
                min(by.values()) == result["worst_ap"] and oracle == result["family_oracle_f1"])
    checks = {"selected_oof_columns_exact": np.array_equal(saved["scores"], expected),
        "endpoint_labels_and_keys_align_with_frozen_baseline": bool(aligned),
        "oof_hash_exact": sha(RUN/"oof_predictions.npz") == summary["oof_sha256"],
        "fusion_prediction_hash_exact": sha(RUN/"fusion_predictions.npz") == fusion["predictions_sha256"],
        "fusion_metrics_reproduced_exact": bool(fusion_metrics_exact),
        "protocol_hashes_exact": (sha(PROTOCOL) == summary["protocol"]["protocol_sha256"]
            and sha(FUSION_PROTOCOL) == fusion["protocol_sha256"]),
        "input_hashes_exact": (sha(FAMILY/"oof_predictions.npz") ==
            summary["protocol"]["input_hashes"]["family_oof"] and
            sha(FAMILY/"oof_predictions.npz") == fusion["input_hashes"]["baseline_oof"]),
        "pressure_and_flow_missing_probabilities_0_50":
            generation["missing_rate_pressure"] == generation["missing_rate_flow"] == .5,
        "promotion_gates_failed_and_later_splits_untouched":
            (not summary["both_families_passed"] and not fusion["both_families_passed"]
             and summary["next_action"] == fusion["next_action"] == "stop_before_calibration"
             and not any(summary[key] or fusion[key] for key in
                         ("calibration_evaluated", "validation_evaluated", "test_evaluated")))}
    audit = {**checks, "all_checks_pass": bool(all(checks.values()))}
    write_json(RUN/"independent_audit.json", audit)
    result = {"summary": summary, "fusion_screen": fusion,
        "independent_audit": audit, "all_checks_pass": audit["all_checks_pass"],
        "claim_limits": {"train_only": True, "promoted": False,
            "online_development_scores_changed": False, "calibration_evaluated": False,
            "validation_evaluated": False, "test_evaluated": False}}
    write_json(Path("thesis_v2/outputs/causal_sequence_experts_results.json"), result)
    if not audit["all_checks_pass"]:
        raise AssertionError(json.dumps(checks, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
