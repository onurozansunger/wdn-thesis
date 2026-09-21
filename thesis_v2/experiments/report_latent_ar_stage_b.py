"""Independently reproduce and report the completed TRAIN-only latent AR screen."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wdn.run_expert_redesign import load_arrays, read_json, sha, write_json
from wdn.screen_latent_ar import (_lag_one, _normal_diagnostics, _normal_gate,
                                  CONTROL, DATA, STAGE_A)
from wdn.screen_normal_nuisance import concatenate, secondary_curve
from wdn.screen_recovery import expert_diagnostics, gate


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/"runs/operational/latent_ar_stage_b_v1"


def main():
    summary, signature = read_json(RUN/"summary.json"), read_json(RUN/"signature.json")
    for path, expected in signature["sources"].items():
        assert sha(ROOT/path) == expected, path
    assert sha(ROOT/STAGE_A/"signature.json") == signature["stage_a_signature_sha"]
    assert sha(ROOT/STAGE_A/"summary.json") == signature["stage_a_summary_sha"]
    assert sha(ROOT/STAGE_A/"independent_audit.json") == signature["stage_a_audit_sha"]
    assert sha(ROOT/STAGE_A/"oof_predictions.npz") == signature["stage_a_oof_sha"]
    assert summary["accessed_scenarios"] == sorted(signature["splits"]["train"])
    assert not any(summary[key] for key in
                   ("calibration_evaluated", "validation_evaluated", "test_evaluated"))
    assert summary["survivors"] == []
    assert summary["next_action"] == "stop_no_latent_ar_candidate_passed"
    assert sha(RUN/"oof_predictions.npz") == summary["artifact_manifest"]["oof_predictions_sha256"]
    outer_parts = []
    for outer, held in enumerate(signature["folds"]):
        folder = RUN/f"outer_{outer}"
        record = read_json(folder/"completed.json")
        scope = read_json(folder/"fit_scope.json")
        assert record == summary["artifact_manifest"]["outer"][str(outer)]
        assert record["held_scenarios"] == held == scope["held_scenarios"]
        assert scope["fit_scenarios"] == sorted(set(signature["splits"]["train"])-set(held))
        assert scope["family_zero_only"] is True
        assert sha(folder/"fit_scope.json") == record["fit_scope_sha256"]
        assert sha(folder/"latent_ar.joblib") == record["model_sha256"]
        assert sha(folder/"held_predictions.npz") == record["predictions_sha256"]
        assert scope["phi"] == summary["fitted_models"][outer]["phi"]
        assert scope["consecutive_pairs"] == summary["fitted_models"][outer]["fit_pair_count"]
        assert len(scope["fit_directed_caches"]) == 2
        predicted = []
        for path_string in scope["fit_directed_caches"]:
            path = ROOT/path_string
            directed_record = read_json(path.parent/"completed.json")
            assert sha(path) == directed_record["rows_sha256"]
            assert not set(directed_record["fit_scenarios"]) & set(directed_record["predicted_scenarios"])
            predicted.extend(directed_record["predicted_scenarios"])
        assert sorted(predicted) == scope["fit_scenarios"]
        outer_parts.append(load_arrays(folder/"held_predictions.npz"))

    arrays = load_arrays(RUN/"oof_predictions.npz")
    joined = concatenate(outer_parts)
    assert set(joined) == set(arrays)
    for key in joined:
        np.testing.assert_array_equal(joined[key], arrays[key])
    scores = {arm: arrays.pop(f"scores_{arm}") for arm in ("B", "A")}
    stage_a = load_arrays(ROOT/STAGE_A/"oof_predictions.npz")
    for key in ("labels", "families", "event", "timestep", "node", "early",
                "scenario", "outer_fold", "error", "mean_B", "scale_B"):
        np.testing.assert_array_equal(arrays[key], stage_a[key])
    np.testing.assert_array_equal(scores["B"], stage_a["scores_B"])
    control = load_arrays(ROOT/CONTROL/"trial_0000/oof_predictions.npz")
    for key in ("labels", "families", "scenario"):
        np.testing.assert_array_equal(arrays[key], control[key])
    all_events = read_json(ROOT/DATA/"events.json")
    train = set(signature["splits"]["train"])
    events = [{**event, "_event_id": event_id} for event_id, event in enumerate(all_events)
              if event["scenario_id"] in train]

    normal = _normal_diagnostics(arrays, events)
    lag = {arm: _lag_one(arrays, arm) for arm in ("B", "A")}
    normal_decision = _normal_gate(normal, lag)
    diagnostics = {arm: expert_diagnostics(arrays, value, events)
                   for arm, value in scores.items()}
    diagnostics["local_control"] = expert_diagnostics(arrays, control["experts"][:, 3:5], events)
    gates, survivors = {}, []
    for family in ("drift", "noise"):
        decision = gate(diagnostics["A"][family], diagnostics["B"][family], True)
        fraction = diagnostics["A"][family]["macro_ap"]/diagnostics["local_control"][family]["macro_ap"]
        decision["checks"]["minimum_control_macro_ap_fraction"] = fraction >= .75
        decision["checks"]["global_normal_gate"] = normal_decision["passed"]
        decision["passed"] = all(decision["checks"].values())
        decision["control_macro_ap_fraction"] = fraction
        decision["comparison"] = "latent AR A versus coherent iid B; global normal gate required"
        gates[family] = decision
        if decision["passed"]:
            survivors.append(family)
    secondary = {arm: secondary_curve(arrays, value, events, .005)
                 for arm, value in scores.items()}
    assert normal == summary["normal_diagnostics"]
    assert lag == summary["lag_one_diagnostics"]
    assert normal_decision == summary["normal_gate"]
    assert diagnostics == summary["evidence_diagnostics"]
    assert gates == summary["evidence_gates"]
    assert secondary == summary["secondary_fpr_005"]
    assert survivors == summary["survivors"]

    verification = {"saved_oof_metrics_reproduced_exact": True,
        "normal_gate_reproduced_exact": True, "evidence_gates_reproduced_exact": True,
        "lag_diagnostics_reproduced_exact": True, "secondary_curve_reproduced_exact": True,
        "artifact_hashes_match": True, "stage_a_metadata_and_baseline_exact": True,
        "accessed_train_scenarios_only": True,
        "oof_predictions_sha256": sha(RUN/"oof_predictions.npz"),
        "calibration_evaluated": False, "validation_evaluated": False,
        "test_evaluated": False}
    write_json(RUN/"independent_audit.json", verification)
    write_json(ROOT/"thesis_v2/outputs/latent_ar_stage_b_results.json",
               {"summary": summary, "independent_audit": verification})

    bnormal = normal["B"]["clean"]["scenario_macro"]
    anormal = normal["A"]["clean"]["scenario_macro"]
    lines = ["# Latent AR Stage B: TRAIN-only results", "",
        "The separate causal AR screen completed on the unchanged operational seed811 data.",
        "Pressure/flow missing remain 0.50. Calibration, validation and test were not extracted or evaluated.",
        "These are TRAIN nested-OOF component diagnostics, not new F1 measurements.", "",
        f"Decision: **{summary['next_action']}**. Surviving families: **{len(survivors)}**.",
        f"Runtime: {summary['elapsed_seconds']:.1f} seconds; AR fits: 3; Optuna trials: 0; weak-head fits: 0.",
        "Outer AR coefficients were " + ", ".join(f"{row['phi']:.3f}" for row in summary["fitted_models"]) + ".", "",
        "## Normal prediction", "",
        "| Arm | Macro MAE (m) | Macro NLL | abs(z)>3 | 95% width (m) | lag-1 r |",
        "|---|---:|---:|---:|---:|---:|",
        f"| B | {bnormal['mae_m']:.4f} | {bnormal['nll']:.4f} | {bnormal['fraction_abs_z_gt3']:.2%} | "
        f"{bnormal['mean_width95_m']:.4f} | {lag['B']['pooled_pearson_r']:.3f} |",
        f"| A | {anormal['mae_m']:.4f} | {anormal['nll']:.4f} | {anormal['fraction_abs_z_gt3']:.2%} | "
        f"{anormal['mean_width95_m']:.4f} | {lag['A']['pooled_pearson_r']:.3f} |", "",
        f"A/B MAE ratio was **{normal_decision['macro_mae_ratio']:.4f}** and all "
        f"**{normal_decision['improved_scenarios']}/14** scenarios improved.",
        "NLL, tail rate and serial dependence improved in every outer fold.",
        f"The normal gate still failed because mean interval width was **{normal_decision['macro_mean_width95_ratio']:.3f}x** B "
        "(frozen maximum: 1.25x).", "",
        "## Raw mechanism evidence", "",
        "| Arm | Drift macro AP | Drift early recall | Clean/post FP | Noise macro AP | Noise early recall | Clean/post FP |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ("B", "A", "local_control"):
        row = diagnostics[arm]
        lines.append(f"| {arm} | {row['drift']['macro_ap']:.4f} | {row['drift']['early_macro_recall']:.4f} | "
                     f"{row['drift']['curve_clean_fp']}/{row['drift']['curve_post_event_fp']} | "
                     f"{row['noise']['macro_ap']:.4f} | {row['noise']['early_macro_recall']:.4f} | "
                     f"{row['noise']['curve_clean_fp']}/{row['noise']['curve_post_event_fp']} |")
    lines += ["",
        f"Drift AP increased by **{gates['drift']['macro_ap_gain']:+.4f}** and improved in all four drift scenarios,",
        "but early recall remained zero, clean/post-event false positives increased, and A reached only "
        f"{gates['drift']['control_macro_ap_fraction']:.1%} of local-control AP.",
        f"Noise AP increased by only **{gates['noise']['macro_ap_gain']:+.4f}**; it reached "
        f"{gates['noise']['control_macro_ap_fraction']:.1%} of local-control AP and added two post-event false positives.",
        "Neither evidence gate passed. The frozen stop rule therefore starts no specialist, fusion, Optuna,",
        "calibration or validation run. This does not show that drift/noise F1 0.80 is impossible.", "",
        "## Interpretation", "",
        "The serial-error hypothesis is real: AR conditioning materially improved normal prediction and drift ranking.",
        "The remaining problem is not another static mean. It is calibrated branch-specific temporal inference:",
        "drift onset evidence is still late, while the approximate noise innovation branch remains too weak.",
        "Any exact switching/Kalman or early-onset extension must be a separately frozen experiment.", "",
        "Independent audit recomputed every metric and gate exactly from saved OOF arrays and verified all hashes/scopes."]
    (ROOT/"thesis_v2/LATENT_AR_STAGE_B_RESULTS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
