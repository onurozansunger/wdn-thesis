"""Independently reproduce and report the completed TRAIN-only Stage A screen."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wdn.run_expert_redesign import load_arrays, read_json, sha, write_json
from wdn.screen_normal_nuisance import (CONTROL, DATA, normal_diagnostics, normal_gate,
                                         secondary_curve)
from wdn.screen_recovery import expert_diagnostics, gate


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/"runs/operational/normal_nuisance_stage_a_v2"


def posthoc_residual_dependence(arrays):
    """Describe remaining TRAIN OOF dependence without fitting or changing a gate."""
    clean = arrays["families"] == 0
    lag_one = {}
    for arm in ("B", "M"):
        residual = arrays["error"] - arrays[f"mean_{arm}"]
        pairs = []
        for scenario in np.unique(arrays["scenario"][clean]):
            scenario_clean = clean & (arrays["scenario"] == scenario)
            for node in np.unique(arrays["node"][scenario_clean]):
                index = np.flatnonzero(scenario_clean & (arrays["node"] == node))
                index = index[np.argsort(arrays["timestep"][index])]
                adjacent = np.diff(arrays["timestep"][index]) == 1
                if np.any(adjacent):
                    pairs.append(np.column_stack((residual[index[:-1][adjacent]],
                                                  residual[index[1:][adjacent]])))
        pooled = np.concatenate(pairs)
        lag_one[arm] = {"pairs": int(len(pooled)),
                        "pooled_pearson_r": float(np.corrcoef(pooled.T)[0, 1])}

    residual = arrays["error"] - arrays["mean_M"]
    peer = np.full(len(residual), np.nan)
    for scenario in np.unique(arrays["scenario"][clean]):
        scenario_clean = clean & (arrays["scenario"] == scenario)
        for timestep in np.unique(arrays["timestep"][scenario_clean]):
            index = np.flatnonzero(scenario_clean & (arrays["timestep"] == timestep))
            group = arrays["node"][index] % 4
            for target_group in np.unique(group):
                outside = index[group != target_group]
                if len(outside):
                    peer[index[group == target_group]] = residual[outside].mean()
    valid = clean & np.isfinite(peer)
    return {"scope": "post-hoc descriptive TRAIN OOF diagnostic; not a gate or fitted feature",
        "clean_definition": "families == 0 (clean episodes)",
        "same_sensor_consecutive_hour_residual": lag_one,
        "conditional_M_external_node_group_peer_mean": {
            "points": int(valid.sum()),
            "pooled_pearson_r": float(np.corrcoef(residual[valid], peer[valid])[0, 1]),
            "group_definition": "node modulo 4; target node group excluded"}}


def main():
    summary, signature = read_json(RUN/"summary.json"), read_json(RUN/"signature.json")
    for path, expected in signature["sources"].items():
        assert sha(ROOT/path) == expected, path
    assert not any(summary[key] for key in
                   ("calibration_evaluated", "validation_evaluated", "test_evaluated"))
    assert summary["accessed_scenarios"] == sorted(signature["splits"]["train"])
    assert summary["survivors"] == []
    assert summary["next_action"] == "stop_no_normal_nuisance_candidate_passed"

    for index, fold in enumerate(signature["folds"]):
        shared = RUN/f"shared_references/train_F{index}"
        record = read_json(shared/"completed.json")
        assert sha(shared/"reference.joblib") == record["model_sha256"]
        assert sha(shared/"fit_scope.json") == record["fit_scope_sha256"]
        assert record["fit_scenarios"] == fold
        outer = RUN/f"outer_{index}"
        record = read_json(outer/"completed.json")
        assert record["held_scenarios"] == fold
        assert sha(outer/"held_predictions.npz") == record["predictions_sha256"]
        assert sha(outer/"conditional_mean.joblib") == record["mean_model_sha256"]
        assert sha(outer/"scale_control.joblib") == record["scale_model_sha256"]
        assert sha(outer/"fit_scope.json") == record["fit_scope_sha256"]
    for train in range(3):
        for held in range(3):
            if train == held:
                continue
            folder = RUN/f"directed_predictions/train_F{train}__held_F{held}"
            record = read_json(folder/"completed.json")
            assert sha(folder/"normal_rows.npz") == record["rows_sha256"]
            assert record["fit_scenarios"] == signature["folds"][train]
            assert record["predicted_scenarios"] == signature["folds"][held]
            assert not set(record["fit_scenarios"]) & set(record["predicted_scenarios"])

    arrays = load_arrays(RUN/"oof_predictions.npz")
    scores = {arm: arrays.pop(f"scores_{arm}") for arm in ("C", "B", "S", "M")}
    control = load_arrays(ROOT/CONTROL/"trial_0000/oof_predictions.npz")
    for key in ("labels", "families", "scenario"):
        np.testing.assert_array_equal(arrays[key], control[key])
    all_events = read_json(ROOT/DATA/"events.json")
    train = set(signature["splits"]["train"])
    events = [{**event, "_event_id": event_id} for event_id, event in enumerate(all_events)
              if event["scenario_id"] in train]

    normal = normal_diagnostics(arrays, events)
    decision = normal_gate(normal)
    diagnostics = {arm: expert_diagnostics(arrays, value, events)
                   for arm, value in scores.items()}
    diagnostics["local_control"] = expert_diagnostics(
        arrays, control["experts"][:, 3:5], events)
    gates = {}
    survivors = []
    for family in ("drift", "noise"):
        row = gate(diagnostics["M"][family], diagnostics["B"][family], True)
        fraction = diagnostics["M"][family]["macro_ap"]/diagnostics["local_control"][family]["macro_ap"]
        row["checks"]["minimum_control_macro_ap_fraction"] = fraction >= .75
        row["checks"]["global_normal_gate"] = decision["passed"]
        row["passed"] = all(row["checks"].values())
        row["control_macro_ap_fraction"] = fraction
        row["comparison"] = "M versus coherent B; global normal gate also required"
        gates[family] = row
        if row["passed"]:
            survivors.append(family)
    secondary = {arm: secondary_curve(arrays, value, events, .005)
                 for arm, value in scores.items()}
    assert normal == summary["normal_diagnostics"]
    assert decision == summary["normal_gate"]
    assert diagnostics == summary["evidence_diagnostics"]
    assert gates == summary["evidence_gates"]
    assert secondary == summary["secondary_fpr_005"]
    assert survivors == summary["survivors"]
    posthoc = posthoc_residual_dependence(arrays)

    verification = {"saved_oof_metrics_reproduced_exact": True,
        "normal_gate_reproduced_exact": True, "evidence_gates_reproduced_exact": True,
        "secondary_curve_reproduced_exact": True, "artifact_hashes_match": True,
        "metadata_matches_frozen_control": True, "accessed_train_scenarios_only": True,
        "oof_predictions_sha256": sha(RUN/"oof_predictions.npz"),
        "runner_resume_caveats": ["Completed summary short-circuits before rechecking child artifacts; this audit rechecks them.",
            "The runner CLI default names the preserved failed v1; explicit v2 is required for completed-run replay."],
        "calibration_evaluated": False, "validation_evaluated": False,
        "test_evaluated": False}
    write_json(RUN/"independent_audit.json", verification)
    write_json(ROOT/"thesis_v2/outputs/normal_nuisance_stage_a_results.json",
               {"summary": summary, "independent_audit": verification,
                "posthoc_diagnostics": posthoc})

    lines = ["# Normal nuisance Stage A: TRAIN-only results", "",
        "The scenario-nested screen completed on the unchanged operational seed811 data.",
        "Pressure/flow missing remain 0.50. Calibration, validation and test were not extracted or evaluated.",
        "These are TRAIN development diagnostics, not final F1 measurements.", "",
        f"Decision: **{summary['next_action']}**. Surviving families: **{len(survivors)}**.",
        f"Completed v2 runtime: {summary['elapsed_seconds']:.1f} seconds. Optuna trials: 0; weak-head fits: 0.",
        "Three unique reference fits support six directed inner-OOF prediction scopes; six nuisance bundles were fitted.",
        "A preserved v1 attempt stopped on a bookkeeping error before any outer fold completed; v2 restarted with a signed fix.", "",
        "## Normal prediction", "",
        "| Arm | Macro MAE (m) | Macro NLL | abs(z)>3 | 95% interval width (m) |",
        "|---|---:|---:|---:|---:|"]
    for arm in ("B", "S", "M"):
        row = normal[arm]["clean"]["scenario_macro"]
        lines.append(f"| {arm} | {row['mae_m']:.4f} | {row['nll']:.4f} | "
                     f"{row['fraction_abs_z_gt3']:.2%} | {row['mean_width95_m']:.4f} |")
    lines += ["", f"The conditional mean M/B macro-MAE ratio was **{decision['macro_mae_ratio']:.4f}**; "
        f"only **{decision['improved_scenarios']}/14** scenarios improved (required: ratio <=0.90 and 10/14).",
        "Scale control S greatly improved NLL/tail coverage by widening the normal interval, but it did not change MAE.",
        "M was slightly worse than S on MAE, NLL and tail coverage. The global normal gate failed.", "",
        "## Raw mechanism evidence", "",
        "| Arm | Drift macro AP | Drift early recall | Noise macro AP | Noise early recall |",
        "|---|---:|---:|---:|---:|"]
    for arm in ("C", "B", "S", "M", "local_control"):
        row = diagnostics[arm]
        lines.append(f"| {arm} | {row['drift']['macro_ap']:.4f} | {row['drift']['early_macro_recall']:.4f} | "
                     f"{row['noise']['macro_ap']:.4f} | {row['noise']['early_macro_recall']:.4f} |")
    lines += ["", "C is the historical marked posterior; B is the coherent likelihood with the frozen old scale;",
        "S changes only cross-fitted scale; M adds the fixed conditional mean. The frozen local tree is context, not a new fit.",
        f"M versus B macro AP changed by {gates['drift']['macro_ap_gain']:+.4f} for drift and "
        f"{gates['noise']['macro_ap_gain']:+.4f} for noise. Neither family passed its evidence gate.",
        "Drift first-three-hour recall remained zero in B/S/M at the .001 diagnostic normal-FPR curve.", "",
        "## Interpretation and stop", "",
        "The experiment confirms that the old fitted scale was overconfident across scenarios. It does not show useful",
        "predictable mean correction from the allowed covariates, and scale correction did not improve weak-attack ranking.",
        "Therefore the frozen protocol stops: no conditional scale/AR extension, weak head, fusion, Optuna, physical Stage B,",
        "calibration or validation run is authorised by this result. This is not proof that drift/noise F1 0.80 is impossible.", "",
        "## Post-hoc direction for a separate experiment", "",
        f"Clean same-sensor consecutive-hour residual correlation was **{posthoc['same_sensor_consecutive_hour_residual']['B']['pooled_pearson_r']:.3f}** "
        f"for B and **{posthoc['same_sensor_consecutive_hour_residual']['M']['pooled_pearson_r']:.3f}** for M "
        f"over {posthoc['same_sensor_consecutive_hour_residual']['B']['pairs']:,} pairs.",
        f"The simultaneous external node-group peer mean had correlation only "
        f"**{posthoc['conditional_M_external_node_group_peer_mean']['pooled_pearson_r']:.3f}** with M residuals.",
        "This post-hoc result was not a gate or fitted feature. It supports a separately frozen branch-aware causal latent-AR",
        "normal-error experiment more directly than another static mean/scale or naive peer-pooling model.", "",
        "Independent audit reproduced every saved metric and gate exactly and verified artifact hashes/split metadata.",
        "The combined pickle archives were deserialised by the existing loader, but only TRAIN scenarios were extracted or used."]
    (ROOT/"thesis_v2/NORMAL_NUISANCE_STAGE_A_RESULTS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
