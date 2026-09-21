"""Verify saved TRAIN-only recovery predictions and publish their gate result."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from wdn.run_expert_redesign import load_arrays, read_json, sha, write_json
from wdn.marked_change import MarkedChangeFilter
from wdn.screen_recovery import (BASE, CONTROL, DATA, TrainOnlyData, expert_diagnostics,
                                 gate, local_features, reference_audit)
from wdn.train_weak_families import add_early


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/"runs/operational/drift_noise_recovery_v1"


def normal_scale_diagnostics(folds):
    """Post-screen diagnosis only; does not change scores, gates or priors."""
    names = read_json(ROOT/BASE/"feature_names.json")
    result = {}
    for index in range(len(folds)):
        row = {}
        for split in ("train", "held_out"):
            a = load_arrays(ROOT/BASE/f"fold_{index}/features_{split}.npz")
            z = a["X"][a["families"] == 0, names.index("residual")]
            row[split] = {"normal_points": len(z), "abs_z_p95": float(np.quantile(np.abs(z), .95)),
                          "fraction_abs_z_gt3": float((np.abs(z) > 3).mean())}
        a = load_arrays(RUN/f"fold_{index}/state_held_out.npz")
        state_names = read_json(RUN/f"fold_{index}/state_names.json")
        score = a["X"][a["families"] == 0, state_names.index("noise_run_probability")]
        row["raw_noise_posterior_normal_fraction_gt_half"] = float((score > .5).mean())
        result[str(index)] = row
    return result


def main():
    summary = read_json(RUN/"screening.json")
    signature = read_json(RUN/"signature.json")
    assert not any(summary[k] for k in ("calibration_evaluated", "validation_evaluated", "test_evaluated"))
    for name, expected in signature["source"].items():
        assert sha(ROOT/name) == expected, name
    for name, expected in signature["prior_signature"]["base_signature"]["data_files"].items():
        assert sha(ROOT/DATA/name) == expected, name
    for name, expected in signature["train_caches"].items():
        assert sha(ROOT/name) == expected, name
    assert sha(ROOT/CONTROL/"trial_0000/oof_predictions.npz") == signature["control_oof_sha"]
    assert sha(ROOT/DATA/"graph.pkl") == signature["graph_sha"]
    assert sha(ROOT/"data/modena.inp") == signature["inp_sha"]
    allowed = set(signature["splits"]["train"])
    events = [e for e in read_json(ROOT/DATA/"events.json") if e["scenario_id"] in allowed]
    observations = TrainOnlyData(ROOT/DATA, signature["splits"])
    parts, references, raw = [], [], []
    scores = {"physics": [], "state": []}
    for index, held in enumerate(signature["folds"]):
        folder = RUN/f"fold_{index}"
        scope = read_json(folder/"fit_scope.json")
        assert set(scope["training_scenarios"]) == allowed-set(held)
        assert set(scope["normal_scale_fitted_only_on"]) == allowed-set(held)
        physical_reference = joblib.load(folder/"physics_reference.joblib")
        normal_errors = []
        for sid in scope["training_scenarios"]:
            observed = observations.scenario(sid)
            cached = load_arrays(folder/f"scenario_{sid:02d}.npz")
            normal = observed["families"] == 0
            normal_errors.append(np.where(observed["mask"][normal],
                np.abs(observed["values"][normal]-cached["prediction"][normal]), np.nan))
        sigma = np.nanmedian(np.concatenate(normal_errors), axis=0)/.67448975
        sigma = np.maximum(sigma, max(float(np.median(sigma))*.1, 1e-6))
        np.testing.assert_array_equal(sigma, physical_reference.noise_scale_)
        # Re-extract one held scenario per fold, not merely predictions from
        # a frozen feature file. This exercises both reference and state paths.
        sid = held[0]
        observed = observations.scenario(sid)
        fresh = physical_reference.predict_details(observed["values"], observed["mask"],
                                                  observed["flow"], observed["flow_mask"])
        cached = load_arrays(folder/f"scenario_{sid:02d}.npz")
        for key, value in zip(("prediction", "support", "spread"), fresh[:3]):
            np.testing.assert_array_equal(value, cached[key])
        for key, value in fresh[3].items():
            np.testing.assert_array_equal(value, cached[key])
        state, _ = MarkedChangeFilter().transform(observed["values"], observed["mask"],
            fresh[3]["fallback_prediction"], physical_reference.baseline.noise_scale_)
        np.testing.assert_array_equal(state, cached["state"])
        features, _ = local_features(observed, *fresh[:3], physical_reference.noise_scale_)
        endpoint = observed["mask"][15:]
        stored_features = load_arrays(folder/"physics_held_out.npz")
        loc = stored_features["scenario"] == sid
        np.testing.assert_array_equal(features[endpoint], stored_features["X"][loc, :features.shape[-1]])
        for mode in scores:
            a = load_arrays(folder/f"{mode}_held_out.npz")
            assert set(a["scenario"]) == set(held)
            model = joblib.load(folder/f"{mode}_expert.joblib")
            predicted = model.predict(a["X"])
            stored = load_arrays(folder/f"{mode}_predictions.npz")["scores"]
            np.testing.assert_array_equal(predicted, stored)
            scores[mode].append(predicted)
        names = read_json(folder/"state_names.json")
        raw.append(a["X"][:, [names.index("drift_run_probability"), names.index("noise_run_probability")]])
        base = add_early(load_arrays(ROOT/BASE/f"fold_{index}/features_held_out.npz"), events)
        parts.append({k: base[k] for k in ("labels", "families", "scenario", "timestep", "node", "event", "early")})
        references.append(load_arrays(folder/"reference_audit.npz"))
    a = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    control = load_arrays(ROOT/CONTROL/"trial_0000/oof_predictions.npz")
    for k in ("labels", "families", "scenario"):
        np.testing.assert_array_equal(a[k], control[k])
    reproduced = {"control": expert_diagnostics(a, control["experts"][:, 3:5], events)}
    for mode in scores:
        joined = np.concatenate(scores[mode])
        stored = load_arrays(RUN/f"{mode}_oof_predictions.npz")
        np.testing.assert_array_equal(stored["scores"], joined)
        for key in a:
            np.testing.assert_array_equal(a[key], stored[key])
        reproduced[mode] = expert_diagnostics(a, joined, events)
    reproduced["raw_state_posterior_diagnostic"] = expert_diagnostics(a, np.concatenate(raw), events)
    assert reproduced == summary["experts"]
    reference = reference_audit(references)
    assert reference == summary["reference"]
    gates = {mode: {family: gate(reproduced[mode][family], reproduced["control"][family],
                        reference["passed"] if mode == "physics" else True)
                   for family in ("drift", "noise")} for mode in scores}
    assert gates == summary["gates"]
    survivors = [{"mechanism": mode, "family": family} for mode in gates
                 for family, row in gates[mode].items() if row["passed"]]
    assert survivors == summary["survivors"]
    previous = read_json(ROOT/CONTROL/"trial_0001/summary.json")["validation"]
    verification = {"model_reload_exact": True, "diagnostics_reproduced": True,
        "gates_reproduced": True, "source_data_and_cache_hashes_match": True,
        "held_scenario_feature_reextraction_exact": True, "train_only_scale_reproduced": True,
        "calibration_evaluated": False, "validation_evaluated": False, "test_evaluated": False}
    normal_scale = normal_scale_diagnostics(signature["folds"])
    report = {**summary, "verification": verification, "post_screen_normal_scale_diagnostic": normal_scale,
              "previous_development_validation_unchanged": previous}
    output = ROOT/"thesis_v2/outputs/drift_noise_recovery_results.json"
    write_json(output, report)
    write_json(RUN/"audit.json", verification)
    lines = ["# Drift/noise recovery: TRAIN-only component screen", "",
        "The approved two-mechanism screen was executed on the unchanged operational seed811 data.",
        "Pressure/flow missing remain 0.50. These are held-scenario TRAIN expert diagnostics, not validation/test F1.",
        "Strong experts, prior checkpoints, attack strengths, labels and outer splits were not changed.", "",
        f"Decision: **{summary['next_action']}**. Passing mechanism/family pairs: **{len(survivors)}**.",
        f"Fit calls: {summary['new_weak_expert_fit_calls']} (two heads each; {summary['new_weak_heads']} weak heads total).",
        f"Optuna trials: {summary['optuna_trials']}. Elapsed screening time: {summary['elapsed_seconds']:.1f} seconds.", "",
        "## Independent specialist discrimination", "",
        "AP is average precision. Scenario means and worst scenarios are distinct from pooled OOF AP.", "",
        "| Variant | Drift mean AP | Drift worst AP | Noise mean AP | Noise worst AP |",
        "|---|---:|---:|---:|---:|"]
    for name, row in reproduced.items():
        lines.append(f"| {name} | {row['drift']['macro_ap']:.4f} | {row['drift']['worst_ap']:.4f} | "
                     f"{row['noise']['macro_ap']:.4f} | {row['noise']['worst_ap']:.4f} |")
    lines += ["", "The raw-state posterior is an additional no-fit diagnostic, not an extra selectable candidate.",
        "The physics heads use the physical/fallback reference with the same fixed tree recipe;",
        "the state heads add the integrated onset/slope/variance evidence to the original local features.", "",
        "## Prespecified gates", "",
        "Every condition is required; no retrospective lowering of the thresholds.", "",
        "| Mechanism | Family | Mean AP gain | Passed | Failed conditions |",
        "|---|---|---:|---|---|"]
    for mode, families in gates.items():
        for family, row in families.items():
            failed = ", ".join(k for k, value in row["checks"].items() if not value) or "none"
            lines.append(f"| {mode} | {family} | {row['macro_ap_gain']:+.4f} | {row['passed']} | {failed} |")
    lines += ["", "## Reference audit", "",
        f"Physical reference gate passed: **{reference['passed']}**.",
        f"Normal points: {reference['normal_points']}; nonzero physical weight: {reference['supported_normal_points']}.",
        f"Paired supported-point MAE ratio (new/control): {reference['supported_mae_ratio']} (required <= 0.90).",
        f"All-normal p95 absolute error: {reference['base_all_normal_p95_m']:.6f} -> {reference['physics_all_normal_p95_m']:.6f} m.",
        "Topological anchor connectivity is not the same as safe physical prediction coverage.",
        "Bridge uncertainty, leave-one-measurement-out instability and agreement with the blind fallback",
        "can reject an anchored component. No target pressure enters these confidence decisions.", "",
        "## Early detection and false-alarm curve diagnostics", "",
        "Each held TRAIN scenario uses its own 0.001 all-normal FPR quantile for this diagnostic only.",
        "These are not deployable thresholds or calibration-selected F1; ties can lower the achieved FPR.",
        "Post-event FP counts cover the next 16 hours on formerly attacked pressure targets, excluding positive labels.", "",
        "| Variant | Family | Mean first-3h recall | Normal FP | Clean FP | Post-event FP |",
        "|---|---|---:|---:|---:|---:|"]
    for name in ("control", "physics", "state"):
        for family, row in reproduced[name].items():
            lines.append(f"| {name} | {family} | {row['early_macro_recall']:.4f} | {row['curve_normal_fp']} | "
                         f"{row['curve_clean_fp']} | {row['curve_post_event_fp']} |")
    lines += ["", "## Why the state posterior is unreliable: normal-scale diagnostic", "",
        "This analysis was added after the frozen screen and does not change its scores or gates.",
        "z is the original residual divided by the normal scale fitted within that fold's training scenarios.", "",
        "| Fold | Fit-scenario normal abs(z)>3 | Held-scenario normal abs(z)>3 | Fit / held abs(z) p95 | Held normal raw-noise posterior>0.5 |",
        "|---|---:|---:|---|---:|"]
    for index, row in normal_scale.items():
        fit, held = row["train"], row["held_out"]
        lines.append(f"| {index} | {100*fit['fraction_abs_z_gt3']:.2f}% | {100*held['fraction_abs_z_gt3']:.2f}% | "
                     f"{fit['abs_z_p95']:.3f} / {held['abs_z_p95']:.3f} | "
                     f"{100*row['raw_noise_posterior_normal_fraction_gt_half']:.2f}% |")
    lines += ["", "The last column is an internal, uncalibrated posterior diagnostic, not the final detector FPR.",
        "Inference: the fitted normal scale/emission distribution transfers poorly to other TRAIN scenarios.",
        "This can make ordinary reference error look like an attack and helps explain why adding onset memory alone is insufficient.",
        "The next concrete hypothesis is normal uncertainty calibration using scenario-cross-fitted residuals,",
        "optionally conditional on observable support. Merely fitting another scale on the reference's own fitting scenarios",
        "would not address the measured mismatch. This correction was not trained here and has no claimed F1 gain."]
    lines += ["", "## Distance to the requested target", "",
        "No new full stack was calibrated or evaluated on validation in this screening stage.",
        f"The last calibration-selected checkpoint's development validation remains overall F1 {previous['overall']['f1']:.4f}, "
        f"drift {previous['per_family']['stealthy']['f1']:.4f}, noise {previous['per_family']['noise']['f1']:.4f}, "
        f"replay {previous['per_family']['replay']['f1']:.4f}.",
        "These historic numbers are not performance measurements of the new components.",
        "The goal remains pointwise F1 >= 0.80 for each weak family, preserving strong families and false alarms.", "",
        "## Verification and next decision", "",
        "All six saved expert bundles reproduce the saved held-scenario predictions exactly.",
        "One held scenario per fold reproduces physical/reference/state features exactly; normal scales reproduce from only fold-training scenarios.",
        "Scenario metrics, reference audit and every gate reproduce; source, data and cache signatures match.",
        "No test, calibration or validation examples were extracted by the screen/audit."]
    if not survivors:
        lines += ["", "The approved stopping rule was applied: neither component passed all conditions,",
            "so no Optuna search, full-stack refit or new validation evaluation was started.",
            "The implementation and negative results are retained. Conditional normal uncertainty or",
            "a shared-onset extension would require a new bounded experiment decision; expanding TRAIN",
            "requires separate explicit authorisation. This failure is not proof that 0.80 is impossible."]
        write_json(RUN/"summary.json", {"status": "completed_stopped_at_screen", **report})
    else:
        lines += ["", "The planned next stage remains outstanding: nested full-stack evaluation, bounded Optuna,",
                  "and one locked candidate. Passing the screen does not establish F1 >= 0.80."]
    lines += ["", "Reproduction: `PYTHONPATH=src python thesis_v2/experiments/report_recovery_screen.py`.", ""]
    (ROOT/"thesis_v2/DRIFT_NOISE_RECOVERY_RESULTS.md").write_text("\n".join(lines))
    print(output, flush=True)


if __name__ == "__main__":
    main()
