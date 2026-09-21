"""Verify and compare completed redesign artifacts; never train a model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import optuna

from wdn.operational_calibration import calibrate_threshold, family_summary
from wdn.run_expert_redesign import independent_audit, operating_point, score_report
from wdn.train_operational_moe import summarise


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/"runs/operational/mechanism_redesign_v1"
FAMILIES = ("random", "replay", "stealthy", "noise", "targeted")


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2))


def event_metrics(arrays, score, threshold, events, scenarios):
    decision = score > threshold
    rows = []
    for event in events:
        sid, start = event["scenario_id"], event["start_timestep"]
        if sid not in scenarios:
            continue
        active = (arrays["scenario"] == sid) & (arrays["timestep"] >= start) & (arrays["timestep"] < start+event["actual_steps"])
        targeted = np.isin(arrays["node"], event["targets"]["pressure"])
        positive = active & (arrays["labels"] > 0)
        hits = positive & decision
        delay = []
        for node in event["targets"]["pressure"]:
            detected = hits & (arrays["node"] == node)
            delay.append(int(arrays["timestep"][detected].min()-start) if detected.any() else None)
        early = positive & (arrays["timestep"] < start+3)
        after = (arrays["scenario"] == sid) & (arrays["timestep"] >= start+event["actual_steps"]) & (arrays["timestep"] < start+event["actual_steps"]+16) & (arrays["labels"] == 0)
        rows.append({"scenario": sid, "family": event["family"], "start": start,
            "tp": int(hits.sum()), "positives": int(positive.sum()),
            "early_3h_tp": int((early & decision).sum()), "early_3h_positives": int(early.sum()),
            "fp_unattacked_during_event": int((active & ~targeted & decision).sum()),
            "sensors_detected_once": sum(d is not None for d in delay), "target_sensors": len(delay),
            "first_detection_delay_by_sensor_h": delay,
            "post_16h_fp_on_formerly_targeted_sensors": int((after & targeted & decision).sum())})
    return rows


def main():
    summary = read(RUN/"summary.json")
    frozen = read(RUN/"selection_frozen.json")
    assert summary["status"] == "completed" and summary["test_evaluated"] is False
    assert frozen == summary["selection"]
    for name, expected in summary["source_and_inputs"]["source"].items():
        assert hashlib.sha256((ROOT/"src/wdn"/name).read_bytes()).hexdigest() == expected
    for name, expected in summary["source_and_inputs"]["data_files"].items():
        assert hashlib.sha256((ROOT/"data/thesis_v2/operational_modena_seed811"/name).read_bytes()).hexdigest() == expected
    splits = summary["source_and_inputs"]["splits"]
    assert sorted(s for fold in summary["inner_folds"] for s in fold) == sorted(splits["train"])
    cal = dict(np.load(RUN/"full/features_calibration.npz"))
    selected_folder = RUN/f"trial_{frozen['trial']:04d}"
    cal_selected = dict(np.load(selected_folder/"predictions_calibration.npz"))
    overall_point = calibrate_threshold(cal_selected["mixture"], cal["labels"], cal["families"],
        objective="legacy", max_fpr=.005, min_replay_f1=.5)
    # Fixed additional diagnostic objective, specified before validation; it
    # does not reselect the trial or replace the primary operating point.
    write(RUN/"diagnostic_overall_calibration.json", overall_point)
    val = dict(np.load(RUN/"full/features_validation.npz"))
    for split, a in (("calibration", cal), ("validation", val)):
        base = dict(np.load(ROOT/f"runs/operational/blind_reference_probe_rank16/features_{split}.npz"))
        np.testing.assert_array_equal(a["labels"], base["labels"])
        np.testing.assert_array_equal(a["families"], base["families"])
        assert set(a["scenario"]) == set(splits[split])
    best = max(summary["trials"].values(), key=lambda row: tuple(row["selection"]["calibration"][key]
               for key in ("worst_family_f1", "macro_f1", "overall_f1")))
    assert best["trial"] == frozen["trial"]
    assert len(summary["trials"]) <= 4
    study = optuna.load_study(study_name="mechanism_redesign_v1", storage=f"sqlite:///{RUN/'study.sqlite3'}")
    trial_states = [{"number": t.number, "state": t.state.name, "params": t.params,
                     "infeasible": t.user_attrs.get("infeasible")} for t in study.trials]
    assert len(trial_states) <= 4
    bundle = joblib.load(selected_folder/"model.joblib")
    selected_scores = {}
    for split, a in (("calibration", cal), ("validation", val)):
        stored = dict(np.load(selected_folder/f"predictions_{split}.npz"))
        fresh = bundle["experts"].predict(a["X"])
        np.testing.assert_array_equal(fresh, stored["experts"])
        np.testing.assert_array_equal(bundle["fusion"].predict(fresh, a["X"]), stored["mixture"])
        selected_scores[split] = stored
    assert operating_point(selected_scores["calibration"]["mixture"], cal) == best["selection"]
    result = score_report(selected_scores["validation"]["mixture"], val, best["selection"])
    assert all(result[key] == best[key] for key in result)
    audit = independent_audit(selected_scores["calibration"]["experts"], selected_scores["validation"]["experts"], cal, val)
    assert audit == read(selected_folder/"expert_audit.json")

    previous = read(ROOT/"runs/operational/blind_residual_experts_v1/summary.json")["results"]["mixture"]
    dynamic = read(ROOT/"runs/operational/family_balance_v1/summary.json")
    rows = [{"name": "Previous blind-reference mixture", "validation": previous, **family_summary(previous)},
            {"name": "Previous dynamic, overall priority", **dynamic["dynamic"]["mixture_legacy"]},
            {"name": "Previous dynamic, family balance", **dynamic["dynamic"]["mixture_macro_f1"]}]
    # Fixed old estimator/thresholds, changing only the reference-derived input.
    # This is an input intervention control, not another fitted search trial.
    old_model = joblib.load(ROOT/"runs/operational/family_balance_v1/model.joblib")
    old_on_new = old_model.predict(val["X"][:, :60])["mixture"]
    for method in ("mixture_legacy", "mixture_macro_f1"):
        threshold = dynamic["dynamic"][method]["selection"]["threshold"]
        r = summarise(old_on_new, val["labels"], val["families"], threshold)
        rows.append({"name": f"Reference-only, fixed old {method} estimator/threshold", "validation": r, **family_summary(r)})
    for number, trial in summary["trials"].items():
        rows.append({"name": f"Trial {number}: {trial['kind']}"+(" [CALIBRATION SELECTED]" if int(number) == frozen["trial"] else ""), **trial})
    rows.append({"name": "Selected model, diagnostic overall-priority point", "trial": frozen["trial"],
                 **score_report(selected_scores["validation"]["mixture"], val, overall_point)})
    events = read(ROOT/"data/thesis_v2/operational_modena_seed811/events.json")
    timing = event_metrics(val, selected_scores["validation"]["mixture"], best["selection"]["threshold"], events, splits["validation"])
    checks = {"test_evaluated": False, "data_hashes_unchanged": True, "all_source_hashes_match": True,
              "selected_model_reload_exact": True, "calibration_choice_reproduced": True,
              "validation_labels_endpoints_match": True, "selected_expert_audit_reproduced": True}
    report = {"selection": frozen, "scope": summary["scope"], "verification": checks,
              "comparisons": rows, "selected_expert_audit": audit, "selected_event_metrics": timing,
              "normal_reference_mae_m": summary["normal_reference_mae_m"], "trial_states": trial_states}
    write(RUN/"verification.json", checks)
    write(ROOT/"thesis_v2/outputs/expert_redesign_results.json", report)

    lines = ["# Expert redesign: frozen-data development results", "",
        "Pressure-only detection, not the original temporal GNN. Same data/splits and pressure/flow missing probabilities 0.50.",
        "This is a heterogeneous expert stack. Its logistic combiner is not a normalised, input-dependent MoE gate; neural/gated MoE integration is not claimed.",
        "No locked test evaluation. Previous validation findings informed the design; these are not independent generalisation results.", "",
        "## Method and selection", "",
        "Noise-scaled, leverage-adjusted robust fitting on three sensor subsets; complete target-group exclusion in every computation.",
        "Causal drift/ramp likelihood and variance-state evidence; no oracle targets, future smoothing or point adjustment.",
        "The likelihood assumptions are approximate and balanced-risk scores are not calibrated attack probabilities.",
        "Three inner scenario folds refit the normal reference. Their held-out expert scores train the logistic combiner.",
        "Expert risk is balanced across families/events and normal scenarios. The new tree control shares this recipe; it is not an isolated reference-only ablation.",
        "Trial 0 uses tree experts; trials 1–3 use logistic mechanism readouts for drift/noise and trees for the other experts.",
        "Optuna tunes regularisation in this small, four-trial campaign. Its best parameters are not claimed to be globally optimal.",
        "The old four-trial GNN study was neither extended nor restarted.", "",
        f"**Calibration-selected trial: {frozen['trial']} ({frozen['kind']}); parameters: {frozen['params']}.**",
        "Selection maximises the worst calibration family F1, with macro/overall F1 tie breaks; normal/clean FPR <= 0.005 and replay F1 >= 0.50.",
        "The choice and threshold were saved before validation extraction/scoring. Other trials below are disclosed, not substituted as winners based on validation.", "",
        "For the selected model, a second calibration-only overall/replay operating point is also reported as a diagnostic; it does not replace the primary worst-family point.", "",
        "## Validation comparison", "",
        "| Variant | Overall F1 | Random | Replay | Drift | Noise | Targeted | Worst family F1 | FP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        v = row["validation"]
        lines.append(f"| {row['name']} | {v['overall']['f1']:.4f} | "+" | ".join(f"{v['per_family'][f]['f1']:.4f}" for f in FAMILIES)
                     +f" | {row['worst_family_f1']:.4f} | {v['overall']['fp']} |")
    lines += ["", "| Trial | Calibration worst F1 | Validation worst F1 | Validation AUPRC | Sensor FPR (%) | Clean FPR (%) |",
              "|---|---:|---:|---:|---:|---:|"]
    for number, row in summary["trials"].items():
        v = row["validation"]
        lines.append(f"| {number} | {row['selection']['calibration']['worst_family_f1']:.4f} | {row['worst_family_f1']:.4f} | "
                     f"{v['overall']['auprc']:.4f} | {100*v['overall']['fpr']:.4f} | {100*v['per_family']['clean']['fpr']:.4f} |")
    lines += ["", "## Selected model: independent expert AUPRC", "",
              "One global calibration-F1 threshold per expert is used for the clean FPR column; it is not a per-family oracle threshold.", "",
              "| Expert | Random | Replay | Drift | Noise | Targeted | Clean FPR (%) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in audit["expert_matrix"].items():
        lines.append(f"| {name} | "+" | ".join(f"{row['per_family'][f]['auprc']:.4f}" for f in FAMILIES)
                     +f" | {100*row['per_family']['clean']['fpr']:.4f} |")
    lines += ["", "## Selected model: event and early-warning diagnostics", "",
              "| Scenario / family | TP / positives | First 3h TP / positives | Sensors detected / targets | FP on unattacked sensors during event | FP on former targets in next 16h |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in timing:
        lines.append(f"| {row['scenario']} / {row['family']} | {row['tp']} / {row['positives']} | {row['early_3h_tp']} / {row['early_3h_positives']} | "
                     f"{row['sensors_detected_once']} / {row['target_sensors']} | {row['fp_unattacked_during_event']} | {row['post_16h_fp_on_formerly_targeted_sensors']} |")
    v = best["validation"]
    old_balanced = dynamic["dynamic"]["mixture_macro_f1"]["validation"]
    old_experts = read(ROOT/"runs/operational/family_balance_v1/expert_audit.json")["expert_matrix"]
    lines += ["", "## Decision: not an all-family replacement", "",
        f"The selected point changes overall F1 from {old_balanced['overall']['f1']:.4f} to {v['overall']['f1']:.4f} and FP from {old_balanced['overall']['fp']} to {v['overall']['fp']} versus the previous balanced point.",
        f"However, noise F1 falls from {old_balanced['per_family']['noise']['f1']:.4f} to {v['per_family']['noise']['f1']:.4f}; the requested all-family improvement is not achieved.",
        f"Independent drift-expert AUPRC changes from {old_experts['drift']['per_family']['stealthy']['auprc']:.4f} to {audit['expert_matrix']['drift']['per_family']['stealthy']['auprc']:.4f}.",
        f"Independent noise-expert AUPRC changes from {old_experts['noise']['per_family']['noise']['auprc']:.4f} to {audit['expert_matrix']['noise']['per_family']['noise']['auprc']:.4f}; the new noise readout is not an improvement.",
        "The robust reference is a useful component candidate, supported by the fixed-old-estimator controls. It does not justify promoting the complete new stack.",
        "All old models are preserved. No default production or thesis-final model was replaced. The four-trial campaign is complete; no further trial is silently added."]
    lines += ["", "## Verification and limits", "",
        "Original raw-data hashes and trained-source hashes match. Reloaded selected-model predictions, calibration selection and independent expert audit reproduce exactly.",
        "All validation labels/family order match the original frozen benchmark. The internal scenario folds cover TRAIN once without overlap.",
        f"Remaining selected-model overall-F1 gap to 0.90: {max(0., .90-v['overall']['f1']):.4f}.",
        "Only one drift and one noise validation event are available. An event alarm is not equivalent to detecting each corrupted reading.",
        "The FPR constraint is a calibration constraint, not a field guarantee. No claim of all-family success is made unless the table supports it.",
        "Reference-only controls reuse the old estimator and threshold; input-distribution changes may make them suboptimal.",
        "The previous oracle target-exclusion diagnostic is not used anywhere in the trained pipeline.", ""]
    (ROOT/"thesis_v2/EXPERT_REDESIGN_RESULTS.md").write_text("\n".join(lines))
    print(json.dumps({"selected_trial": frozen["trial"], "validation": v, "verification": checks}, indent=2))


if __name__ == "__main__":
    main()
