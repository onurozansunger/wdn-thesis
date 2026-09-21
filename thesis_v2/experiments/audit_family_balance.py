"""Reproduce frozen family probes and report every comparison; no model fitting."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.metrics import average_precision_score

from wdn.audit_operational_experts import best_f1_threshold
from wdn.operational_calibration import calibrate_threshold, family_summary
from wdn.operational_data import OperationalConfig
from wdn.probe_expert_calibration import NormalTailEvidence
from wdn.probe_residual_experts import MECHANISMS
from wdn.train_operational_moe import FAMILY_NAMES, _binary_counts, select_threshold, summarise


ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ("random", "replay", "stealthy", "noise", "targeted")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    reference = ROOT/"runs/operational/blind_reference_probe_rank16"
    original = ROOT/"runs/operational/blind_residual_experts_v1"
    dynamic = ROOT/"runs/operational/family_balance_v1"
    fusion = ROOT/"runs/operational/family_calibration_v1"
    old, new, tuned = (read(p/"summary.json") for p in (original, dynamic, fusion))
    if any(r["test_evaluated"] for r in (old, new, tuned)):
        raise ValueError("Unexpected test evaluation")
    config_path = ROOT/"configs/operational_modena_v1.yaml"
    expected_config_sha = "b407f63deba884db27e763cad1c16a1e42d422e546b942741320cb36012b8dea"
    # Dataset provenance hashes canonical configuration JSON, not YAML bytes.
    config = OperationalConfig(**yaml.safe_load(config_path.read_text()))
    config.validate()
    canonical_sha = hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()
    manifest = read(ROOT/config.output_dir/"manifest.json")
    assert canonical_sha == manifest["config_sha256"] == expected_config_sha, "Physical configuration changed"
    splits = read(reference/"splits.json")
    assert splits == read(ROOT/"runs/operational/mechanism_experts_v2/splits.json")
    assert splits["calibration"] == [3, 4, 13] and splits["validation"] == [6, 10, 11]
    for name, expected in new["source_sha256"].items():
        assert sha(ROOT/"src/wdn"/name) == expected, f"Changed training source: {name}"
    assert sha(ROOT/"src/wdn/probe_expert_calibration.py") == tuned["source_sha256"]

    models = {"original": joblib.load(original/"model.joblib"),
              "dynamic": joblib.load(dynamic/"model.joblib")}
    labels, predictions = {}, {"original": {}, "dynamic": {}}
    hashes = {}
    for split in ("train", "calibration", "validation"):
        assert sha(reference/f"features_{split}.npz") == old["reference_feature_sha256"][split]
        base = dict(np.load(reference/f"features_{split}.npz"))
        augmented = dict(np.load(dynamic/f"features_{split}.npz"))
        for key in ("labels", "families"):
            np.testing.assert_array_equal(augmented[key], base[key])
        np.testing.assert_array_equal(augmented["X"][:, :base["X"].shape[1]], base["X"])
        assert set(augmented["scenario"].tolist()) == set(splits[split])
        if split == "train":
            continue
        labels[split] = augmented
        for kind, directory, X in (("original", original, base["X"]),
                                    ("dynamic", dynamic, augmented["X"])):
            saved = dict(np.load(directory/f"predictions_{split}.npz"))
            fresh = models[kind].predict(X)
            for key in saved:
                np.testing.assert_array_equal(fresh[key], saved[key])
            predictions[kind][split] = fresh
            hashes[str((directory/f"predictions_{split}.npz").relative_to(ROOT))] = sha(directory/f"predictions_{split}.npz")

    cal, val = labels["calibration"], labels["validation"]
    for kind in ("original", "dynamic"):
        for method, row in new[kind].items():
            name, objective = method.split("_", 1)
            selection = calibrate_threshold(predictions[kind]["calibration"][name], cal["labels"], cal["families"],
                objective=objective, max_fpr=1. if objective == "legacy" else .005,
                min_replay_f1=0. if objective == "legacy" else .5)
            assert selection == row["selection"]
            result = summarise(predictions[kind]["validation"][name], val["labels"], val["families"], selection["threshold"])
            assert result == row["validation"]
            assert family_summary(result) == {k: row[k] for k in family_summary(result)}

    # Independent expert measurements use one global calibration F1 threshold
    # each, not an oracle threshold chosen separately for validation families.
    audit = {"test_evaluated": False, "threshold_protocol": "global calibration F1 per expert",
             "source_model_sha256": sha(dynamic/"model.joblib"), "expert_matrix": {}}
    for i, name in enumerate(MECHANISMS):
        threshold = best_f1_threshold(predictions["dynamic"]["calibration"]["experts"][:, i], cal["labels"])
        row = {"threshold": threshold, "per_family": {}}
        for family, label in enumerate(FAMILY_NAMES):
            mask = val["families"] == family
            y, score = val["labels"][mask], predictions["dynamic"]["validation"]["experts"][mask, i]
            row["per_family"][label] = {**_binary_counts(score, y, threshold),
                "auprc": float(average_precision_score(y, score)) if y.any() else None}
        audit["expert_matrix"][name] = row
    (dynamic/"expert_audit.json").write_text(json.dumps(audit, indent=2))
    audit_lines = ["# Dynamic residual expert audit", "", "Validation only; independent expert scores without routing.",
        "Each expert uses one global threshold selected on calibration F1.", "",
        "| Expert | Random AUPRC | Replay AUPRC | Drift AUPRC | Noise AUPRC | Targeted AUPRC | Clean FPR |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in audit["expert_matrix"].items():
        audit_lines.append(f"| {name} | "+" | ".join(f"{row['per_family'][f]['auprc']:.4f}" for f in FAMILIES)
                           +f" | {row['per_family']['clean']['fpr']:.6f} |")
    audit_lines += ["", "The drift and noise owners still do not beat every other expert on their own family.",
                   "This is not evidence that all experts are strong. No independent-seed test was performed.", ""]
    (dynamic/"expert_audit.md").write_text("\n".join(audit_lines))

    # Reproduce the separate calibration-only Optuna selection, including all
    # trials. Validation cannot be used to pick a different trial here.
    bundle = joblib.load(fusion/"calibration.joblib")
    normal_tail = NormalTailEvidence().fit(predictions["dynamic"]["calibration"]["experts"][cal["labels"] == 0])
    np.testing.assert_array_equal(bundle["normal_tail"].normal_, normal_tail.normal_)
    scores = dict(np.load(fusion/"scores.npz"))
    for split in labels:
        reproduced = (bundle["normal_tail"].transform(predictions["dynamic"][split]["experts"])+bundle["offsets"]).max(1)
        np.testing.assert_array_equal(scores[split], reproduced)
    for name, expected in tuned["input_sha256"].items():
        assert sha(dynamic/name) == expected
    trials = read(fusion/"trials.json")
    assert len(trials) == tuned["trials"] == 40
    best = max(trials, key=lambda trial: trial["score"])
    assert best["selection"] == tuned["selection"]
    assert np.array_equal(bundle["offsets"], tuned["offsets"])
    assert bundle["threshold"] == tuned["selection"]["threshold"]
    assert calibrate_threshold(scores["calibration"], cal["labels"], cal["families"], max_fpr=.005) == tuned["selection"]
    assert summarise(scores["validation"], val["labels"], val["families"], bundle["threshold"]) == tuned["validation"]

    rows = []
    baseline = old["results"]["mixture"]
    assert select_threshold(predictions["original"]["calibration"]["mixture"], cal["labels"], cal["families"]) == baseline["threshold"]
    assert summarise(predictions["original"]["validation"]["mixture"], val["labels"], val["families"], baseline["threshold"]) == baseline
    rows.append({"name": "Previous reported mixture", "run": str(original.relative_to(ROOT)),
                 "validation": baseline, **family_summary(baseline),
                 "selection_note": "Original coarse-grid overall/replay calibration objective"})
    for title, kind, method in (
        ("Old features, family-balanced threshold", "original", "mixture_macro_f1"),
        ("Dynamic features, overall priority", "dynamic", "mixture_legacy"),
        ("Dynamic features, family balance", "dynamic", "mixture_macro_f1")):
        rows.append({"name": title, "run": str(dynamic.relative_to(ROOT)), "method": f"{kind}/{method}", **new[kind][method]})
    rows.append({"name": "Tail-evidence fusion, calibration Optuna winner", "run": str(fusion.relative_to(ROOT)),
                 **{k: tuned[k] for k in ("selection", "validation", "macro_f1", "worst_family_f1", "macro_recall")}})
    verified = {"test_evaluated": False, "frozen_configuration_sha256": expected_config_sha,
        "splits": splits, "prediction_hashes": hashes, "model_reload_predictions_exact": True,
        "baseline_features_labels_endpoints_exact": True, "all_operating_points_reproduced": True,
        "tail_calibration_selection_reproduced": True, "tail_calibration_trials": len(trials),
        "validation_observations": len(val["labels"]), "validation_positives": int(val["labels"].sum())}
    (dynamic/"verification.json").write_text(json.dumps(verified, indent=2))
    output = {"scope": "pressure detection development probes; not the temporal GNN or locked test",
        "test_evaluated": False, "verification": verified, "comparisons": rows,
        "full_operating_points": {k: new[k] for k in ("original", "dynamic")},
        "expert_audit": audit,
        "selection_warning": "All development variants disclosed. No final model is declared from these validation comparisons."}
    (ROOT/"thesis_v2/outputs/family_balance_results.json").write_text(json.dumps(output, indent=2))

    lines = ["# Weak-family follow-up: drift and noise", "",
        "## Scope and fixed protocol", "",
        "Same frozen rank-16 blind reference, data, scenario splits and 16-step observation endpoints.",
        "Pressure and flow missing probabilities remain 0.50. Attack amplitudes, prevalence, noise and splits were not changed.",
        "This is a pressure-detection tree-expert prototype, not a temporal-GNN result or full pressure/flow reconstruction.",
        "All figures below are development validation results. No test example is used for training, features, inference or scoring.", "",
        "## What changed", "",
        "31 causal features were added to the original 29: robust online residual baseline/scale, cumulative signed drift evidence,",
        "past-only forecast error, recent-versus-past level shift, and innovation statistics over 8/16/32/48 hours.",
        "States reset per scenario and never use attack labels to select clean history. General/abrupt/drift/noise heads and the router",
        "can access the new context; the replay expert retains its old inputs. The fixed tree-training recipe is unchanged.", "",
        "Threshold selection now includes an equal-weight mean F1 over the five attack families, rather than only aggregate F1/replay.",
        "A provisional 0.5% calibration false-positive budget applies to both all normal readings and clean episodes, with replay F1 >= 0.5.",
        "This budget is an explicit sensitivity experiment, not a field-approved alarm rate or an unseen-data guarantee.",
        "Exact legacy-objective thresholds and frozen-feature controls are also reported, since the old search used a coarse score grid.",
        "One global threshold is used at inference: true attack-family labels do not select a threshold or expert.", "",
        "Score learning and threshold selection are distinct tasks; calibration does not itself establish better discrimination.",
        "See [scikit-learn threshold guidance](https://scikit-learn.org/stable/modules/classification_threshold.html).", "",
        "## Validation operating points", "",
        "Family F1 is binary detection within that family's episodes, not multiclass family-identification accuracy.",
        "Clean-episode false positives enter overall F1/FPR but not the mean over attack families; the FPR guard is therefore essential.", "",
        "| Variant | Overall F1 | Random | Replay | Drift | Noise | Targeted | Family macro F1 | FP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        v = row["validation"]
        lines.append(f"| {row['name']} | {v['overall']['f1']:.4f} | "
                     +" | ".join(f"{v['per_family'][f]['f1']:.4f}" for f in FAMILIES)
                     +f" | {row['macro_f1']:.4f} | {v['overall']['fp']} |")
    lines += ["", "| Variant | Overall recall | Sensor FPR (%) | Clean-episode FPR (%) | AUPRC | Calibration macro F1 |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        v = row["validation"]
        cm = row.get("selection", {}).get("calibration", {}).get("macro_f1")
        lines.append(f"| {row['name']} | {v['overall']['recall']:.4f} | {100*v['overall']['fpr']:.4f} | "
                     f"{100*v['per_family']['clean']['fpr']:.4f} | {v['overall']['auprc']:.4f} | "
                     +(f"{cm:.4f}" if cm is not None else "not recorded")+" |")
    balanced = new["dynamic"]["mixture_macro_f1"]["validation"]
    lines += ["", "## What the balanced point still misses", "",
              "| Family | TP / positives | Recall | Precision | F1 |",
              "|---|---:|---:|---:|---:|"]
    for family in FAMILIES:
        r = balanced["per_family"][family]
        lines.append(f"| {family} | {r['tp']} / {r['tp']+r['fn']} | {r['recall']:.4f} | {r['precision']:.4f} | {r['f1']:.4f} |")
    lines += ["", "Crossing 0.5 F1 for each family does not mean catching all attacks. Drift/noise still have low recall.",
        "The balanced point is a demonstrated tradeoff, not a final model selected after looking at validation.",
        "The aggregate-priority point improves overall F1 while leaving weak-family F1 below 0.5.", "",
        "## Separate score-calibration experiment: negative finding", "",
        "A normal-tail evidence transform and expert offsets were fitted on calibration only; 40 CPU Optuna trials operated on",
        "already frozen predictions, with no retraining and no extension of the completed four-trial GNN search.",
        "Validation arrays were loaded only after selecting the winning offsets/threshold. All 40 trials are saved.",
        "This winner scores better on calibration macro F1 but worse on validation drift than the dynamic balanced point.",
        "It is not presented as an improvement or replaced by a validation-picked Optuna trial. Tail evidence is not an attack probability.", "",
        "## Independent experts", "",
        "The audit uses each expert without routing; AUPRC is independent of the reported F1 threshold.", "",
        "| Own-family expert | Previous AUPRC | Dynamic AUPRC | Previous clean FPR (%) | Dynamic clean FPR (%) |",
        "|---|---:|---:|---:|---:|"]
    for name, family in (("drift", "stealthy"), ("noise", "noise")):
        before, after = old["expert_matrix"][name]["per_family"], audit["expert_matrix"][name]["per_family"]
        lines.append(f"| {name} | {before[family]['auprc']:.4f} | {after[family]['auprc']:.4f} | "
                     f"{100*before['clean']['fpr']:.4f} | {100*after['clean']['fpr']:.4f} |")
    lines += ["", "Dedicated drift/noise experts improve their own-family rankings but still do not beat every other expert.",
        "See `runs/operational/family_balance_v1/expert_audit.md` for the full matrix, including losses on other families.", "",
        "## Verification and limits", "",
        "Saved original/dynamic models reproduce every calibration and validation prediction exactly after reload.",
        "Original 29 features, labels, family ordering and endpoints match the frozen baseline exactly.",
        "All 12 original/dynamic operating points and the score-calibration winner reproduce from calibration and frozen predictions.",
        "The original configuration hash is unchanged; scenario splits match the GNN pilot. Verification is saved with the dynamic run.",
        "Validation contains 62,618 observed readings, 419 positives and only 35 replay positives in three scenarios.",
        "Repeated use of these development scenarios, correlated readings, and a large calibration-to-validation family-score gap",
        "prevent a claim of generalisation. No independent seed, event-level false-alarm rate or detection-delay validation is established.",
        "The 0.90 overall-F1 goal and strong recall for every family are not achieved.", "",
        "Next work should prioritise scenario-level validation and missed-event/delay analysis for drift/noise before further adaptive",
        "search on these same validation results. Training-only richer temporal models are an option, not a promised gain.", ""]
    (ROOT/"thesis_v2/FAMILY_BALANCE_RESULTS.md").write_text("\n".join(lines))
    print(json.dumps({"verified": verified, "comparisons": [{"name": r["name"],
        "overall_f1": r["validation"]["overall"]["f1"], "macro_f1": r["macro_f1"],
        "worst_family_f1": r["worst_family_f1"]} for r in rows]}, indent=2))


if __name__ == "__main__":
    main()
