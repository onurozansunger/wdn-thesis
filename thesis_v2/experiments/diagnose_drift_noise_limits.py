"""Descriptive audit of frozen TRAIN caches; no fitting or model evaluation.

Hindsight thresholds and clean-period means below are deliberately nondeployable.
They diagnose fixed rankings and residual error, not a task-wide performance limit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/operational/drift_noise_recovery_v1"
BASE = ROOT / "runs/operational/mechanism_redesign_v1"
CONTROL = ROOT / "runs/operational/weak_family_v1/trial_0000"


def threshold_counts(labels, scores):
    """Every realizable strict-> threshold, with equal scores kept together."""
    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores)
    if not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Finite scores and binary labels required")
    order = np.argsort(-scores, kind="stable")
    ranked, sorted_scores = labels[order], scores[order]
    ends = np.flatnonzero(np.r_[sorted_scores[:-1] != sorted_scores[1:], True])
    return np.r_[0, np.cumsum(ranked)[ends]], np.r_[0, ends + 1]


def hindsight_envelope(labels, scores, scenario):
    """Exact finite-choice fractional optimum, one threshold per scenario.

    At ratio r each independent scenario maximizes 2*TP-r*predicted_count.
    The total positive count is constant. This is not a threshold selector.
    """
    curves, per_scenario = [], {}
    positives = int(labels.sum())
    for sid in np.unique(scenario):
        loc = scenario == sid
        tp, predicted = threshold_counts(labels[loc], scores[loc])
        p = int(labels[loc].sum())
        curves.append((tp, predicted))
        per_scenario[str(int(sid))] = float(np.max(2 * tp / (p + predicted)))
    ratio = 0.0
    for _ in range(1000):
        choices = [int(np.argmax(2 * tp - ratio * n)) for tp, n in curves]
        tp_sum = sum(int(tp[k]) for (tp, _), k in zip(curves, choices))
        n_sum = sum(int(n[k]) for (_, n), k in zip(curves, choices))
        updated = 2 * tp_sum / (positives + n_sum)
        if abs(updated - ratio) < 1e-14:
            residual = sum(float(np.max(2 * tp - updated * n)) for tp, n in curves)
            assert abs(residual - updated * positives) < 1e-8
            return {"aggregate_f1": updated, "by_scenario_f1": per_scenario}
        ratio = updated
    raise RuntimeError("Threshold-envelope calculation failed to converge")


def main():
    inputs = {}

    def read(path):
        inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix == ".json":
            return json.loads(path.read_text())
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    signature = read(RUN / "signature.json")
    allowed = set(signature["splits"]["train"])
    a = read(RUN / "state_oof_predictions.npz")
    assert set(a["scenario"]) == allowed
    scores = {"state": a["scores"]}
    for name, path, key in (("physics", RUN / "physics_oof_predictions.npz", "scores"),
                            ("control", CONTROL / "oof_predictions.npz", "experts")):
        other = read(path)
        for field in ("labels", "families", "scenario"):
            np.testing.assert_array_equal(a[field], other[field])
        scores[name] = other[key][:, 3:5] if name == "control" else other[key]
    envelopes = {}
    for name, score in scores.items():
        envelopes[name] = {}
        for col, fid, family in ((0, 3, "drift"), (1, 4, "noise")):
            loc = a["families"] == fid
            envelopes[name][family] = hindsight_envelope(
                a["labels"][loc], score[loc, col], a["scenario"][loc])

    names = read(BASE / "feature_names.json")
    clean_parts, reference_parts = [], []
    for index, held in enumerate(signature["folds"]):
        features = read(BASE / f"fold_{index}/features_held_out.npz")
        assert set(features["scenario"]) == set(held) <= allowed
        clean = features["families"] == 0
        x = features["X"][clean].astype(float)
        error = x[:, names.index("residual")] * x[:, names.index("normal_error_scale")]
        clean_parts.append({"error": error, **{key: features[key][clean]
                            for key in ("scenario", "node", "timestep")}})
        ref = read(RUN / f"fold_{index}/reference_audit.npz")
        for key in ("labels", "families", "scenario"):
            np.testing.assert_array_equal(ref[key], features[key])
        reference_parts.append(ref)
    clean = {key: np.concatenate([part[key] for part in clean_parts]) for key in clean_parts[0]}
    _, group = np.unique(np.column_stack([clean["scenario"], clean["node"]]),
                         axis=0, return_inverse=True)
    counts = np.bincount(group)
    means = np.bincount(group, weights=clean["error"]) / counts
    centered = clean["error"] - means[group]
    mse = float(np.mean(clean["error"] ** 2))
    bias_mse = float(np.sum(counts * means ** 2) / len(group))
    np.testing.assert_allclose(mse, bias_mse + np.mean(centered ** 2), rtol=1e-12)
    order = np.lexsort((clean["timestep"], clean["node"], clean["scenario"]))
    adjacent = ((group[order][1:] == group[order][:-1])
                & (np.diff(clean["timestep"][order]) == 1))
    centered = centered[order]
    normal_error = {"clean_observations": len(group), "mse_m2": mse,
        "sensor_scenario_mean_component_mse_m2": bias_mse,
        "sensor_scenario_mean_fraction_of_mse": bias_mse / mse,
        "consecutive_clean_pairs": int(adjacent.sum()),
        "posthoc_centered_lag1_correlation": float(np.corrcoef(
            centered[:-1][adjacent], centered[1:][adjacent])[0, 1]),
        "caveat": "Whole-clean-period sensor/scenario means use hindsight and labels. "
                  "This variance decomposition is descriptive, not a deployable correction "
                  "or an estimate of recoverable detector performance."}
    ref = {key: np.concatenate([part[key] for part in reference_parts])
           for key in reference_parts[0]}
    coverage = {}
    for name, loc in (("clean", ref["families"] == 0),
                      ("drift_positive", (ref["families"] == 3) & (ref["labels"] > 0)),
                      ("noise_positive", (ref["families"] == 4) & (ref["labels"] > 0)),
                      ("drift_early_positive", (ref["families"] == 3) & (ref["labels"] > 0) & ref["early"]),
                      ("noise_early_positive", (ref["families"] == 4) & (ref["labels"] > 0) & ref["early"])):
        coverage[name] = {"observations": int(loc.sum()),
            "positive_physical_weight": int((ref["errors"][loc, 2] > 0).sum()),
            "at_least_two_anchors": int((ref["errors"][loc, 3] >= 2).sum())}
    result = {"scope": "Frozen TRAIN OOF caches only; no fit, prediction or split extraction.",
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "threshold_envelope_caveat": "Family event-period metric; a different hindsight threshold "
            "per scenario, no FPR constraint. Bound for each fixed score ordering only, not "
            "a deployable result, validation result, new-fusion bound or task/Bayes limit.",
        "threshold_envelopes": envelopes, "normal_error_decomposition": normal_error,
        "physical_replacement_coverage": coverage, "inputs_sha256": inputs}
    output = ROOT / "thesis_v2/outputs/drift_noise_limits.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "inputs_sha256"}, indent=2))


if __name__ == "__main__":
    main()
