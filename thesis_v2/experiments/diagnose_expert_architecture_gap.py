"""TRAIN-only diagnosis for separate drift/noise expert architectures."""
from __future__ import annotations

from math import erf, sqrt
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from wdn.run_expert_redesign import load_arrays, read_json, write_json
from wdn.screen_normal_nuisance import concatenate
from wdn.screen_recovery import expert_diagnostics


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/"runs/operational/mechanism_redesign_v1"
WEAK = ROOT/"runs/operational/weak_family_v1/trial_0000/oof_predictions.npz"
AR = ROOT/"runs/operational/latent_ar_stage_b_v1"
DATA = ROOT/"data/thesis_v2/operational_modena_seed811"


def best_f1(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels) > 0
    order = np.argsort(-scores, kind="stable")
    score, y = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(score[:-1] != score[1:]), len(score)-1]
    predicted = ends+1
    tp = np.cumsum(y)[ends]
    fp, fn = predicted-tp, y.sum()-tp
    f1 = 2*tp/np.maximum(2*tp+fp+fn, 1)
    best = int(np.argmax(f1))
    precision = tp[best]/max(predicted[best], 1)
    recall = tp[best]/max(y.sum(), 1)
    return {"f1": float(f1[best]), "precision": float(precision),
        "recall": float(recall),
        "threshold": float(np.nextafter(score[ends[best]], np.array(-np.inf, dtype=score.dtype))),
        "positives": int(y.sum()), "negatives": int((~y).sum())}


def feature_ranking(X, names, arrays, family_id, limit=15):
    selected = arrays["families"] == family_id
    y = arrays["labels"][selected] > 0
    rows = []
    for index, name in enumerate(names):
        values = X[selected, index]
        direct = float(average_precision_score(y, values))
        inverse = float(average_precision_score(y, -values))
        rows.append({"feature": name, "absolute_direction_ap": max(direct, inverse),
                     "direction": "+" if direct >= inverse else "-"})
    return sorted(rows, key=lambda row: row["absolute_direction_ap"], reverse=True)[:limit]


def expected_drift_magnitude(events, event_ids, timesteps, nodes):
    result = np.full(len(event_ids), np.nan)
    for index, (event_id, timestep, node) in enumerate(zip(event_ids, timesteps, nodes)):
        event = events[int(event_id)]
        targets = event["targets"]["pressure"]
        position = targets.index(int(node))
        age = int(timestep)-event["start_timestep"]+1
        result[index] = event["magnitudes"]["pressure"][position]*min(age/event["ramp_steps"], 1.)
    return result


def main():
    stage = read_json(AR/"summary.json")
    signature = read_json(AR/"signature.json")
    assert not any(stage[key] for key in
                   ("calibration_evaluated", "validation_evaluated", "test_evaluated"))
    arrays = load_arrays(AR/"oof_predictions.npz")
    feature_parts = [load_arrays(BASE/f"fold_{fold}/features_held_out.npz") for fold in range(3)]
    features = concatenate([{key: part[key] for key in
        ("X", "labels", "families", "event", "scenario", "timestep", "node")}
        for part in feature_parts])
    for key in ("labels", "families", "event", "scenario", "timestep", "node"):
        np.testing.assert_array_equal(features[key], arrays[key])
    weak = load_arrays(WEAK)
    for key in ("labels", "families", "scenario"):
        np.testing.assert_array_equal(weak[key], arrays[key])
    names = read_json(BASE/"feature_names.json")
    events = read_json(DATA/"events.json")
    train = set(signature["splits"]["train"])
    train_events = [{**event, "_event_id": index} for index, event in enumerate(events)
                    if event["scenario_id"] in train]

    score_sets = {
        "legacy_local_tree": weak["experts"][:, 3:5],
        "latent_ar_likelihood": arrays["scores_A"],
    }
    score_diagnostics = {name: expert_diagnostics(arrays, score, train_events)
                         for name, score in score_sets.items()}
    oracle_f1 = {}
    for family, family_id, column in (("drift", 3, 0), ("noise", 4, 1)):
        selected = arrays["families"] == family_id
        oracle_f1[family] = {name: best_f1(score[selected, column], arrays["labels"][selected])
                             for name, score in score_sets.items()}

    ar_innovation = (arrays["error"]-arrays["mean_A"])/arrays["scale_A"]
    extra = np.column_stack([ar_innovation, np.abs(ar_innovation), ar_innovation**2,
                             arrays["scores_A"]])
    extra_names = ["latent_ar_innovation", "latent_ar_abs_innovation",
                   "latent_ar_squared_innovation", "latent_ar_drift_probability",
                   "latent_ar_noise_probability"]
    X = np.column_stack([features["X"], extra])
    all_names = names+extra_names
    ranking = {"drift": feature_ranking(X, all_names, arrays, 3),
               "noise": feature_ranking(X, all_names, arrays, 4)}

    positive_phase = {}
    for family, family_id, column in (("drift", 3, 0), ("noise", 4, 1)):
        positive = (arrays["families"] == family_id) & (arrays["labels"] > 0)
        event_ids = arrays["event"][positive]
        ages = np.asarray([int(t)-events[int(e)]["start_timestep"]+1
                           for e, t in zip(event_ids, arrays["timestep"][positive])])
        family_rows = {}
        for label, selected_age in (("hours_1_3", ages <= 3),
                                    ("hours_4_6", (ages >= 4) & (ages <= 6)),
                                    ("hours_7_plus", ages >= 7)):
            row = {"positive_endpoints": int(selected_age.sum())}
            for name, score in score_sets.items():
                threshold = oracle_f1[family][name]["threshold"]
                row[f"{name}_recall_at_family_oracle_threshold"] = float(
                    (score[positive, column][selected_age] > threshold).mean()) if selected_age.any() else None
            family_rows[label] = row
        positive_phase[family] = family_rows

    drift_pos = (arrays["families"] == 3) & (arrays["labels"] > 0)
    drift_magnitude = expected_drift_magnitude(events, arrays["event"][drift_pos],
        arrays["timestep"][drift_pos], arrays["node"][drift_pos])
    drift_age = np.asarray([int(t)-events[int(e)]["start_timestep"]+1
                            for e, t in zip(arrays["event"][drift_pos], arrays["timestep"][drift_pos])])
    normal_mae = stage["normal_diagnostics"]["A"]["clean"]["scenario_macro"]["mae_m"]
    detectability = {"normal_ar_macro_mae_m": normal_mae,
        "drift_expected_injected_m": {
            "all_quantiles_10_50_90": np.quantile(drift_magnitude, [.1, .5, .9]).tolist(),
            "first_3h_quantiles_10_50_90": np.quantile(drift_magnitude[drift_age <= 3], [.1, .5, .9]).tolist(),
            "first_3h_fraction_below_sensor_noise_0_10m": float((drift_magnitude[drift_age <= 3] < .10).mean()),
            "first_3h_fraction_below_ar_normal_mae": float((drift_magnitude[drift_age <= 3] < normal_mae).mean())}}
    noise_pos = (arrays["families"] == 4) & (arrays["labels"] > 0)
    noise_sigma = np.asarray([.1*events[int(event_id)]["noise_factor"]
                              for event_id in arrays["event"][noise_pos]])
    probability_below_mae = np.asarray([erf(normal_mae/(sqrt(2)*sigma)) for sigma in noise_sigma])
    detectability["noise_injected_std_m"] = {
        "quantiles_10_50_90": np.quantile(noise_sigma, [.1, .5, .9]).tolist(),
        "mean_probability_single_injection_abs_below_ar_normal_mae": float(probability_below_mae.mean())}

    event_counts = {split: {family: sum(event["scenario_id"] in set(sids) and event["family"] == source
        for event in events) for family, source in (("drift", "stealthy"), ("noise", "noise"))}
        for split, sids in signature["splits"].items()}
    result = {"scope": "TRAIN-only descriptive architecture diagnosis",
        "pressure_and_flow_missing": .50, "attacks_or_splits_changed": False,
        "calibration_evaluated": False, "validation_evaluated": False,
        "test_evaluated": False, "event_counts_by_split": event_counts,
        "score_diagnostics": score_diagnostics, "family_oracle_f1": oracle_f1,
        "positive_phase_recall": positive_phase, "top_univariate_features": ranking,
        "detectability": detectability}
    write_json(ROOT/"thesis_v2/outputs/expert_architecture_gap.json", result)

    lines = ["# Drift/noise expert architecture gap: TRAIN-only diagnosis", "",
        "No calibration, validation or test examples were evaluated. Data, attacks, splits and 0.50 missing rates are unchanged.",
        "Family-oracle thresholds below are hindsight diagnostics and are not deployable F1 results.", "",
        "## What the current scores can separate", "",
        "| Family | Score | Macro AP | Early recall @ .001 FPR | Family-oracle F1 |",
        "|---|---|---:|---:|---:|"]
    for family in ("drift", "noise"):
        for score_name in score_sets:
            row = score_diagnostics[score_name][family]
            lines.append(f"| {family} | {score_name} | {row['macro_ap']:.4f} | "
                         f"{row['early_macro_recall']:.4f} | {oracle_f1[family][score_name]['f1']:.4f} |")
    lines += ["", "## Positive recall by event age at each score's family-oracle threshold", "",
        "| Family / phase | Positives | Local tree recall | Latent AR recall |", "|---|---:|---:|---:|"]
    for family, phases in positive_phase.items():
        for phase, row in phases.items():
            lines.append(f"| {family} / {phase} | {row['positive_endpoints']} | "
                f"{row['legacy_local_tree_recall_at_family_oracle_threshold']:.4f} | "
                f"{row['latent_ar_likelihood_recall_at_family_oracle_threshold']:.4f} |")
    drift = detectability["drift_expected_injected_m"]
    noise = detectability["noise_injected_std_m"]
    lines += ["", "## Physical observability", "",
        f"AR normal macro MAE is {normal_mae:.4f} m. During the first three drift hours, "
        f"{drift['first_3h_fraction_below_sensor_noise_0_10m']:.1%} of observed positive injections are below 0.10 m and "
        f"{drift['first_3h_fraction_below_ar_normal_mae']:.1%} are below the AR normal MAE.",
        f"The first-three-hour drift injection 10/50/90% quantiles are "
        f"{drift['first_3h_quantiles_10_50_90'][0]:.3f}/{drift['first_3h_quantiles_10_50_90'][1]:.3f}/"
        f"{drift['first_3h_quantiles_10_50_90'][2]:.3f} m.",
        f"Noise injection standard deviation 10/50/90% quantiles are "
        f"{noise['quantiles_10_50_90'][0]:.3f}/{noise['quantiles_10_50_90'][1]:.3f}/"
        f"{noise['quantiles_10_50_90'][2]:.3f} m; a single injected draw has mean "
        f"{noise['mean_probability_single_injection_abs_below_ar_normal_mae']:.1%} probability of being smaller than the AR normal MAE.", "",
        "## Architecture diagnosis", "",
        "1. Existing tree experts never receive latent-AR innovations; the AR likelihood and nonlinear local expert remain disconnected.",
        "2. Both families use endpoint classifiers. They carry generic score memory, not family-specific latent state or matched temporal evidence.",
        "3. Drift positives are labelled from ramp onset even when their displacement is below ordinary reference error; early detection needs evidence pooling across time and related sensors.",
        "4. Noise needs a variance-regime likelihood with uncertainty about the latent AR state; a one-step excess-variance approximation discards temporal covariance.",
        "5. There are only four TRAIN events per family and one calibration/validation event per family, so broad neural/Optuna search would overfit scenario identity.", "",
        "Recommended next screen: a drift signed matched-filter bank plus persistence state, and a separate exact AR variance/Kalman noise expert, "
        "followed by a small monotone expert combiner trained only on nested scenario-OOF scores."]
    (ROOT/"thesis_v2/EXPERT_ARCHITECTURE_GAP.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"family_oracle_f1": oracle_f1, "positive_phase_recall": positive_phase,
                      "detectability": detectability}, indent=2))


if __name__ == "__main__":
    main()
