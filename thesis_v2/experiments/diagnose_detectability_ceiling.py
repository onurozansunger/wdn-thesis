"""Where the drift/noise misses live, and what any detector could reach.

Joins the frozen generator-held OOF expert scores to the injected displacement
recorded in each dataset's event ledger, then reports:

1. drift recall as a function of injected displacement in clean-residual sd,
2. the pointwise F1 an ideal zero-false-alarm detector reaches if it resolves
   every observation at or above a given displacement and nothing below it,
3. how that ideal ceiling moves when the hourly decision for time t may be
   finalised at t+delta instead of at t,
4. noise recall by hour-in-event and by injected noise factor.

The ceilings are properties of the benchmark, not of any model: they assume
perfect detection above the stated displacement, perfect rejection below it and
zero false alarms. The clean-residual sd is read from the frozen
`drift_noise_limits.json` decomposition; the raw and mean-removed variants are
both reported because the ceiling is sensitive to that choice.

Reads only frozen TRAIN artefacts and dataset event ledgers. No model is
fitted, no data is generated, and calibration, validation and test are not
read.

    python3 thesis_v2/experiments/diagnose_detectability_ceiling.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OOF = ROOT / "runs/operational/seasonal_family_experts_v3/oof_predictions.npz"
LIMITS = ROOT / "thesis_v2/outputs/drift_noise_limits.json"
DATA = ROOT / "data/thesis_v2"
OUT = ROOT / "thesis_v2/outputs/detectability_ceiling.json"
SOURCES = {811: "operational_modena_seed811",
           1811: "operational_train_expansion_seed1811",
           2811: "operational_train_expansion_seed2811",
           3811: "operational_train_expansion_seed3811"}
DELTAS = (0, 1, 2, 3, 4, 6)
BANDS = ((0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, np.inf))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def oracle_threshold(score, label):
    """The threshold maximising F1 for this fixed ordering. Hindsight only."""
    positives = int(label.sum())
    order = np.argsort(-score, kind="stable")
    y = label[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    f1 = 2 * tp / (2 * tp + fp + positives - tp)
    best = int(f1.argmax())
    return float(score[order][best]), {"f1": float(f1[best]), "tp": int(tp[best]),
                                       "fp": int(fp[best]), "positives": positives,
                                       "recall": float(tp[best] / positives),
                                       "precision": float(tp[best] / (tp[best] + fp[best]))}


def ledger(family):
    events = {}
    for code, name in SOURCES.items():
        for event in json.loads((DATA / name / "events.json").read_text()):
            if event["family"] == family:
                events.setdefault((code, event["scenario_id"]), []).append(event)
    return events


def drift_amplitude(events, source, scenario, timestep, node):
    """Injected metres at this hour: magnitude * min((age+1)/ramp, 1)."""
    local = int(scenario) - int(source) * 1000
    for event in events.get((int(source), local), []):
        age = int(timestep) - event["start_timestep"]
        if 0 <= age < event["actual_steps"]:
            magnitude = dict(zip(event["targets"]["pressure"],
                                 event["magnitudes"]["pressure"])).get(int(node))
            if magnitude is None:
                continue
            return magnitude, age, event["ramp_steps"], event["actual_steps"]
    return None


def ideal_f1(reached, positives):
    """F1 of a detector that resolves `reached` positives with no false alarm."""
    return float(2 * reached / (2 * reached + (positives - reached)))


def main():
    data = np.load(OOF, allow_pickle=True)
    label, family = data["labels"], data["families"]
    source, scenario = data["source"], data["scenario"]
    timestep, node = data["timestep"], data["node"]
    scores = data["scores"]

    decomposition = json.loads(LIMITS.read_text())["normal_error_decomposition"]
    mse = decomposition["mse_m2"]
    residual_sd = {
        "clean_residual_rms_m": float(np.sqrt(mse)),
        "mean_removed_rms_m": float(np.sqrt(mse * (1 - decomposition["sensor_scenario_mean_fraction_of_mse"]))),
    }

    result = {"scope": "expanded TRAIN generator-held OOF joined to event ledgers",
              "oof_sha256": sha(OOF), "limits_sha256": sha(LIMITS),
              "residual_scale_m": residual_sd,
              "ceiling_assumptions": "perfect detection above the stated displacement, "
                                     "perfect rejection below it, zero false alarms",
              "calibration_evaluated": False, "validation_evaluated": False,
              "test_evaluated": False}

    # ---- drift -----------------------------------------------------------
    events = ledger("stealthy")
    rows = np.where((family == 3) & (label == 1))[0]
    matched = [drift_amplitude(events, source[i], scenario[i], timestep[i], node[i]) for i in rows]
    if any(m is None for m in matched):
        raise ValueError("Unmatched drift positive: ledger and OOF disagree")
    magnitude, age, ramp, length = (np.array([m[k] for m in matched], float) for k in range(4))
    amplitude = magnitude * np.minimum((age + 1) / ramp, 1.0)
    threshold, point = oracle_threshold(scores[family == 3, 0], label[family == 3])
    detected = scores[rows, 0] >= threshold
    positives = rows.size

    drift = {"positives": positives, "oracle_point": point,
             "early_share": float((age < 3).mean()),
             "amplitude_quantiles_m": {"p10": float(np.percentile(amplitude, 10)),
                                       "p50": float(np.percentile(amplitude, 50)),
                                       "p90": float(np.percentile(amplitude, 90))},
             "early_amplitude_quantiles_m": {"p10": float(np.percentile(amplitude[age < 3], 10)),
                                             "p50": float(np.percentile(amplitude[age < 3], 50)),
                                             "p90": float(np.percentile(amplitude[age < 3], 90))},
             "recall_by_displacement": {}, "ceiling_by_latency": {}}

    for name, sd in residual_sd.items():
        z = amplitude / sd
        drift["recall_by_displacement"][name] = [
            {"band_sd": [lo, None if np.isinf(hi) else hi],
             "n": int(((z >= lo) & (z < hi)).sum()),
             "recall": float(detected[(z >= lo) & (z < hi)].mean()) if ((z >= lo) & (z < hi)).any() else None}
            for lo, hi in BANDS]
        latency = {}
        for delta in DELTAS:
            visible_age = np.minimum(age + delta, length - 1)
            visible = magnitude * np.minimum((visible_age + 1) / ramp, 1.0)
            reached = int((visible >= sd).sum())
            latency[str(delta)] = {"resolvable_at_1sd": reached,
                                   "share": float(reached / positives),
                                   "ideal_f1": ideal_f1(reached, positives)}
        drift["ceiling_by_latency"][name] = latency
    # headroom above one sd at the current operating point
    z = amplitude / residual_sd["clean_residual_rms_m"]
    needed = int(np.ceil((point["fp"] + positives) * 0.8 / 1.2))
    drift["gap_to_080"] = {
        "tp_now": point["tp"], "tp_required_at_same_fp": needed,
        "missed_at_or_above_1sd": int(((~detected) & (z >= 1)).sum()),
        "sufficient": bool(int(((~detected) & (z >= 1)).sum()) >= needed - point["tp"])}

    # ---- noise -----------------------------------------------------------
    events = ledger("noise")
    rows = np.where((family == 4) & (label == 1))[0]
    ages, factors = [], []
    for i in rows:
        local = int(scenario[i]) - int(source[i]) * 1000
        hit = None
        for event in events.get((int(source[i]), local), []):
            a = int(timestep[i]) - event["start_timestep"]
            if 0 <= a < event["actual_steps"]:
                hit = (a, event["noise_factor"])
                break
        if hit is None:
            raise ValueError("Unmatched noise positive: ledger and OOF disagree")
        ages.append(hit[0])
        factors.append(hit[1])
    ages, factors = np.array(ages), np.array(factors)
    threshold, point = oracle_threshold(scores[family == 4, 1], label[family == 4])
    detected = scores[rows, 1] >= threshold
    noise = {"positives": rows.size, "oracle_point": point,
             "recall_by_age": {str(a): {"n": int((ages == a).sum()),
                                        "recall": float(detected[ages == a].mean())}
                               for a in range(int(ages.max()) + 1) if (ages == a).any()},
             "recall_by_factor": [
                 {"factor_range": [lo, hi], "n": int(((factors >= lo) & (factors < hi)).sum()),
                  "injected_sd_over_residual_sd": float(0.1 * (lo + hi) / 2 / residual_sd["clean_residual_rms_m"]),
                  "recall": float(detected[(factors >= lo) & (factors < hi)].mean())}
                 for lo, hi in ((3, 4.5), (4.5, 6.0), (6.0, 8.1))]}
    noise["gap_to_080"] = {"tp_now": point["tp"],
                           "tp_required_at_same_fp": int(np.ceil((point["fp"] + rows.size) * 0.8 / 1.2))}

    result["drift"] = drift
    result["noise"] = noise
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print("wrote", OUT)
    for name in residual_sd:
        print(f"\ndrift ceiling, {name} = {residual_sd[name]:.4f} m")
        for delta, entry in drift["ceiling_by_latency"][name].items():
            print(f"  delta={delta}: resolvable {entry['resolvable_at_1sd']}/{positives}"
                  f" -> ideal F1 {entry['ideal_f1']:.4f}")
    print("\ndrift gap:", drift["gap_to_080"])
    print("noise gap:", noise["gap_to_080"])


if __name__ == "__main__":
    main()
