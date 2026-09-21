"""Is this actually a mixture of experts, and does the mixture earn its keep?

Four questions, answered on calibration only:

1. Specialisation. Is each mechanism expert the best expert on its own
   mechanism, or is one general expert carrying everything?
2. Routing. Does the router put its mass on the right mechanism, and how
   confident is it on clean water?
3. Contribution. Does routing beat uniform averaging and beat the general
   expert alone? Without this the mixture is decoration.
4. Cross-mechanism discipline. How often does each expert fire inside another
   mechanism's event? This is what a shared false-alarm budget punishes, and it
   is what the cross-mechanism refit was meant to fix.

    python3 thesis_v2/experiments/audit_moe_architecture.py
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import joblib
import lightgbm  # noqa: F401  loaded before scikit-learn's OpenMP
import numpy as np
from sklearn.metrics import average_precision_score

from wdn.latency_deployment import forward_max, quantile_threshold, specialist_bank, specialist_scores
from wdn.run_expert_redesign import CampaignData, read_json, write_json

DATA = Path("data/thesis_v2")
ORIGINAL = DATA / "operational_modena_seed811"
CALIBRATION_SEEDS = (12811, 13811, 14811, 15811)
DETECTOR = Path("runs/operational/expanded_detector_v1/bundle.joblib")
CROSS = Path("runs/operational/cross_mechanism_experts_v1/bundle.joblib")
REFERENCE = Path("runs/operational/seasonal_family_deployment_v2/full/reference.joblib")
SPLITS = Path("runs/operational/blind_reference_probe_rank16/splits.json")
OUT = Path("thesis_v2/outputs/moe_architecture_audit.json")
MECHANISMS = ("general", "abrupt", "replay", "drift", "noise")
ROUTER_TARGET = {1: "abrupt", 5: "abrupt", 2: "replay", 3: "drift", 4: "noise"}
FAMILIES = {1: "random", 2: "replay", 3: "drift", 4: "noise", 5: "targeted"}
CLEAN_BUDGET = .005
DELTA = 3


def oracle_f1(score, labels):
    labels = np.asarray(labels) > 0
    order = np.argsort(-np.asarray(score), kind="stable")
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    return float((2 * tp / (2 * tp + fp + labels.sum() - tp)).max())


def main():
    detector = joblib.load(DETECTOR)
    cross = joblib.load(CROSS)
    reference = joblib.load(REFERENCE)
    splits = read_json(SPLITS)
    pieces = [(ORIGINAL, 811, splits["calibration"])] + [
        (DATA / f"operational_calibration_expansion_seed{seed}", seed, list(range(24)))
        for seed in CALIBRATION_SEEDS]

    parts = []
    for directory, seed, scenarios in pieces:
        print("scoring", directory.name, flush=True)
        data = CampaignData(directory)
        arrays, _ = specialist_bank(data, scenarios, reference)
        predicted = detector["mixture"].predict(arrays["X"])
        entry = {key: np.asarray(arrays[key]) for key in ("labels", "families", "timestep", "node")}
        entry["scenario"] = np.asarray(arrays["scenario"]) + seed * 1000
        entry["source"] = np.full(len(entry["labels"]), seed)
        entry["experts"] = predicted["experts"]
        entry["routing"] = predicted["routing"]
        entry["mixture"] = predicted["mixture"]
        entry["uniform"] = predicted["uniform"]
        entry["general"] = predicted["general"]
        entry["promoted"] = specialist_scores(detector, arrays["X"])
        entry["cross"] = specialist_scores(cross, arrays["X"])
        parts.append(entry)
        del data, arrays, predicted
        gc.collect()
    c = {key: np.concatenate([part[key] for part in parts], axis=0) for key in parts[0]}
    del parts
    gc.collect()

    labels = c["labels"] > 0
    families = c["families"]
    clean = (families == 0) & ~labels
    result = {"scope": "calibration only: 3 original scenarios plus seeds 12811-15811",
              "rows": int(len(labels)), "clean_fpr_budget": CLEAN_BUDGET}

    # 1. specialisation ----------------------------------------------------
    matrix, own_best = {}, {}
    for index, mechanism in enumerate(MECHANISMS):
        row = {}
        for code, family in FAMILIES.items():
            scope = families == code
            row[family] = {"oracle_f1": oracle_f1(c["experts"][scope, index], labels[scope]),
                           "auprc": float(average_precision_score(labels[scope], c["experts"][scope, index]))}
        matrix[mechanism] = row
    for code, family in FAMILIES.items():
        target = ROUTER_TARGET[code]
        ranked = sorted(MECHANISMS, key=lambda m: -matrix[m][family]["auprc"])
        own_best[family] = {"expected_expert": target, "best_expert": ranked[0],
                            "expected_is_best": ranked[0] == target,
                            "expected_auprc": matrix[target][family]["auprc"],
                            "best_auprc": matrix[ranked[0]][family]["auprc"],
                            "general_auprc": matrix["general"][family]["auprc"]}
    result["specialisation"] = {"expert_by_family": matrix, "verdict": own_best}

    # 2. routing -----------------------------------------------------------
    routing = {}
    for code, family in FAMILIES.items():
        scope = (families == code) & labels
        posterior = c["routing"][scope]
        argmax = np.bincount(posterior.argmax(1), minlength=len(MECHANISMS))
        target = MECHANISMS.index(ROUTER_TARGET[code])
        routing[family] = {"n": int(scope.sum()),
            "mean_posterior": {m: float(v) for m, v in zip(MECHANISMS, posterior.mean(0))},
            "argmax_counts": {m: int(v) for m, v in zip(MECHANISMS, argmax)},
            "accuracy": float(argmax[target] / max(1, scope.sum())),
            "mass_on_expected": float(posterior[:, target].mean())}
    clean_posterior = c["routing"][clean]
    routing["_clean"] = {"mean_posterior": {m: float(v) for m, v in
                                            zip(MECHANISMS, clean_posterior.mean(0))},
                         "general_argmax_share": float((clean_posterior.argmax(1) == 0).mean())}
    result["routing"] = routing

    # 3. does routing pay? -------------------------------------------------
    contribution = {}
    for name, score in (("mixture", c["mixture"]), ("uniform", c["uniform"]),
                        ("general", c["general"])):
        threshold = quantile_threshold(score, clean, CLEAN_BUDGET)
        decision = score > threshold
        row = {}
        for code, family in FAMILIES.items():
            scope = families == code
            y, p = labels[scope], decision[scope]
            tp = int(np.sum(y & p)); fp = int(np.sum(~y & p)); fn = int(np.sum(y & ~p))
            row[family] = 2 * tp / max(1, 2 * tp + fp + fn)
        row["_worst_family"] = min(row.values())
        row["_macro"] = float(np.mean(list(row.values())[:len(FAMILIES)]))
        contribution[name] = row
    result["routing_contribution"] = contribution

    # 4. cross-mechanism discipline ----------------------------------------
    discipline = {}
    for label, scores in (("promoted", c["promoted"]), ("cross_mechanism", c["cross"])):
        entry = {}
        for column, owner in ((0, "drift"), (1, "noise")):
            pooled = forward_max(scores[:, column], c, DELTA)
            threshold = quantile_threshold(pooled, clean, CLEAN_BUDGET)
            decision = pooled > threshold
            own = {"clean_fpr": float(decision[clean].mean())}
            for code, family in FAMILIES.items():
                scope = (families == code) & ~labels
                own[f"fp_rate_in_{family}_events"] = float(decision[scope].mean())
                own[f"fp_count_in_{family}_events"] = int(decision[scope].sum())
            own["own_family_f1"] = oracle_f1(
                pooled[families == (3 if owner == "drift" else 4)],
                labels[families == (3 if owner == "drift" else 4)])
            entry[owner] = own
        discipline[label] = entry
    result["cross_mechanism_discipline"] = discipline

    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_json(OUT, result)
    print(json.dumps({"specialisation_verdict": {k: v["expected_is_best"] for k, v in own_best.items()},
                      "router_accuracy": {k: round(v["accuracy"], 3) for k, v in routing.items()
                                          if not k.startswith("_")},
                      "routing_contribution": {k: {"worst": round(v["_worst_family"], 4),
                                                   "macro": round(v["_macro"], 4)}
                                               for k, v in contribution.items()}}, indent=2), flush=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
