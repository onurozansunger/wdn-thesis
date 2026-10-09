"""Pure endpoint metrics for the prespecified reference factorial.

No fitting, file access, model inference, or protocol mutation. The threshold
and cumulative-count semantics match the retained operating-point comparison.
Run this file directly for small deterministic numerical checks only.
"""
from __future__ import annotations

import numpy as np

FAMILIES = {1: "random", 2: "replay", 3: "drift", 4: "noise", 5: "targeted"}
BRANCHES = ("general", "drift", "noise")
CAPS = (.0005, .005)
OBJECTIVES = ("pooled", "balanced")
LAMBDAS = np.unique(np.r_[0., np.geomspace(.01, 100., 81), 1.])
CONDITIONS = ("e0r0", "e0r1", "e1r0", "e1r1")


def _labels(arrays):
    y = np.asarray(arrays["labels"]) > 0
    f = np.asarray(arrays["families"])
    if y.ndim != 1 or f.shape != y.shape:
        raise ValueError("Expected aligned one-dimensional labels and families")
    if not np.any(f == 0) or np.any(y & (f == 0)):
        raise ValueError("Family-0 clean rows must exist and contain no positives")
    return y, f


def report(decision, arrays):
    y, f = _labels(arrays)
    decision = np.asarray(decision, dtype=bool)
    if decision.shape != y.shape:
        raise ValueError("Decision and labels differ in shape")
    result = {}
    for code, name in [(None, "_overall"), *FAMILIES.items()]:
        mask = np.ones(len(y), bool) if code is None else f == code
        tp = int(np.sum(mask & y & decision))
        fp = int(np.sum(mask & ~y & decision))
        fn = int(np.sum(mask & y & ~decision))
        tn = int(np.sum(mask & ~y & ~decision))
        result[name] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                        "f1": 2 * tp / max(1, 2 * tp + fp + fn)}
    clean = f == 0
    o = result["_overall"]
    o.update(clean_rows=int(clean.sum()),
             clean_period_fp=int(np.sum(clean & decision)),
             clean_fpr=float(np.mean(decision[clean])),
             all_negative_fpr=o["fp"] / max(1, o["fp"] + o["tn"]),
             family_macro_f1=float(np.mean([result[n]["f1"] for n in FAMILIES.values()])),
             worst_family_f1=min(result[n]["f1"] for n in FAMILIES.values()))
    return result


def flat(value):
    o = value["_overall"]
    return {**{n: value[n]["f1"] for n in FAMILIES.values()},
            "pooled_f1": o["f1"], "clean_fpr": o["clean_fpr"],
            "all_negative_fpr": o["fp"] / max(1, o["fp"] + o["tn"]),
            "family_macro_f1": float(np.mean([value[n]["f1"] for n in FAMILIES.values()])),
            "worst_family_f1": min(value[n]["f1"] for n in FAMILIES.values())}


def quantiles(values, budgets):
    values = np.sort(np.asarray(values, dtype=float))
    budgets = np.asarray(budgets, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite clean score vector")
    if budgets.ndim != 1 or not np.isfinite(budgets).all() or np.any(budgets < 0):
        raise ValueError("Expected finite nonnegative budget vector")
    positions = np.floor((1 - np.clip(budgets, 0, 1)) * len(values)).astype(np.int64)
    positions = np.clip(positions, 0, len(values) - 1)
    out = values[positions].copy()
    out[budgets <= 0] = np.nextafter(values[-1], np.inf)
    return out


def crossing(scores, thresholds, allow=None):
    """First index at which a strict score > descending threshold holds."""
    scores = np.asarray(scores, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Expected a finite score vector")
    if (thresholds.ndim != 1 or not len(thresholds)
            or np.isnan(thresholds).any() or np.any(np.diff(thresholds) > 0)):
        raise ValueError("Expected nonincreasing thresholds without NaNs")
    idx = np.searchsorted(-thresholds, -scores, side="right").astype(np.int64)
    if allow is not None:
        allow = np.asarray(allow, dtype=bool)
        if allow.shape != scores.shape:
            raise ValueError("Score and allow-mask shapes differ")
        idx[~allow] = len(thresholds)
    return idx


def path_reports(first, arrays, length):
    """Cumulative confusion counts from one first-crossing histogram."""
    y, f = _labels(arrays)
    first = np.asarray(first)
    if (first.shape != y.shape or not np.issubdtype(first.dtype, np.integer)
            or np.any(first < 0) or np.any(first > length)):
        raise ValueError("Invalid first-crossing indices")
    results = [{} for _ in range(length)]
    for code, name in [(None, "_overall"), *FAMILIES.items()]:
        mask = np.ones(len(y), bool) if code is None else f == code
        pos = mask & y
        neg = mask & ~y
        tp = np.cumsum(np.bincount(first[pos], minlength=length + 1))[:length]
        fp = np.cumsum(np.bincount(first[neg], minlength=length + 1))[:length]
        positives = int(pos.sum())
        negatives = int(neg.sum())
        for k in range(length):
            t = int(tp[k]); p = int(fp[k]); miss = positives - t
            results[k][name] = {"tp": t, "fp": p, "fn": miss,
                                "tn": negatives - p,
                                "f1": 2 * t / max(1, 2 * t + p + miss)}
    clean = f == 0
    clean_fp = np.cumsum(np.bincount(first[clean], minlength=length + 1))[:length]
    for k, value in enumerate(results):
        o = value["_overall"]
        o.update(clean_rows=int(clean.sum()), clean_period_fp=int(clean_fp[k]),
                 clean_fpr=float(clean_fp[k] / int(clean.sum())),
                 all_negative_fpr=o["fp"] / max(1, o["fp"] + o["tn"]),
                 family_macro_f1=float(np.mean([value[n]["f1"] for n in FAMILIES.values()])),
                 worst_family_f1=min(value[n]["f1"] for n in FAMILIES.values()))
    return results


def hybrid_path_reports(arrays, thresholds, allow=None):
    """OR of General and two guarded branches across a frozen threshold path.

    ``arrays`` contains general/drift/noise, labels and families. Supply either
    ``allow={"drift": mask, "noise": mask}`` or arrays' allow_drift/allow_noise.
    A true allow value permits, rather than vetoes, that specialist branch.
    """
    lengths = {len(thresholds[b]) for b in BRANCHES}
    if len(lengths) != 1:
        raise ValueError("Branch threshold path lengths differ")
    first = crossing(arrays["general"], thresholds["general"])
    for branch in ("drift", "noise"):
        permitted = arrays[f"allow_{branch}"] if allow is None else allow[branch]
        first = np.minimum(first, crossing(arrays[branch], thresholds[branch], permitted))
    return path_reports(first, arrays, lengths.pop())


def select(reports, thresholds, caps=CAPS, objectives=OBJECTIVES, lambdas=LAMBDAS):
    lambdas = np.asarray(lambdas, dtype=float)
    if (len(reports) != len(lambdas) or any(len(thresholds[b]) != len(lambdas)
                                          for b in BRANCHES)):
        raise ValueError("Reports, lambdas and threshold paths differ in length")
    if np.any(np.diff(lambdas) < 0):
        raise ValueError("Lambda order must be nondecreasing for strict tie preference")
    result = []
    for cap in caps:
        allowed = [i for i, r in enumerate(reports) if r["_overall"]["clean_fpr"] <= cap]
        if not allowed:
            raise ValueError(f"No operating point satisfies calibration clean-FPR cap {cap}")
        for objective in objectives:
            if objective not in OBJECTIVES:
                raise ValueError(f"Unknown calibration objective: {objective}")
            def key(i):
                o = reports[i]["_overall"]
                first, second = ((o["f1"], o["worst_family_f1"]) if objective == "pooled"
                                 else (o["worst_family_f1"], o["f1"]))
                return first, second, -o["clean_fpr"], -i
            best = max(allowed, key=key)
            result.append({"cap": float(cap), "objective": objective,
                           "lambda_index": best, "lambda": float(lambdas[best]),
                           "thresholds": {b: float(thresholds[b][best]) for b in BRANCHES},
                           "calibration": reports[best]})
    return result


def factorial_contrasts(values):
    """Scalar paired contrasts; apply within each seed/source cell first."""
    if set(values) != set(CONDITIONS):
        raise ValueError("A contrast requires exactly four factor conditions")
    a, b, c, d = (float(values[k]) for k in CONDITIONS)
    if not np.isfinite([a, b, c, d]).all():
        raise ValueError("Nonfinite condition value")
    return {"exclusion_when_irls_on": d - b,
            "exclusion_when_irls_off": c - a,
            "irls_when_exclusion_on": d - c,
            "irls_when_exclusion_off": b - a,
            "interaction": d - c - b + a,
            "exclusion_main_effect": .5 * ((d - b) + (c - a)),
            "irls_main_effect": .5 * ((d - c) + (b - a))}


def self_check():
    # Tied score thresholds are excluded under strict >, including duplicates.
    assert crossing([.1, .5, .5, 1.], [1., .5, .5, 0.]).tolist() == [3, 3, 3, 1]
    assert crossing([.5, 1.], [1., .5, 0.], [False, True]).tolist() == [3, 1]
    assert len(LAMBDAS) == 82 and LAMBDAS[0] == 0 and np.sum(LAMBDAS == 1.) == 1
    budgets = np.array([0., .25, .5, 1.])
    q = quantiles([.1, .5, .5, 1.], budgets)
    assert q.tolist() == [np.nextafter(1., np.inf), 1., .5, .1]
    assert np.array_equal(quantiles([.1, .5, .5, 1.], [LAMBDAS[LAMBDAS == 1.][0] * .5]), [.5])
    # Multiple tied-score populations, nontrivial guards and every path index.
    rng = np.random.default_rng(821)
    checks = 0
    for _ in range(12):
        families = np.tile(np.arange(6), 11)
        arrays = {"families": families,
                  "labels": (families > 0) & (rng.random(len(families)) > .3)}
        for b in BRANCHES:
            arrays[b] = rng.integers(0, 9, len(families)) / 8
        for b in ("drift", "noise"):
            arrays[f"allow_{b}"] = rng.random(len(families)) > .3
        thresholds = {b: quantiles(arrays[b][families == 0], LAMBDAS * .005)
                      for b in BRANCHES}
        path = hybrid_path_reports(arrays, thresholds)
        for i, value in enumerate(path):
            direct = arrays["general"] > thresholds["general"][i]
            for b in ("drift", "noise"):
                direct |= arrays[f"allow_{b}"] & (arrays[b] > thresholds[b][i])
            assert report(direct, arrays) == value
            checks += 1
        selections = select(path, thresholds)
        for selected in selections:
            cap, objective = selected["cap"], selected["objective"]
            keys = []
            for i, value in enumerate(path):
                o = value["_overall"]
                if o["clean_fpr"] <= cap:
                    a, b = ((o["f1"], o["worst_family_f1"]) if objective == "pooled"
                            else (o["worst_family_f1"], o["f1"]))
                    keys.append((a, b, -o["clean_fpr"], -i))
            assert selected["lambda_index"] == -max(keys)[-1]
    contrasts = factorial_contrasts({"e0r0": 1, "e0r1": 3, "e1r0": 4, "e1r1": 10})
    assert contrasts == {"exclusion_when_irls_on": 7., "exclusion_when_irls_off": 3.,
                         "irls_when_exclusion_on": 6., "irls_when_exclusion_off": 2.,
                         "interaction": 4., "exclusion_main_effect": 5., "irls_main_effect": 4.}
    print(f"PASS: {checks} direct-Boolean/path comparisons; strict ties, guards, quantiles, lambda anchor, selection and factorial contrasts")


if __name__ == "__main__":
    self_check()
