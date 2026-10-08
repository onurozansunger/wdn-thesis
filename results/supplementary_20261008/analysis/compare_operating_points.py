"""Calibration-first retrospective comparison of already fitted detectors.

Reads only lossless extracted hybrid scores and existing LightGBM predictions.
No model loading, fitting, feature extraction, generation or original-file writes.
Run --stage freeze, then --stage calibrate/evaluate --network N --seed S.
Evaluation waits for all twenty immutable calibration_selection.json records.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CAMPAIGN = ROOT / "runs/operational/early_warning_multiseed_v1"
BASELINE = CAMPAIGN / "single_model_baseline_v1"
EXT = CAMPAIGN / "final_ten_model_seeds_v1"
EXTRACTED = HERE / "hybrid_scores"
OUT = HERE / "operating_points"
FAMILIES = {1: "random", 2: "replay", 3: "drift", 4: "noise", 5: "targeted"}
BRANCHES = ("general", "drift", "noise")
META = ("labels", "families", "source", "scenario", "timestep", "node")
CAPS = (0., .0001, .00025, .0005, .001, .0025, .005)
LAMBDAS = np.unique(np.r_[0., np.geomspace(.01, 100., 81), 1.])
OBJECTIVES = ("pooled", "balanced")
SEEDS = tuple(range(701, 711))
NETWORKS = ("modena", "ltown")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write_frozen(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise RuntimeError(f"Frozen analysis record differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def original_sources(network, role):
    if role == "evaluation":
        return [int(s) for s in read(EXT / "final_results.json")["evaluation_sources"][network]]
    manifest = read(CAMPAIGN / "features" / f"{network}_calibration" / "manifest.json")
    return [int(p["seed"]) for p in manifest["pieces"]]


def freeze():
    # Only source identifiers, never metric values, are read to define the policy.
    policy = {
        "version": 1,
        "study": "retrospective inference-only operating-point sensitivity and frozen-rule diagnostics",
        "model_seeds": list(SEEDS), "networks": list(NETWORKS),
        "evaluation_sources": {n: original_sources(n, "evaluation") for n in NETWORKS},
        "calibration_sources": {n: original_sources(n, "calibration") for n in NETWORKS},
        "calibration_clean_fpr_caps": list(CAPS), "lambda_grid": LAMBDAS.tolist(),
        "hybrid_path": "Frozen gates and General routing; scale all original raw branch clean budgets by one lambda. Thresholds are original strict-greater clean empirical quantiles. Original thresholds retained exactly at lambda=1 after reproduction check.",
        "original_budgets": "Modena: .005 times original candidate shares. L-Town: final history mixture_clean_budget for General, .005 times original Stage-E candidate specialist shares.",
        "zero_lambda": "Zero calibration clean budget; threshold nextafter(maximum clean score,+infinity), not necessarily no positive predictions.",
        "objectives": {"pooled": ["maximum pooled F1", "maximum worst-family F1", "minimum clean FPR", "stricter threshold/lambda"],
                       "balanced": ["maximum worst-family F1", "maximum pooled F1", "minimum clean FPR", "stricter threshold/lambda"]},
        "baseline_selection": "Exact tied-score calibration sweep, both 0h and 3h fitted classifiers; strict > semantics. 3h is primary matched maximum decision-time allowance; 0h is contextual.",
        "baseline_display_path": "Clean calibration quantiles at .005 times the same lambda grid, plus separately reported selected and original points. Display path does not restrict exact baseline calibration selection.",
        "guard_policy": "Original replay veto, verifier cutoffs and verifier training candidate region remain fixed. Branch removals remove alarm contributions, not upstream feature generation. No separate early-warning effect is inferred.",
        "diagnostics": ["original", "general_only", "general_plus_drift", "general_plus_noise", "specialists_only", "without_verifier", "without_router_feedback", "without_both_guards"],
        "aggregation": "Equal-weight means over 10 model seeds x 6 shared sources; model and source means reported separately. Partial results never represented as complete. No independent-cell significance claim.",
        "limits": ["Threshold path fixes branch allocation and all gate settings; not an exhaustive hybrid frontier.",
                   "No invented hybrid probability or intrinsic hybrid AUPRC.",
                   "Calibration caps do not guarantee the same achieved evaluation FPR.",
                   "Already-used evaluation sources; not fresh independent confirmation.",
                   "Frozen-rule removals are diagnostic conditional interventions, not retrained ablations."],
        "locked_test_read": False, "fitting_permitted": False, "new_data_permitted": False,
    }
    record = {"policy": policy, "policy_sha256": hashlib.sha256(canonical(policy)).hexdigest()}
    write_frozen(OUT / "protocol.json", record)
    return record


def require_protocol():
    record = read(OUT / "protocol.json")
    if hashlib.sha256(canonical(record["policy"])).hexdigest() != record["policy_sha256"]:
        raise RuntimeError("Protocol hash mismatch")
    if record != freeze():
        raise RuntimeError("Implemented policy differs from frozen protocol")
    return record


def report(decision, arrays):
    y = np.asarray(arrays["labels"]) > 0
    f = np.asarray(arrays["families"])
    clean = f == 0
    assert not np.any(y & clean)
    result = {}
    for code, name in [(None, "_overall"), *FAMILIES.items()]:
        mask = np.ones(len(y), bool) if code is None else f == code
        tp = int(np.sum(mask & y & decision)); fp = int(np.sum(mask & ~y & decision))
        fn = int(np.sum(mask & y & ~decision)); tn = int(np.sum(mask & ~y & ~decision))
        result[name] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                        "f1": 2 * tp / max(1, 2 * tp + fp + fn)}
    o = result["_overall"]
    o.update(clean_rows=int(clean.sum()), clean_period_fp=int(np.sum(clean & decision)),
             clean_fpr=float(np.mean(decision[clean])),
             all_negative_fpr=o["fp"] / max(1, o["fp"] + o["tn"]),
             family_macro_f1=float(np.mean([result[n]["f1"] for n in FAMILIES.values()])),
             worst_family_f1=min(result[n]["f1"] for n in FAMILIES.values()))
    return result


def flat(r):
    o = r["_overall"]
    return {**{n: r[n]["f1"] for n in FAMILIES.values()}, "pooled_f1": o["f1"],
            "clean_fpr": o["clean_fpr"], "all_negative_fpr": o["fp"] / max(1, o["fp"] + o["tn"]),
            "family_macro_f1": float(np.mean([r[n]["f1"] for n in FAMILIES.values()])),
            "worst_family_f1": min(r[n]["f1"] for n in FAMILIES.values())}


def compare_counts(actual, expected):
    for name in ("_overall", *FAMILIES.values()):
        for k in ("tp", "fp", "fn", "tn", "clean_rows", "clean_period_fp"):
            if k in expected[name] and actual[name][k] != expected[name][k]:
                raise AssertionError(f"Original count mismatch: {name}/{k}")


def quantiles(values, budgets):
    values = np.sort(np.asarray(values, float))
    assert len(values) and np.isfinite(values).all()
    positions = np.floor((1 - np.clip(budgets, 0, 1)) * len(values)).astype(np.int64)
    positions = np.clip(positions, 0, len(values) - 1)
    out = values[positions].copy()
    out[np.asarray(budgets) <= 0] = np.nextafter(values[-1], np.inf)
    return out


def crossing(scores, thresholds, allow=None):
    """First index at which strict score > decreasing threshold becomes true."""
    thresholds = np.asarray(thresholds, float)
    if np.any(np.diff(thresholds) > 0):
        raise AssertionError("Threshold path must be nonincreasing")
    idx = np.searchsorted(-thresholds, -np.asarray(scores), side="right").astype(np.int16)
    if allow is not None:
        idx[~allow] = len(thresholds)
    return idx


def path_reports(first, arrays, length):
    """Histogram first crossings once; all path confusion counts are cumulative."""
    y = arrays["labels"] > 0; f = arrays["families"]
    results = [{} for _ in range(length)]
    for code, name in [(None, "_overall"), *FAMILIES.items()]:
        mask = np.ones(len(y), bool) if code is None else f == code
        pos = mask & y; neg = mask & ~y
        tp = np.cumsum(np.bincount(first[pos], minlength=length + 1))[:length]
        fp = np.cumsum(np.bincount(first[neg], minlength=length + 1))[:length]
        positives = int(pos.sum()); negatives = int(neg.sum())
        for k in range(length):
            t = int(tp[k]); p = int(fp[k]); miss = positives - t
            results[k][name] = {"tp": t, "fp": p, "fn": miss, "tn": negatives-p,
                                 "f1": 2*t/max(1, 2*t+p+miss)}
    clean = f == 0
    cf = np.cumsum(np.bincount(first[clean], minlength=length+1))[:length]
    for k, r in enumerate(results):
        o = r["_overall"]
        o.update(clean_rows=int(clean.sum()), clean_period_fp=int(cf[k]), clean_fpr=float(cf[k]/clean.sum()))
        o.update(all_negative_fpr=o["fp"]/max(1, o["fp"]+o["tn"]),
                 family_macro_f1=float(np.mean([r[n]["f1"] for n in FAMILIES.values()])),
                 worst_family_f1=min(r[n]["f1"] for n in FAMILIES.values()))
    return results


def hybrid_reports(arrays, thresholds):
    first = crossing(arrays["general"], thresholds["general"])
    for b in ("drift", "noise"):
        first = np.minimum(first, crossing(arrays[b], thresholds[b], arrays[f"allow_{b}"]))
    return path_reports(first, arrays, len(thresholds["general"]))


def choose_path(reports, thresholds):
    result = []
    for cap in CAPS:
        allowed = [i for i, r in enumerate(reports) if r["_overall"]["clean_fpr"] <= cap]
        if not allowed:
            raise RuntimeError(f"No hybrid operating point at clean cap {cap}")
        for objective in OBJECTIVES:
            def key(i):
                o = reports[i]["_overall"]
                first, second = ((o["f1"], o["worst_family_f1"]) if objective == "pooled" else
                                 (o["worst_family_f1"], o["f1"]))
                return first, second, -o["clean_fpr"], -i
            best = max(allowed, key=key)
            result.append({"cap": cap, "objective": objective, "lambda_index": best,
                           "lambda": float(LAMBDAS[best]),
                           "thresholds": {b: float(thresholds[b][best]) for b in BRANCHES},
                           "calibration": reports[best]})
    return result


def exact_baseline_selection(score, arrays):
    order = np.argsort(-score, kind="stable"); s = score[order]
    y = arrays["labels"][order] > 0; f = arrays["families"][order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s)-1]
    tp = np.cumsum(y, dtype=np.int64)[ends]
    pooled = np.r_[0., 2*tp/np.maximum(1, int(y.sum())+ends+1)]
    worst = np.ones(len(ends))
    for code in FAMILIES:
        mask = f == code
        p = np.cumsum(mask & y, dtype=np.int64)[ends]
        pred = np.cumsum(mask, dtype=np.int64)[ends]
        worst = np.minimum(worst, 2*p/np.maximum(1, int(np.sum(mask & y))+pred))
    worst = np.r_[0., worst]
    fpr = np.r_[0., np.cumsum(f == 0, dtype=np.int64)[ends]/int(np.sum(f == 0))]
    ts = np.r_[float(s[0]), np.nextafter(s[ends], -np.inf)]
    chosen = []
    for cap in CAPS:
        allowed = np.flatnonzero(fpr <= cap)
        for objective in OBJECTIVES:
            a, b = (pooled, worst) if objective == "pooled" else (worst, pooled)
            # Ascending threshold is NOT the tie preference: higher is stricter.
            best = allowed[np.lexsort((ts[allowed], -fpr[allowed], b[allowed], a[allowed]))[-1]]
            chosen.append({"cap": cap, "objective": objective, "threshold": float(ts[best]),
                           "calibration": report(score > ts[best], arrays)})
    return chosen


def original_budgets(network, seed):
    base = CAMPAIGN if seed <= 705 else EXT
    stage = base / f"stage_e_{network}/seed/{seed}/operating_points.json"
    stage_rule = read(stage)["arms"]["candidate"]["rule"]
    budgets = dict(zip(BRANCHES, [.005*x for x in stage_rule["budget_shares"]]))
    rule_paths = [stage]
    if network == "ltown":
        path = base / f"protected_history_system_v1/seed/{seed}/full_history_selection.json"
        rule = read(path)["rule"]
        budgets["general"] = rule["mixture_clean_budget"]
        # Specialist thresholds must truly have remained unchanged.
        for b in ("drift", "noise"):
            assert rule["specialist_thresholds"][b] == stage_rule["thresholds"][b]
        rule_paths.append(path)
    return budgets, {str(p.relative_to(ROOT)): sha(p) for p in rule_paths}


def load_hybrid(network, seed, role, source):
    path = EXTRACTED / network / f"seed{seed}" / f"{role}_source{source}.npz"
    meta = read(path.with_suffix(".json"))
    if sha(path) != meta["npz_sha256"]:
        raise RuntimeError(f"Extracted score hash mismatch: {path}")
    if not meta["baseline_endpoints_aligned"]:
        raise RuntimeError("Extractor has not verified baseline endpoint alignment")
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    return arrays, meta


def align(a, b):
    for name in META:
        if not np.array_equal(a[name], b[name]):
            raise AssertionError(f"Endpoint alignment differs: {name}")


def diagnostics(a, thresholds):
    raw = {b: a[b] > thresholds[b] for b in BRANCHES}
    guarded = {"general": raw["general"], **{b: raw[b] & a[f"allow_{b}"] for b in ("drift", "noise")}}
    variants = {"original": tuple(BRANCHES), "general_only": ("general",),
                "general_plus_drift": ("general", "drift"), "general_plus_noise": ("general", "noise"),
                "specialists_only": ("drift", "noise")}
    out = {n: report(np.logical_or.reduce([guarded[b] for b in bs]), a) for n, bs in variants.items()}
    no_verifier = raw["general"] | (raw["drift"] & ~a["veto_drift"]) | (raw["noise"] & ~a["veto_noise"])
    out["without_verifier"] = report(no_verifier, a)
    # Extracted allow = ~veto & verifier_allowed; verifier probabilities retain the
    # original candidate-region behavior, including probability 1 outside it.
    cutoffs = thresholds.get("verifier_cutoffs")
    if cutoffs is not None:
        no_veto = raw["general"].copy()
        for b in ("drift", "noise"):
            no_veto |= raw[b] & (a[f"verifier_{b}"] >= cutoffs[b])
        out["without_router_feedback"] = report(no_veto, a)
    out["without_both_guards"] = report(np.logical_or.reduce(list(raw.values())), a)
    compare_counts(out["original"], report(a["original_decision"], a))
    return out


def calibrate(network, seed):
    protocol = require_protocol()
    dest = OUT / network / f"seed{seed}" / "calibration_selection.json"
    if dest.exists():
        saved = read(dest)
        assert saved["policy_sha256"] == protocol["policy_sha256"]
        assert saved["analysis_code_sha256"] == sha(__file__)
        print(f"REUSE calibration {network} {seed}", flush=True)
        return
    validation_path = EXTRACTED / network / f"seed{seed}" / "calibration_validation.json"
    validation = read(validation_path)
    if not validation["original_pooled_report_matches"]:
        raise RuntimeError("Original pooled calibration reproduction did not pass")
    parts = []; metas = []
    for source in protocol["policy"]["calibration_sources"][network]:
        a, m = load_hybrid(network, seed, "calibration", source)
        parts.append(a); metas.append(m)
    arrays = {k: np.concatenate([a[k] for a in parts]) for k in parts[0]}
    del parts; gc.collect()
    with np.load(BASELINE / "cache" / f"{network}_calibration" / "metadata.npz") as z:
        baseline_meta = {k: z[k] for k in META}
    align(arrays, baseline_meta); del baseline_meta
    clean = arrays["families"] == 0
    original = metas[0]["thresholds"]
    assert all(m["thresholds"] == original for m in metas)
    budgets, rule_hashes = original_budgets(network, seed)
    thresholds = {b: quantiles(arrays[b][clean], LAMBDAS*budgets[b]) for b in BRANCHES}
    anchor = int(np.flatnonzero(LAMBDAS == 1.)[0])
    for b in BRANCHES:
        if thresholds[b][anchor] != original[b]:
            raise AssertionError(f"Original threshold reproduction differs: {network}/{seed}/{b}")
        thresholds[b][anchor] = original[b]
    hybrid = hybrid_reports(arrays, thresholds)
    compare_counts(hybrid[anchor], validation["counts"])
    baseline = {}; input_hashes = dict(rule_hashes)
    for delta in (0, 3):
        path = BASELINE / network / str(seed) / f"delta{delta}_calibration_scores.npz"
        input_hashes[str(path.relative_to(ROOT))] = sha(path)
        with np.load(path) as z:
            score = z["score"]
        saved = read(BASELINE / network / str(seed) / f"delta{delta}_selection.json")
        old = {}
        for objective, point in saved["objectives"].items():
            actual = report(score > point["threshold"], arrays)
            compare_counts(actual, point["calibration"])
            old[objective] = {"threshold": point["threshold"], "calibration": actual}
        display = quantiles(score[clean], .005*LAMBDAS)
        baseline[f"delta{delta}"] = {"selected": exact_baseline_selection(score, arrays), "original": old,
                                     "display_thresholds": display.tolist(),
                                     "display_calibration": path_reports(crossing(score, display), arrays, len(display))}
        del score; gc.collect()
    record = {"network": network, "model_seed": seed, "policy_sha256": protocol["policy_sha256"],
              "analysis_code_sha256": sha(__file__), "evaluation_read_for_selection": False,
              "calibration_validation_sha256": sha(validation_path), "source_npz_sha256": validation["source_npz_sha256"],
              "original_rule_and_baseline_score_sha256": input_hashes,
              "hybrid": {"original_raw_budgets": budgets, "original_thresholds": original,
                         "path_thresholds": {b: t.tolist() for b, t in thresholds.items()},
                         "path_calibration": hybrid, "selected": choose_path(hybrid, thresholds), "original": hybrid[anchor]},
              "baseline": baseline, "locked_test_read": False, "fit_called": False}
    write_frozen(dest, record)
    print(f"CALIBRATION FROZEN {network} {seed}", flush=True)


def evaluate(network, seed, requested_source=None):
    protocol = require_protocol()
    # Freeze every comparison operating point before reading any new evaluation
    # analysis outcome; extraction's original-count checks are a separate stage.
    missing = []
    for n in NETWORKS:
        for s in SEEDS:
            calibration = OUT / n / f"seed{s}" / "calibration_selection.json"
            if not calibration.exists():
                missing.append(f"{n}/{s}")
                continue
            frozen = read(calibration)
            if frozen["policy_sha256"] != protocol["policy_sha256"] or frozen["analysis_code_sha256"] != sha(__file__):
                raise RuntimeError(f"Calibration policy/code identity differs: {calibration}")
    if missing:
        raise RuntimeError("All 20 calibration selections must be frozen before evaluation: " + ", ".join(missing))
    if requested_source is not None and requested_source not in protocol["policy"]["evaluation_sources"][network]:
        raise ValueError("Requested source is outside the frozen evaluation population")
    folder = OUT / network / f"seed{seed}"
    selection_path = folder / "calibration_selection.json"
    selection = read(selection_path)  # Must exist before any evaluation scores are loaded.
    assert selection["policy_sha256"] == protocol["policy_sha256"]
    assert selection["analysis_code_sha256"] == sha(__file__)
    for source in protocol["policy"]["evaluation_sources"][network]:
        if requested_source is not None and source != requested_source:
            continue
        target = folder / f"evaluation_source{source}.json"
        if target.exists():
            saved = read(target)
            assert saved["calibration_selection_sha256"] == sha(selection_path)
            print(f"REUSE evaluation {network} {seed} {source}", flush=True)
            continue
        arrays, meta = load_hybrid(network, seed, "evaluation", source)
        hybrid = hybrid_reports(arrays, selection["hybrid"]["path_thresholds"])
        anchor = int(np.flatnonzero(LAMBDAS == 1.)[0])
        compare_counts(hybrid[anchor], meta["original_counts"])
        chosen = [{**{k: p[k] for k in ("cap", "objective", "lambda", "lambda_index")},
                   "metrics": hybrid[p["lambda_index"]]} for p in selection["hybrid"]["selected"]]
        basepath = BASELINE / network / f"evaluation_predictions_{source}.npz"
        baseline = {}
        original_rows = read(BASELINE / network / f"evaluation_source_{source}.json")["rows"]
        with np.load(basepath, allow_pickle=False) as z:
            align(arrays, {k: z[k] for k in META})
            for delta in (0, 3):
                score = z[f"seed{seed}_delta{delta}"]
                cal = selection["baseline"][f"delta{delta}"]
                old = {}
                for objective, point in cal["original"].items():
                    actual = report(score > point["threshold"], arrays)
                    previous = next(r for r in original_rows if r["model_seed"] == seed and r["delay_hours"] == delta and r["objective"] == objective)
                    compare_counts(actual, previous["metrics"])
                    old[objective] = actual
                baseline[f"delta{delta}"] = {"original": old,
                    "selected": [{"cap": p["cap"], "objective": p["objective"], "threshold": p["threshold"],
                                  "metrics": report(score > p["threshold"], arrays)} for p in cal["selected"]],
                    "path": path_reports(crossing(score, cal["display_thresholds"]), arrays, len(LAMBDAS))}
                del score; gc.collect()
        diag = diagnostics(arrays, {**meta["thresholds"], "verifier_cutoffs": meta["verifier_cutoffs"]})
        record = {"network": network, "model_seed": seed, "source_seed": source,
                  "policy_sha256": protocol["policy_sha256"], "calibration_selection_sha256": sha(selection_path),
                  "hybrid_source_npz_sha256": meta["npz_sha256"], "baseline_source_npz_sha256": sha(basepath),
                  "original_counts_reproduced": True,
                  "hybrid": {"original": hybrid[anchor], "path": hybrid, "selected": chosen},
                  "baseline": baseline, "frozen_rule_diagnostics": diag,
                  "locked_test_read": False, "fit_called": False}
        write_frozen(target, record)
        print(f"EVALUATED {network} {seed} {source}", flush=True)
        del arrays; gc.collect()


def aggregate():
    protocol = require_protocol(); output = {}
    for network in NETWORKS:
        expected = [(s, d) for s in SEEDS for d in protocol["policy"]["evaluation_sources"][network]]
        missing = []; records = []
        for seed, source in expected:
            path = OUT / network / f"seed{seed}" / f"evaluation_source{source}.json"
            if not path.exists():
                missing.append({"model_seed": seed, "source_seed": source})
            else:
                value = read(path)
                assert value["policy_sha256"] == protocol["policy_sha256"]
                records.append(value)
        info = {"status": "complete" if not missing else "partial", "expected_cells": len(expected),
                "completed_cells": len(records), "missing_cells": missing}
        # Do not publish means of a convenient partial subset as campaign results.
        if missing:
            output[network] = info
            continue
        groups = {}
        for r in records:
            def add(key, metrics):
                groups.setdefault(key, []).append({"model_seed": r["model_seed"], "source_seed": r["source_seed"], **flat(metrics)})
            add("hybrid_original", r["hybrid"]["original"])
            for p in r["hybrid"]["selected"]:
                add(f"hybrid_{p['objective']}_cap{p['cap']:g}", p["metrics"])
            for i, p in enumerate(r["hybrid"]["path"]):
                add(f"hybrid_path_{i}", p)
            for delta, base in r["baseline"].items():
                for obj, p in base["original"].items(): add(f"lightgbm_{delta}_original_{obj}", p)
                for p in base["selected"]: add(f"lightgbm_{delta}_{p['objective']}_cap{p['cap']:g}", p["metrics"])
                for i, p in enumerate(base["path"]): add(f"lightgbm_{delta}_path_{i}", p)
            for name, p in r["frozen_rule_diagnostics"].items(): add(f"diagnostic_{name}", p)
        summary = {}
        for key, rows in groups.items():
            assert len(rows) == 60
            metrics = [k for k in rows[0] if k not in ("model_seed", "source_seed")]
            mean = lambda rs: {k: float(np.mean([r[k] for r in rs])) for k in metrics}
            by_model = {str(s): mean([r for r in rows if r["model_seed"] == s]) for s in SEEDS}
            by_source = {str(s): mean([r for r in rows if r["source_seed"] == s]) for s in protocol["policy"]["evaluation_sources"][network]}
            summary[key] = {"mean": mean(rows), "by_model": by_model, "by_source": by_source,
                            "model_mean_sd": {k: float(np.std([r[k] for r in by_model.values()], ddof=1)) for k in metrics},
                            "source_mean_sd": {k: float(np.std([r[k] for r in by_source.values()], ddof=1)) for k in metrics}}
        info["groups"] = summary
        paired = {}
        for cap in CAPS:
            for objective in OBJECTIVES:
                hkey = f"hybrid_{objective}_cap{cap:g}"
                for delta in (0, 3):
                    bkey = f"lightgbm_delta{delta}_{objective}_cap{cap:g}"
                    left = {(r["model_seed"], r["source_seed"]): r for r in groups[hkey]}
                    right = {(r["model_seed"], r["source_seed"]): r for r in groups[bkey]}
                    assert left.keys() == right.keys()
                    metrics = [k for k in next(iter(left.values())) if k not in ("model_seed", "source_seed")]
                    changes = [{"model_seed": pair[0], "source_seed": pair[1],
                                **{k: left[pair][k]-right[pair][k] for k in metrics}} for pair in sorted(left)]
                    paired[f"hybrid_minus_delta{delta}_{objective}_cap{cap:g}"] = {
                        "mean_difference": {k: float(np.mean([r[k] for r in changes])) for k in metrics},
                        "by_source": {str(s): {k: float(np.mean([r[k] for r in changes if r["source_seed"] == s])) for k in metrics}
                                      for s in protocol["policy"]["evaluation_sources"][network]},
                        "by_model": {str(s): {k: float(np.mean([r[k] for r in changes if r["model_seed"] == s])) for k in metrics} for s in SEEDS},
                        "higher_cell_count": {k: sum(r[k] > 0 for r in changes) for k in metrics},
                        "equal_cell_count": {k: sum(r[k] == 0 for r in changes) for k in metrics},
                    }
        info["paired_comparisons"] = paired
        output[network] = info
    result = {"policy_sha256": protocol["policy_sha256"], "networks": output,
              "status": "complete" if all(n["status"] == "complete" for n in output.values()) else "partial",
              "shared_sources_not_independent_cells": True}
    # Progress is intentionally mutable; completed source/selection records are frozen.
    path = OUT / "aggregation_progress.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] == "complete": write_frozen(OUT / "complete_summary.json", result)
    print(json.dumps({n: {k: v for k, v in x.items() if k not in ("groups", "paired_comparisons")} for n, x in output.items()}), flush=True)


def self_check():
    # Ties, vetoed rows, duplicate thresholds and never-crossing rows.
    score = np.array([10., 8., 7., 6., -1.]); ts = np.array([10., 8., 8., 6.])
    allow = np.array([True, True, False, True, True])
    idx = crossing(score, ts, allow)
    for i, t in enumerate(ts): assert np.array_equal(idx <= i, (score > t) & allow)
    a = {"labels": np.array([1, 1, 0, 0, 0]), "families": np.array([1, 2, 0, 3, 0])}
    curves = path_reports(idx, a, len(ts))
    for i, t in enumerate(ts): compare_counts(curves[i], report((score > t) & allow, a))
    q = quantiles(np.array([0., .1, .2, .3]), np.array([0., .25, .5, 1.]))
    assert q[0] > .3 and np.array_equal(q[1:], [.3, .2, 0.])
    # Exact tied-score calibration selection checked against brute-force search.
    toy = np.array([.9, .8, .8, .1, .1])
    exact = exact_baseline_selection(toy, a)
    options = np.r_[toy.max(), np.nextafter(np.unique(toy), -np.inf)]
    for selected in exact:
        candidates = []
        for t in options:
            m = report(toy > t, a); o = m["_overall"]
            if o["clean_fpr"] <= selected["cap"]:
                first, second = ((o["f1"], o["worst_family_f1"]) if selected["objective"] == "pooled" else (o["worst_family_f1"], o["f1"]))
                candidates.append(((first, second, -o["clean_fpr"], t), t))
        assert selected["threshold"] == max(candidates)[1]
    assert len(np.flatnonzero(LAMBDAS == 1.)) == 1
    print("SELF CHECK PASS: strict ties, guards, cumulative counts, exact threshold optimization, lambda anchor", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True, choices=("freeze", "self-check", "calibrate", "evaluate", "aggregate"))
    p.add_argument("--network", choices=NETWORKS)
    p.add_argument("--seed", type=int, choices=SEEDS)
    p.add_argument("--source", type=int)
    args = p.parse_args()
    if args.stage == "self-check": self_check(); return
    if args.stage == "freeze":
        print(json.dumps(freeze(), indent=2)); return
    if args.stage == "aggregate": aggregate(); return
    if args.network is None or args.seed is None: p.error("This stage requires --network and --seed")
    if args.stage == "calibrate": calibrate(args.network, args.seed)
    else: evaluate(args.network, args.seed, args.source)


if __name__ == "__main__":
    main()
