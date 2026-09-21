"""Protected calibration and one fresh confirmation of received-history General.

The prior TRAIN-only rejection is preserved. Five existing internal components,
three top-level branches, and the existing three-hour specialized delay remain.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import sklearn

import run_campaign as rc
from build_feature_cache import (CAMPAIGN, ROOT, CORPORA, atomic_npz, build,
                                 check_distribution, config_of, load_corpus, sha, write_json)
from replicate_observed_history import frozen_write
from screen_observed_history import build_bank
from screen_shared_history import load
from screen_router_temperature import TEMPERATURES, UNIFORM_SHRINKAGE, transformed_mixture
from wdn.latency_deployment import FAMILY_NAMES, family_scores
from wdn.shared_history import SharedHistoryExpertMixture, fit_presampled

OUT = CAMPAIGN / "protected_history_system_v1"
STAGE = CAMPAIGN / "stage_e_ltown/seed"
SELECTION = CAMPAIGN / "router_temperature_screen_v1/ltown/seed"
SEEDS = (701, 702, 703, 704, 705)
FRESH = (110811, 111811, 112811, 113811, 114811, 115811)
PURPOSE = "ltown_protected_history_confirmation"
BUDGETS = tuple(sorted(set((0., .000025, .00005, .000075) + tuple(np.linspace(.0001, .005, 25)))))
PROTECTED = ("random", "drift", "noise", "targeted")
VARIANTS = ("full_history", "selective_history")


def existing_rule(seed):
    return json.loads((SELECTION / str(seed) / "selection_frozen.json").read_text())["rule"]


def reserve():
    from reserve_seeds import observed_seeds, KNOWN_FORBIDDEN
    path = CAMPAIGN / "seed_manifest.json"
    record = json.loads(path.read_text())
    if PURPOSE in record["generator_seeds"]:
        if record["generator_seeds"][PURPOSE] != list(FRESH):
            raise RuntimeError("Fresh reservation changed")
        return
    occupied = set(observed_seeds()) | set(KNOWN_FORBIDDEN)
    occupied |= {s for seeds in record["generator_seeds"].values() for s in seeds}
    if occupied & set(FRESH):
        raise RuntimeError("Fresh confirmation seed collision")
    record["generator_seeds"][PURPOSE] = list(FRESH)
    record.setdefault("amendments", []).append({"purpose": PURPOSE, "seeds": FRESH,
        "reserved_before_generation": True, "reason": "Protected received-history full-system confirmation, once after design freeze"})
    write_json(path, record)


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    reserve()
    files = sorted((ROOT / "src/wdn").rglob("*.py"))
    files += sorted(Path(__file__).parent.glob("*.py"))
    files += [CAMPAIGN / "stage_e_ltown/reference.joblib", ROOT / "data/L-Town.inp"]
    benchmark = config_of(ROOT / "data/thesis_v2/ew_ltown_ltown_train_seed60811")
    sources = {}
    for corpus in ("ltown_train", "ltown_calibration"):
        path = CAMPAIGN / "features" / corpus / "manifest.json"
        files.append(path)
        manifest = json.loads(path.read_text())
        sources[corpus] = manifest["pieces"]
        for piece in manifest["pieces"]:
            directory = ROOT / "data/thesis_v2" / piece["directory"]
            check_distribution(directory, benchmark)
            cache = ROOT / piece["cache"]
            if sha(cache) != piece["cache_sha256"]:
                raise RuntimeError("Canonical feature cache changed")
            files.append(cache)
            files.extend(directory / name for name in ("generate_config.yaml", "corrupted.pkl", "snapshots.pkl"))
    for seed in SEEDS:
        files.extend(STAGE / str(seed) / name for name in ("base_bundle.joblib", "delayed_bundle.joblib", "heads_bundle.joblib"))
        files.append(SELECTION / str(seed) / "selection_frozen.json")
    signature = {"training_seeds": SEEDS, "sources": sources,
        "variants": VARIANTS, "full_history": "Same 42 common received-history features, all five internal components and internal router refitted on full TRAIN with the deployed component seed (600 + training seed), same sample and objective.",
        "selective_history": "Replace only internal general and replay components with their full-history versions; retain baseline abrupt/drift/noise components and baseline internal router. Still exactly five internal components.",
        "sequence": "Evaluate full_history on all five seeds. If it is not feasible on all seeds, or mean calibration replay F1 is below 0.80, also evaluate selective_history. Choose one globally feasible variant for all seeds by mean worst-family F1, then mean pooled F1. Never choose a variant separately per seed.",
        "selection_grid": {"temperatures": TEMPERATURES, "uniform_shrinkage": UNIFORM_SHRINKAGE, "general_clean_budgets": BUDGETS},
        "selection_constraints": {"other_family_max_f1_drop": .005, "pooled_max_drop": 0., "clean_fpr_max_increase": 0., "absolute_clean_fpr_max": .005,
            "replay_must_improve": True, "per_source_other_family_max_drop": .01,
            "per_source_pooled_max_drop": .005, "per_source_clean_fpr_max_increase": .0001,
            "per_source_replay_max_drop": .01},
        "selection_objective": "Maximize worst-family F1, then pooled F1, then lower clean FPR, on calibration only; fixed specialist thresholds/verifier cutoffs/external router/feedback.",
        "fresh_confirmation": {"purpose": PURPOSE, "generator_seeds": FRESH, "scenarios_per_seed": 24, "runs": 30,
            "policy": "Generate/read only after all five model/rule selections and the single architecture variant are frozen. Evaluate every paired baseline/candidate once. No post-confirmation tuning or second attempt.",
            "success": "Protected improvement: positive mean replay delta in every fresh source, >=24/30 positive paired replay changes; overall mean pooled delta >=0, each other-family mean delta >=-0.005, mean clean-FPR delta <=0. Per-source protection as in calibration. Replay target separately requires mean full-system replay F1 >=0.80; report all below-target runs."},
        "architecture": {"top_level": ["General", "Drift", "Noise"], "general_internal_components": ["general", "abrupt", "replay", "drift", "noise"], "specialist_delay_hours": 3},
        "prior_train_rejection_preserved": True, "missing_pressure": .5, "missing_flow": .5,
        "locked_test_read": False, "old_confirmation_read": False,
        "versions": {"python": sys.version, "numpy": np.__version__, "sklearn": sklearn.__version__, "joblib": joblib.__version__},
        "input_code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in files}}
    frozen_write(OUT / "protocol_frozen.json", signature)
    print("Protected full-system protocol frozen and verified", flush=True)


def fit(seed):
    output = OUT / "seed" / str(seed)
    output.mkdir(parents=True, exist_ok=True)
    frozen_write(output / "signature.json", {"seed": seed, "protocol_sha256": sha(OUT / "protocol_frozen.json")})
    if (output / "history.joblib").exists():
        return
    print(seed, "loading full TRAIN", flush=True)
    train, manifest = load_corpus("ltown_train")
    names = train.pop("_feature_names")
    baseline = joblib.load(STAGE / str(seed) / "base_bundle.joblib")["mixture"]
    if baseline.seed != 600 + seed or list(baseline.names) != names:
        raise RuntimeError("Unexpected deployed baseline recipe")
    positive, negative = np.flatnonzero(train["labels"] > .5), np.flatnonzero(train["labels"] <= .5)
    chosen = np.random.default_rng(baseline.seed).choice(negative, min(60000, len(negative)), replace=False)
    rows = np.r_[positive, chosen]
    weights = np.r_[np.ones(len(positive)), np.full(len(chosen), len(negative)/len(chosen))]
    train = {k: v[rows] for k, v in train.items()}
    del rows, positive, negative, chosen
    gc.collect()
    history, added = build_bank(train, names, manifest, (0.,))
    model = SharedHistoryExpertMixture(names + added, seed=baseline.seed)
    print(seed, "fitting received-history General on full TRAIN", flush=True)
    fit_presampled(model, np.column_stack((train["X"], history[0.])), train["labels"], train["families"], weights)
    joblib.dump(model, output / "history.tmp")
    (output / "history.tmp").replace(output / "history.joblib")
    write_json(output / "fit.json", {"training_seed": seed, "component_seed": baseline.seed,
        "training_scenarios": manifest["total_scenarios"], "sampled_rows": len(weights),
        "feature_names": names+added, "history_sha256": sha(output / "history.joblib"),
        "baseline_sha256": sha(STAGE / str(seed) / "base_bundle.joblib")})


def fixed_predictions(seed, arrays, names):
    """Existing deployed specialized pipeline; General is scored separately."""
    experts = joblib.load(STAGE / str(seed) / "base_bundle.joblib")
    head = joblib.load(STAGE / str(seed) / "delayed_bundle.joblib")
    heads = joblib.load(STAGE / str(seed) / "heads_bundle.joblib")
    causal, delayed = rc.apply_experts(experts, head, arrays)
    scores = rc.specialist_score_parts(causal, delayed, arrays)
    local = rc.early_local_evidence(arrays["X"], names, causal)
    early = heads["early"].predict_groups(local, arrays)
    del local
    features, found = rc.verifier_features(arrays["X"], names, scores, early, arrays)
    if list(found) != list(heads["columns"]):
        raise RuntimeError("Verifier schema mismatch")
    keep = heads["verifier"].verify(features, scores)
    del features, early
    guard = heads["guard"].predict(rc.local_evidence(arrays["X"], names, rc.router_score_parts(scores)), arrays)
    veto = rc.specialist_veto_masks(guard["router"], guard["feedback"], .10, .10)
    rule = existing_rule(seed)
    specialist = np.zeros(len(arrays["labels"]), bool)
    for j, branch in enumerate(rc.BRANCHES):
        allowed = ~veto[:, j]
        if rule["verifier_cutoffs"][branch] > 0:
            allowed &= keep[branch] >= rule["verifier_cutoffs"][branch]
        specialist |= allowed & (scores[f"final_{branch}"] > rule["specialist_thresholds"][branch])
    return specialist, experts["mixture"]


def score_piece(seed, manifest, piece, target):
    if target.exists():
        return
    print(seed, "scoring frozen full system source", piece["seed"], flush=True)
    arrays = load(ROOT / piece["cache"])
    arrays["event"] = np.where(arrays["event"] >= 0, arrays["source"] * 10**6 + arrays["event"], -1)
    names = manifest["feature_names"]
    specialist, baseline = fixed_predictions(seed, arrays, names)
    history, _ = build_bank(arrays, names, manifest, (0.,))
    model = joblib.load(OUT / "seed" / str(seed) / "history.joblib")
    payload = {k: arrays[k] for k in ("labels", "families", "source", "scenario", "event")}
    payload["specialist"] = specialist
    for prefix, fitted in (("baseline", baseline), ("history", model)):
        parts = []
        for start in range(0, len(arrays["labels"]), 50000):
            X = arrays["X"][start:start+50000]
            if prefix == "history":
                X = np.column_stack((X, history[0.][start:start+50000]))
            prediction = fitted.predict(X)
            parts.append({k: prediction[k] for k in ("experts", "routing")})
        for k in ("experts", "routing"):
            payload[f"{prefix}_{k}"] = np.concatenate([p[k] for p in parts])
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_npz(target, **payload)
    del arrays, history, model, baseline, payload
    gc.collect()


def protected(candidate, baseline, source=False):
    tol = .01 if source else .005
    return (all(candidate[f]["f1"] >= baseline[f]["f1"]-tol for f in PROTECTED)
        and candidate["_overall"]["f1"] >= baseline["_overall"]["f1"]-(.005 if source else 0.)
        and candidate["_overall"]["clean_fpr"] <= baseline["_overall"]["clean_fpr"]+(.0001 if source else 0.)
        and candidate["_overall"]["clean_fpr"] <= .005
        and (candidate["replay"]["f1"] >= baseline["replay"]["f1"]-.01 if source
             else candidate["replay"]["f1"] > baseline["replay"]["f1"]))


def threshold_reports(score, thresholds, arrays):
    """Exact OR-with-specialist counts at many thresholds, avoiding repeated scans."""
    labels, families, specialist = arrays["labels"] > 0, arrays["families"], arrays["specialist"]
    reports = [{} for _ in thresholds]
    for code, family in [(None, "_overall")] + list(FAMILY_NAMES.items()):
        scope = np.ones(len(score), bool) if code is None else families == code
        pos, neg = scope & labels, scope & ~labels
        positives = int(pos.sum())
        tp0, fp0 = int(np.sum(pos & specialist)), int(np.sum(neg & specialist))
        ps = np.sort(score[pos & ~specialist]); ns = np.sort(score[neg & ~specialist])
        tp = tp0 + len(ps)-np.searchsorted(ps, thresholds, side="right")
        fp = fp0 + len(ns)-np.searchsorted(ns, thresholds, side="right")
        for j, r in enumerate(reports):
            r[family] = {"tp": int(tp[j]), "fp": int(fp[j]), "fn": int(positives-tp[j]),
                         "f1": float(2*tp[j]/max(1, positives+tp[j]+fp[j]))}
    clean = families == 0
    cs = np.sort(score[clean & ~specialist])
    clean_fp = int(np.sum(clean & specialist)) + len(cs)-np.searchsorted(cs, thresholds, side="right")
    for j, r in enumerate(reports):
        r["_overall"].update(clean_fpr=float(clean_fp[j]/clean.sum()), clean_period_fp=int(clean_fp[j]),
            all_negative_fpr=float(r["_overall"]["fp"]/(~labels).sum()),
            worst_family_f1=min(r[f]["f1"] for f in FAMILY_NAMES.values()))
    return reports


def variant_scores(arrays, variant):
    if variant == "full_history":
        return arrays["history_experts"], arrays["history_routing"]
    if variant == "selective_history":
        experts = arrays["baseline_experts"].copy()
        experts[:, [0, 2]] = arrays["history_experts"][:, [0, 2]]
        return experts, arrays["baseline_routing"]
    raise ValueError(variant)


def selection(seed, variant):
    target = OUT / "seed" / str(seed) / f"{variant}_selection.json"
    if target.exists():
        return json.loads(target.read_text())
    manifest = json.loads((CAMPAIGN / "features/ltown_calibration/manifest.json").read_text())
    parts = []
    for piece in manifest["pieces"]:
        path = OUT / "seed" / str(seed) / "calibration" / f"source_{piece['seed']}.npz"
        score_piece(seed, manifest, piece, path)
        parts.append(load(path))
    arrays = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    del parts
    rule0 = existing_rule(seed)
    bs = transformed_mixture(arrays["baseline_experts"], arrays["baseline_routing"], rule0["router_temperature"], rule0["uniform_shrinkage"])
    baseline_decision = arrays["specialist"] | (bs > rule0["mixture_threshold"])
    baseline = threshold_reports(bs, [rule0["mixture_threshold"]], arrays)[0]
    expected = json.loads((SELECTION / str(seed) / "selection_frozen.json").read_text())["report"]
    actual = rc.compact(baseline)
    for k, v in expected.items():
        if abs(actual[k]-v) > 1e-8:
            raise RuntimeError(f"Paired baseline calibration reproduction failed: {seed} {k} {actual[k]} != {v}")
    sources = np.unique(arrays["source"])
    source_rows = {int(s): np.flatnonzero(arrays["source"] == s) for s in sources}
    source_base = {s: family_scores(baseline_decision[rows], {k: arrays[k][rows] for k in ("labels", "families")}) for s, rows in source_rows.items()}
    experts, routing = variant_scores(arrays, variant)
    clean = arrays["families"] == 0
    best, feasible_count, frontier = None, 0, None
    print(seed, variant, "searching protected calibration grid", flush=True)
    for temperature in TEMPERATURES:
        for shrinkage in UNIFORM_SHRINKAGE:
            score = transformed_mixture(experts, routing, temperature, shrinkage)
            clean_score = np.sort(score[clean])
            thresholds = np.array([np.nextafter(clean_score[-1], np.inf) if b == 0 else
                clean_score[min(int(np.floor((1-b)*len(clean_score))), len(clean_score)-1)] for b in BUDGETS])
            reports = threshold_reports(score, thresholds, arrays)
            for budget, threshold, report in zip(BUDGETS, thresholds, reports):
                key = (report["_overall"]["worst_family_f1"], report["_overall"]["f1"], -report["_overall"]["clean_fpr"])
                if frontier is None or key > frontier[0]:
                    frontier = (key, report)
                if not protected(report, baseline):
                    continue
                decision = arrays["specialist"] | (score > threshold)
                source_reports = {s: family_scores(decision[rows], {k: arrays[k][rows] for k in ("labels", "families")}) for s, rows in source_rows.items()}
                if not all(protected(source_reports[s], source_base[s], source=True) for s in source_reports):
                    continue
                feasible_count += 1
                if best is None or key > best[0]:
                    rule = {**rule0, "router_temperature": temperature, "uniform_shrinkage": shrinkage,
                            "mixture_clean_budget": float(budget), "mixture_threshold": float(threshold)}
                    best = (key, rule, report, source_reports)
    result = {"training_seed": seed, "variant": variant, "status": "selected" if best else "no_feasible_improvement",
        "baseline": baseline, "baseline_sources": source_base, "feasible_points": feasible_count,
        "grid_points": len(TEMPERATURES)*len(UNIFORM_SHRINKAGE)*len(BUDGETS),
        "unconstrained_frontier": frontier[1], "selection_surface": "calibration_only"}
    if best:
        result.update(rule=best[1], report=best[2], source_reports=best[3])
    frozen_write(target, result)
    print(seed, variant, result["status"], rc.compact(result["report"]) if best else "", flush=True)
    return result


def finalize_design():
    options = {}
    for variant in VARIANTS:
        paths = [OUT / "seed" / str(s) / f"{variant}_selection.json" for s in SEEDS]
        if not all(p.exists() for p in paths):
            continue
        results = [json.loads(p.read_text()) for p in paths]
        if all(r["status"] == "selected" for r in results):
            options[variant] = (float(np.mean([r["report"]["_overall"]["worst_family_f1"] for r in results])),
                                float(np.mean([r["report"]["_overall"]["f1"] for r in results])))
    variant = max(options, key=options.get) if options else None
    files = [OUT / "protocol_frozen.json"]
    if variant:
        for seed in SEEDS:
            folder = OUT / "seed" / str(seed)
            baseline = joblib.load(STAGE / str(seed) / "base_bundle.joblib")["mixture"]
            history = joblib.load(folder / "history.joblib")
            selected = [history if variant == "full_history" or j in (0, 2) else baseline for j in range(5)]
            router_model = history if variant == "full_history" else baseline
            bundle = {"names": history.names, "experts": [m.experts[j] for j, m in enumerate(selected)],
                      "profiles": [m.profiles[j] for j, m in enumerate(selected)], "router": router_model.router,
                      "router_columns": np.arange(len(router_model.names)), "variant": variant,
                      "top_level_branch": "General", "internal_component_count": 5}
            path = folder / "candidate_general.joblib"
            if not path.exists():
                joblib.dump(bundle, path.with_suffix(".tmp"))
                path.with_suffix(".tmp").replace(path)
            files += [folder / "history.joblib", folder / f"{variant}_selection.json", path]
    frozen_write(OUT / "design_frozen.json", {"status": "qualified" if variant else "retain_baseline",
        "variant": variant, "feasible_options": options, "fresh_seeds": FRESH,
        "hashes": {str(p.relative_to(ROOT)): sha(p) for p in files}, "confirmation_read": False})
    return variant


def check_design():
    design = json.loads((OUT / "design_frozen.json").read_text())
    if design["status"] != "qualified":
        raise RuntimeError("No qualified design for confirmation")
    for p, digest in design["hashes"].items():
        if sha(ROOT / p) != digest:
            raise RuntimeError("Frozen design changed before confirmation")
    protocol = json.loads((OUT / "protocol_frozen.json").read_text())
    for p, digest in protocol["input_code_hashes"].items():
        if sha(ROOT / p) != digest:
            raise RuntimeError(f"Frozen input/code changed: {p}")
    return design


def prepare_confirmation():
    check_design()
    for seed in FRESH:
        corpus = f"ltown_protected_history_confirmation_seed{seed}"
        directory = f"ew_ltown_{PURPOSE}_seed{seed}"
        subprocess.run([sys.executable, str(Path(__file__).with_name("generate_corpus.py")),
                        "--network", "ltown", "--purpose", PURPOSE, "--seed", str(seed), "--scenarios", "24"], cwd=ROOT, check=True)
        CORPORA[corpus] = {"reference": CAMPAIGN / "stage_e_ltown/reference.joblib", "network": "ltown", "pieces": [(directory, seed, "all:24")]}
        if not (CAMPAIGN / "features" / corpus / "manifest.json").exists():
            build(corpus)


def confirm(seed):
    design = check_design()
    variant = design["variant"]
    selection_file = OUT / "seed" / str(seed) / f"{variant}_selection.json"
    rule = json.loads(selection_file.read_text())["rule"]
    rule0 = existing_rule(seed)
    output = OUT / "seed" / str(seed) / "confirmation.json"
    if output.exists():
        return
    results = []
    for source in FRESH:
        manifest = json.loads((CAMPAIGN / "features" / f"ltown_protected_history_confirmation_seed{source}" / "manifest.json").read_text())
        piece = manifest["pieces"][0]
        cache = OUT / "seed" / str(seed) / "confirmation" / f"source_{source}.npz"
        score_piece(seed, manifest, piece, cache)
        arrays = load(cache)
        e, r = variant_scores(arrays, variant)
        baseline = transformed_mixture(arrays["baseline_experts"], arrays["baseline_routing"], rule0["router_temperature"], rule0["uniform_shrinkage"])
        candidate = transformed_mixture(e, r, rule["router_temperature"], rule["uniform_shrinkage"])
        results.append({"source": source, "baseline": threshold_reports(baseline, [rule0["mixture_threshold"]], arrays)[0],
                        "candidate": threshold_reports(candidate, [rule["mixture_threshold"]], arrays)[0]})
        del arrays, baseline, candidate, e, r
        gc.collect()
    frozen_write(output, {"seed": seed, "variant": variant, "design_sha256": sha(OUT / "design_frozen.json"), "pairs": results})


def report():
    design = json.loads((OUT / "design_frozen.json").read_text())
    output = {"design": design, "calibration": {}, "locked_test_read": False,
              "previous_independent_replay_f1": .6757}
    for variant in VARIANTS:
        paths = [OUT / "seed" / str(s) / f"{variant}_selection.json" for s in SEEDS]
        if all(p.exists() for p in paths):
            output["calibration"][variant] = [json.loads(p.read_text()) for p in paths]
    paths = [OUT / "seed" / str(s) / "confirmation.json" for s in SEEDS]
    if all(p.exists() for p in paths):
        pairs = [p | {"training_seed": s} for s, path in zip(SEEDS, paths) for p in json.loads(path.read_text())["pairs"]]
        keys = list(FAMILY_NAMES.values()) + ["pooled_f1", "clean_fpr"]
        def val(p, arm, k):
            return p[arm]["_overall"]["f1" if k == "pooled_f1" else k] if k in ("pooled_f1", "clean_fpr") else p[arm][k]["f1"]
        metrics = {k: {"baseline_mean": float(np.mean([val(p, "baseline", k) for p in pairs])),
            "candidate_mean": float(np.mean([val(p, "candidate", k) for p in pairs])),
            "mean_delta": float(np.mean([val(p, "candidate", k)-val(p, "baseline", k) for p in pairs])),
            "candidate_min": min(val(p, "candidate", k) for p in pairs),
            "candidate_max": max(val(p, "candidate", k) for p in pairs)} for k in keys}
        source_deltas = {str(s): {k: float(np.mean([val(p, "candidate", k)-val(p, "baseline", k) for p in pairs if p["source"] == s])) for k in keys} for s in FRESH}
        wins = sum(val(p, "candidate", "replay") > val(p, "baseline", "replay") for p in pairs)
        checks = {"replay_24_of_30_wins": wins >= 24, "replay_each_source_positive": all(d["replay"] > 0 for d in source_deltas.values()),
            "pooled_mean_preserved": metrics["pooled_f1"]["mean_delta"] >= 0,
            "clean_fpr_mean_preserved": metrics["clean_fpr"]["mean_delta"] <= 0,
            "other_family_mean_protection": all(metrics[f]["mean_delta"] >= -.005 for f in PROTECTED),
            "absolute_clean_fpr_budget": all(val(p, "candidate", "clean_fpr") <= .005 for p in pairs),
            "source_protection": all(d["pooled_f1"] >= -.005 and d["clean_fpr"] <= .0001 and all(d[f] >= -.01 for f in PROTECTED) for d in source_deltas.values())}
        output["confirmation"] = {"pairs": pairs, "metrics": metrics, "source_mean_deltas": source_deltas,
            "checks": checks, "protected_improvement_confirmed": all(checks.values()),
            "replay_target_0_80_reached": metrics["replay"]["candidate_mean"] >= .8,
            "replay_wins": wins, "replay_runs_below_0_80": sum(val(p, "candidate", "replay") < .8 for p in pairs),
            "independence": "Six new generator sources; five model seeds per source share scenarios, not 30 independent datasets. No confirmation-based model/rule tuning."}
    write_json(OUT / "summary.json", output)
    print(json.dumps({k:v for k,v in output.items() if k != "calibration" and k != "confirmation"}, indent=2), flush=True)
    if "confirmation" in output:
        print(json.dumps({k:v for k,v in output["confirmation"].items() if k != "pairs"}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "freeze", "fit", "calibrate", "design", "data", "confirm", "report"), default="all")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--variant", choices=VARIANTS, default="full_history")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.stage in ("freeze", "all"):
        freeze()
    if args.stage == "fit":
        fit(args.seed)
    elif args.stage == "calibrate":
        selection(args.seed, args.variant)
    elif args.stage == "design":
        finalize_design()
    elif args.stage == "data":
        prepare_confirmation()
    elif args.stage == "confirm":
        confirm(args.seed)
    elif args.stage == "report":
        report()
    elif args.stage == "all":
        def batch(stage, variant="full_history"):
            def worker(seed):
                logdir = OUT / "logs"
                logdir.mkdir(exist_ok=True)
                with (logdir / f"{stage}_{variant}_{seed}.log").open("a") as log:
                    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--stage", stage,
                                    "--seed", str(seed), "--variant", variant], cwd=ROOT,
                        env=dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                        stdout=log, stderr=subprocess.STDOUT, check=True)
                print("Completed", stage, variant, seed, flush=True)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(worker, SEEDS))
        batch("fit")
        batch("calibrate")
        full = [json.loads((OUT / "seed" / str(s) / "full_history_selection.json").read_text()) for s in SEEDS]
        if not all(r["status"] == "selected" for r in full) or np.mean([r["report"]["replay"]["f1"] for r in full]) < .8:
            batch("calibrate", "selective_history")
        if finalize_design():
            prepare_confirmation()
            batch("confirm")
        report()


if __name__ == "__main__":
    main()
