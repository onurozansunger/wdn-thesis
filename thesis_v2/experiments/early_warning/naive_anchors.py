"""Naive detector anchors on the *same* fresh evaluation corpora.

An F1 is uninterpretable without knowing what the task's floor and its trivial
ceiling look like. The campaign compares a candidate against the project's own
prior system; that says how much was gained, but not whether the absolute number
is good. These anchors answer the second question on exactly the corpora the
candidate was scored on:

* **always alarm** — the degenerate detector, i.e. the prevalence floor;
* **k-sigma residual rule** — the standard textbook detector, a fixed threshold
  on the standardised reference residual;
* **oracle single threshold** — the best F1 any single-threshold rule on that
  residual could achieve if its threshold were chosen with the labels in hand.
  Nothing simpler than the candidate can beat this, so it is the ceiling of the
  naive approach, not a competitor.

The oracle uses labels to pick its threshold and is therefore optimistically
biased in its own favour. That is deliberate: it makes the comparison harder for
the candidate, not easier.

    /opt/miniconda3/bin/python thesis_v2/experiments/early_warning/naive_anchors.py
"""
from __future__ import annotations

import json

import numpy as np

from build_feature_cache import CORPORA, ROOT, load_corpus, write_json

OUTPUT = ROOT / "runs/operational/early_warning_multiseed_v1/naive_anchors.json"
EVAL_SEEDS = (40811, 41811, 42811, 43811, 44811, 45811)
SIGMAS = (3, 4, 5)


def register_eval_corpora():
    reference = ROOT / "runs/operational/seasonal_family_deployment_v2/full/reference.joblib"
    for seed in EVAL_SEEDS:
        CORPORA[f"modena_eval_seed{seed}"] = {
            "reference": reference, "network": "modena",
            "pieces": [(f"ew_modena_modena_eval_seed{seed}", seed, "all:24")]}


def counts(predicted, labels):
    tp = int(np.sum(predicted & labels))
    fp = int(np.sum(predicted & ~labels))
    fn = int(np.sum(~predicted & labels))
    return {"f1": 2 * tp / max(1, 2 * tp + fp + fn), "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + fn), "tp": tp, "fp": fp, "fn": fn}


def main():
    register_eval_corpora()
    parts, labels = [], []
    for seed in EVAL_SEEDS:
        corpus, _ = load_corpus(f"modena_eval_seed{seed}",
                                columns=("abs_residual", "normal_error_scale"))
        parts.append(corpus["X"][:, 0])
        labels.append(np.asarray(corpus["labels"]) > 0)
    residual = np.concatenate(parts)
    y = np.concatenate(labels)

    result = {
        "scope": "the six fresh Modena evaluation corpora, 144 scenarios",
        "endpoints": int(len(y)), "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "residual_feature": "abs_residual, the standardised absolute reference residual",
        "negative_quantiles": {q: float(v) for q, v in zip(
            ("p50", "p90", "p99", "p999"),
            np.quantile(residual[~y], [.5, .9, .99, .999]))},
        "anchors": {"always_alarm": counts(np.ones(len(y), bool), y)},
    }
    for k in SIGMAS:
        result["anchors"][f"{k}_sigma_rule"] = counts(residual > k, y)

    # Best F1 reachable by any single threshold on this residual, labels in hand.
    order = np.argsort(-residual)
    ranked = y[order]
    cumulative = np.cumsum(ranked)
    flagged = np.arange(1, len(ranked) + 1)
    f1 = 2 * cumulative / (flagged + y.sum())
    best = int(np.argmax(f1))
    result["anchors"]["oracle_single_threshold"] = {
        "f1": float(f1[best]), "threshold": float(residual[order][best]),
        "flagged": int(flagged[best]),
        "note": "threshold chosen with the labels; an upper bound on every "
                "single-threshold residual rule, not an achievable detector"}
    result["campaign_for_comparison"] = {
        "deployed_baseline_pooled_f1": 0.6863,
        "candidate_pooled_f1": 0.8672,
        "source": "stage_e_modena evaluation reports, same six corpora"}
    write_json(OUTPUT, result)
    print(json.dumps({k: (round(v["f1"], 4) if isinstance(v, dict) and "f1" in v else v)
                      for k, v in result["anchors"].items()}, indent=2))


if __name__ == "__main__":
    main()
