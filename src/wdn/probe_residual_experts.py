"""Small learned mixture on cached blind-reference features; validation only.

This diagnostic uses tree experts and a sensor-level router. It is separate
from the temporal GNN and must not be reported as a GNN/MoE improvement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from wdn.audit_operational_experts import best_f1_threshold
from wdn.train_operational_moe import FAMILY_NAMES, _binary_counts, select_threshold, summarise


MECHANISMS = ("general", "abrupt", "replay", "drift", "noise")


def feature_profiles(names):
    common = {"residual", "abs_residual", "reference_support", "normal_error_scale", "last_gap", "residual_rate"}
    profiles = [set(names),
        common | {n for n in names if n.endswith("_4")},
        common | {n for n in names if n.startswith("lag_") or n.endswith("_16")},
        common | {n for n in names if n.startswith(("coverage_", "mean_", "slope_", "sign_consistency_"))},
        common | {n for n in names if n.startswith(("coverage_", "std_", "rms_", "max_abs_"))}]
    return [np.array([i for i, name in enumerate(names) if name in profile]) for profile in profiles]


class ResidualExpertMixture:
    def __init__(self, names, seed=601):
        self.names, self.seed = names, seed
        self.profiles = feature_profiles(names)

    def _model(self):
        return HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15,
            min_samples_leaf=30, learning_rate=.08, l2_regularization=10.,
            early_stopping=False, random_state=self.seed)

    def fit(self, X, labels, families):
        positive = labels > .5
        # Random and targeted share the same bias mechanism; topology changes
        # where the corruption occurs, not its signal-level form.
        target = np.zeros(len(labels), dtype=int)
        for mechanism, family_ids in enumerate(((1, 5), (2,), (3,), (4,)), start=1):
            target[positive & np.isin(families, family_ids)] = mechanism
        rng = np.random.default_rng(self.seed)
        negatives = np.flatnonzero(~positive)
        chosen = rng.choice(negatives, min(len(negatives), 60000), replace=False)
        subset = np.r_[np.flatnonzero(positive), chosen]
        base_weight = np.r_[np.ones(positive.sum()), np.full(len(chosen), len(negatives)/len(chosen))]
        self.experts = []
        for mechanism, cols in enumerate(self.profiles):
            domain = np.ones(len(subset), dtype=bool) if mechanism == 0 else ((target[subset] == mechanism) | ~positive[subset])
            idx, weight = subset[domain], base_weight[domain].copy()
            y = labels[idx]
            if len(np.unique(y)) != 2:
                raise ValueError(f"Missing training class for {MECHANISMS[mechanism]}")
            # Specialist risks balance positives and negatives; the general
            # control retains the original natural-prevalence objective.
            if mechanism:
                for cls in (0, 1):
                    weight[y == cls] *= len(weight)/(2*weight[y == cls].sum())
            expert = self._model().fit(X[idx][:, cols], y, sample_weight=weight)
            self.experts.append(expert)
            print("trained", MECHANISMS[mechanism], "positives", int(y.sum()), flush=True)
        route_weight = base_weight.copy()
        for mechanism in range(len(MECHANISMS)):
            selected = target[subset] == mechanism
            if not selected.any():
                raise ValueError("Missing mechanism for router")
            route_weight[selected] *= len(route_weight)/(len(MECHANISMS)*route_weight[selected].sum())
        self.router = self._model().fit(X[subset], target[subset], sample_weight=route_weight)
        return self

    def predict(self, X):
        # Inference receives features only, never attack-family labels.
        experts = np.column_stack([model.predict_proba(X[:, cols])[:, 1]
                                   for model, cols in zip(self.experts, self.profiles)])
        routing = self.router.predict_proba(X)
        return {"experts": experts, "routing": routing,
                "mixture": (experts*routing).sum(1), "general": experts[:, 0],
                "uniform": experts.mean(1)}


def probe(reference_run, output_dir):
    source, output = Path(reference_run), Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    names = json.loads((source/"feature_names.json").read_text())
    arrays = {name: dict(np.load(source/f"features_{name}.npz"))
              for name in ("train", "calibration", "validation")}
    train, cal, val = (arrays[name] for name in ("train", "calibration", "validation"))
    model = ResidualExpertMixture(names).fit(train["X"], train["labels"], train["families"])
    joblib.dump(model, output/"model.joblib")
    predictions = {name: model.predict(arrays[name]["X"]) for name in ("calibration", "validation")}
    c, v = predictions["calibration"], predictions["validation"]
    report = {"status": "tree_expert_diagnostic_not_temporal_GNN", "test_evaluated": False,
              "reference_run": str(source), "mechanisms": MECHANISMS, "results": {}, "expert_matrix": {},
              "reference_feature_sha256": {name: hashlib.sha256((source/f"features_{name}.npz").read_bytes()).hexdigest()
                                           for name in arrays},
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for name in ("general", "mixture", "uniform"):
        threshold = select_threshold(c[name], cal["labels"], cal["families"])
        report["results"][name] = summarise(v[name], val["labels"], val["families"], threshold)
        r = report["results"][name]
        print(name, "F1", r["overall"]["f1"], "replay", r["per_family"]["replay"]["f1"], flush=True)
    for expert, name in enumerate(MECHANISMS):
        threshold = best_f1_threshold(c["experts"][:, expert], cal["labels"])
        row = {}
        for family, label in enumerate(FAMILY_NAMES):
            mask = val["families"] == family
            s, y = v["experts"][mask, expert], val["labels"][mask]
            row[label] = _binary_counts(s, y, threshold)
            row[label]["auprc"] = float(average_precision_score(y, s)) if y.any() else None
        report["expert_matrix"][name] = {"threshold": threshold, "per_family": row}
    for name, prediction in predictions.items():
        np.savez_compressed(output/f"predictions_{name}.npz", **prediction)
    (output/"summary.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_run", required=True)
    parser.add_argument("--output_dir", required=True)
    # Keep saved estimator classes importable rather than pickling __main__.
    from wdn.probe_residual_experts import probe as run_probe
    run_probe(**vars(parser.parse_args()))
