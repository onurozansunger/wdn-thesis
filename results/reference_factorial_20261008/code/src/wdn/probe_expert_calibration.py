"""Calibrate expert evidence on normal scores; optimise family balance on calibration.

The score is max(expert normal-tail evidence + learned offset). Offsets never
depend on the true inference family. One global threshold controls the final
decision. This is a separately reported fusion/calibration variant, not the
original soft router.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import optuna

from wdn.operational_calibration import calibrate_threshold, family_summary
from wdn.train_operational_moe import summarise


class NormalTailEvidence:
    def fit(self, normal_scores):
        normal_scores = np.asarray(normal_scores)
        if normal_scores.ndim != 2 or not len(normal_scores) or not np.isfinite(normal_scores).all():
            raise ValueError("Finite normal calibration score matrix required")
        self.normal_ = np.sort(normal_scores, axis=0)
        return self

    def transform(self, scores):
        scores = np.asarray(scores)
        if scores.ndim != 2 or scores.shape[1] != self.normal_.shape[1] or not np.isfinite(scores).all():
            raise ValueError("Expert score dimensions do not match calibration")
        columns = []
        for i in range(scores.shape[1]):
            normal = self.normal_[:, i]
            # Includes ties in the upper tail and a finite-sample pseudocount.
            tail = (len(normal)-np.searchsorted(normal, scores[:, i], side="left")+1)/(len(normal)+1)
            columns.append(-np.log(tail))
        return np.column_stack(columns)


def probe(run_dir, output_dir, trials=40):
    source, output = Path(run_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    cal = dict(np.load(source/"features_calibration.npz"))
    c = dict(np.load(source/"predictions_calibration.npz"))
    tail = NormalTailEvidence().fit(c["experts"][cal["labels"] == 0])
    evidence = tail.transform(c["experts"])
    jobs = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=619, n_startup_trials=10))
    jobs.enqueue_trial({f"offset_{i}": 0. for i in range(1, evidence.shape[1])})
    def objective(trial):
        offsets = np.r_[0., [trial.suggest_float(f"offset_{i}", -3., 3.) for i in range(1, evidence.shape[1])]]
        score = (evidence+offsets).max(1)
        point = calibrate_threshold(score, cal["labels"], cal["families"], max_fpr=.005)
        trial.set_user_attr("selection", point)
        return point["calibration"]["macro_f1"]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    jobs.optimize(objective, n_trials=trials)
    best = jobs.best_trial
    offsets = np.r_[0., [best.params[f"offset_{i}"] for i in range(1, evidence.shape[1])]]
    point = best.user_attrs["selection"]
    # Only now open validation arrays: they cannot influence Optuna proposals,
    # offsets, reference distributions or threshold selection.
    val = dict(np.load(source/"features_validation.npz"))
    v = dict(np.load(source/"predictions_validation.npz"))
    score = (tail.transform(v["experts"])+offsets).max(1)
    result = summarise(score, val["labels"], val["families"], point["threshold"])
    joblib.dump({"normal_tail": tail, "offsets": offsets, "threshold": point["threshold"]}, output/"calibration.joblib")
    report = {"test_evaluated": False, "source_run": str(source), "trials": trials,
              "selected_by": "calibration macro F1; validation loaded after selection",
              "inference": "max normal-tail expert evidence with fixed offsets, one global threshold",
              "selection": point, "offsets": offsets.tolist(), "validation": result,
              **family_summary(result), "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "input_sha256": {name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                               for name in ("features_calibration.npz", "predictions_calibration.npz")}}
    np.savez_compressed(output/"scores.npz", calibration=(evidence+offsets).max(1), validation=score)
    (output/"summary.json").write_text(json.dumps(report, indent=2))
    (output/"trials.json").write_text(json.dumps([{"number": t.number, "score": t.value, "params": t.params,
        "selection": t.user_attrs["selection"]} for t in jobs.trials], indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", default="runs/operational/family_balance_v1")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--trials", type=int, default=40)
    from wdn.probe_expert_calibration import probe as run_probe
    run_probe(**vars(parser.parse_args()))
