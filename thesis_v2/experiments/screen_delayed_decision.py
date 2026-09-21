"""TRAIN-only score-level screen for a bounded decision latency.

Second stage only, and deliberately cheap: it reads the frozen generator-held
OOF expert scores from `runs/operational/seasonal_family_experts_v3` and asks
whether anything can be recovered from the expert's *output* alone once the
hourly decision for time t may be finalised at t+delta.

    frozen      the promoted expert score, no latency
    maxpool     score'_t = max(score_t .. score_{t+delta})
    learned     LightGBM over the causal history and the bounded future of the
                frozen scores, cross-fitted leave-one-source-out

Neighbouring hours are matched by timestep, never by position in the series:
with a 50% missing rate most sensor series are gappy.

The feature-level counterpart, which is the one that matters, is
`screen_delayed_decision_head.py`. No reference is refitted, no feature bank is
rebuilt, no data is generated, no threshold is deployed, and calibration,
validation, the locked test and the locked EVAL seeds are not read. Reported F1
values are hindsight optima over the score ordering on held sources.

    python3 thesis_v2/experiments/screen_delayed_decision.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from wdn.delayed_decision_features import offset_index

ROOT = Path(__file__).resolve().parents[2]
OOF = ROOT / "runs/operational/seasonal_family_experts_v3/oof_predictions.npz"
OUT = ROOT / "thesis_v2/outputs/delayed_decision_screen.json"
DELTAS = (0, 1, 2, 3, 4, 6)
HISTORY = 3
FAMILIES = {"drift": (3, 0), "noise": (4, 1)}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gather(values, index, column):
    """`values` at the neighbour named by `column`, NaN where that hour is absent."""
    out = np.full(values.shape, np.nan)
    present = index[:, column] >= 0
    out[present] = values[index[present, column]]
    return out


def oracle_f1(score, label):
    """Best F1 over every threshold of this fixed ordering. Hindsight only."""
    positives = int(label.sum())
    if not positives:
        raise ValueError("A family with no positives cannot be scored")
    order = np.argsort(-score, kind="stable")
    y = label[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    f1 = 2 * tp / (2 * tp + fp + positives - tp)
    best = int(f1.argmax())
    return {"f1": float(f1[best]), "recall": float(tp[best] / positives),
            "precision": float(tp[best] / (tp[best] + fp[best])),
            "tp": int(tp[best]), "fp": int(fp[best]), "positives": positives}


def main():
    import lightgbm as lgb

    data = np.load(OOF, allow_pickle=True)
    label, family = data["labels"].astype(np.int8), data["families"]
    source = data["source"]
    arrays = {key: data[key] for key in ("source", "scenario", "node", "timestep")}
    raw = data["scores"]
    sources = np.unique(source)

    result = {"scope": "expanded TRAIN generator-held OOF, score level only",
              "oof_sha256": sha(OOF), "history_hours": HISTORY,
              "held_sources": sources.tolist(),
              "neighbour_matching": "by timestep within the sensor series",
              "metric": "family-conditioned hindsight F1 over the score ordering",
              "calibration_evaluated": False, "validation_evaluated": False,
              "test_evaluated": False, "locked_eval_evaluated": False, "families": {}}

    for name, (code, column) in FAMILIES.items():
        rows = family == code
        train_rows = rows | (family == 0)
        entry = {}
        for delta in DELTAS:
            offsets = list(range(-HISTORY, delta + 1))
            index = offset_index(arrays, offsets)
            columns, names = [], []
            for label_name, source_column in (("own", column), ("other", 1 - column)):
                for position, offset in enumerate(offsets):
                    columns.append(raw[:, source_column] if offset == 0
                                   else gather(raw[:, source_column], index, position))
                    names.append(f"{label_name}_t{offset:+d}")
            future = np.column_stack([raw[:, column]] + [
                gather(raw[:, column], index, offsets.index(k))
                for k in range(1, delta + 1)])
            with np.errstate(invalid="ignore"):
                pooled = np.nanmax(future, axis=1)
                columns += [pooled, np.nanmean(future, axis=1), pooled - raw[:, column]]
            names += ["future_max", "future_mean", "future_rise"]
            features = np.column_stack(columns)

            learned = np.full(raw.shape[0], np.nan)
            for held in sources:
                fit = train_rows & (source != held)
                model = lgb.LGBMClassifier(
                    n_estimators=300, learning_rate=0.05, num_leaves=31,
                    min_child_samples=50, subsample=0.8, subsample_freq=1,
                    colsample_bytree=0.8, random_state=811, verbose=-1)
                model.fit(features[fit], label[fit])
                mask = source == held
                learned[mask] = model.predict_proba(features[mask])[:, 1]

            entry[str(delta)] = {
                "frozen": oracle_f1(raw[rows, column], label[rows]),
                "maxpool": oracle_f1(pooled[rows], label[rows]),
                "learned": oracle_f1(learned[rows], label[rows]),
                "n_features": len(names)}
            print(name, "delta", delta, "frozen %.4f maxpool %.4f learned %.4f" % (
                entry[str(delta)]["frozen"]["f1"], entry[str(delta)]["maxpool"]["f1"],
                entry[str(delta)]["learned"]["f1"]), flush=True)
        result["families"][name] = entry

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
