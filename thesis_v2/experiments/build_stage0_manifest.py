"""Stage 0 sizing manifest for the fresh operational seeds.

Counts, per generator seed and per family, how many scenarios carry an event
and how many observed pressure rows are labelled positive, then hashes every
dataset file. This is dataset metadata only: no model is loaded and no score
is computed, so it does not consume the locked EVAL seeds.

    python3 thesis_v2/experiments/build_stage0_manifest.py
"""
from __future__ import annotations

import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/thesis_v2"
OUT = ROOT / "thesis_v2/outputs/stage0_manifest.json"
ROLES = {"locked_eval": [4811, 5811, 6811, 7811, 8811, 9811],
         "train_expansion2": [10811, 11811],
         "calibration_expansion": [12811]}
DIRNAME = {"locked_eval": "operational_eval_seed{}",
           "train_expansion2": "operational_train_expansion2_seed{}",
           "calibration_expansion": "operational_calibration_expansion_seed{}"}
FAMILY = {1: "random", 2: "replay", 3: "stealthy", 4: "noise", 5: "targeted"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def summarise(directory: Path):
    corrupted = pickle.loads((directory / "corrupted.pkl").read_bytes())
    base = pickle.loads((directory / "snapshots.pkl").read_bytes())
    scenarios = defaultdict(set)
    positives = defaultdict(int)
    for snapshot, source in zip(corrupted, base):
        code = int(snapshot.attack_type_id)
        if code in FAMILY:
            scenarios[FAMILY[code]].add(int(source.scenario_id))
            positives[FAMILY[code]] += int(snapshot.pressure_anomaly.sum())
    return ({name: len(ids) for name, ids in scenarios.items()},
            {name: count for name, count in positives.items()},
            len({int(s.scenario_id) for s in base}))


def main():
    result = {"generator_bounds": "identical to operational_v1 train expansion; "
                                  "only the seed differs",
              "roles": {}, "totals": {}, "files": {}}
    totals = {role: {"scenarios": 0, "event_scenarios": defaultdict(int),
                     "positive_pressure_rows": defaultdict(int)} for role in ROLES}
    for role, seeds in ROLES.items():
        entries = {}
        for seed in seeds:
            directory = DATA / DIRNAME[role].format(seed)
            counts, positives, scenarios = summarise(directory)
            entries[str(seed)] = {"scenarios": scenarios, "event_scenarios": counts,
                                  "positive_pressure_rows": positives}
            totals[role]["scenarios"] += scenarios
            for name, value in counts.items():
                totals[role]["event_scenarios"][name] += value
            for name, value in positives.items():
                totals[role]["positive_pressure_rows"][name] += value
            result["files"][directory.name] = {
                path.name: sha(path) for path in sorted(directory.iterdir())}
        result["roles"][role] = entries
    result["totals"] = {role: {"scenarios": value["scenarios"],
                               "event_scenarios": dict(value["event_scenarios"]),
                               "positive_pressure_rows": dict(value["positive_pressure_rows"])}
                        for role, value in totals.items()}
    eval_totals = result["totals"]["locked_eval"]
    result["stage0_gate"] = {
        "min_event_scenarios_per_family": min(eval_totals["event_scenarios"].values()),
        "min_positive_rows_per_family": min(eval_totals["positive_pressure_rows"].values()),
        "requires_event_scenarios": 40, "requires_positive_rows": 800,
        "passes": bool(min(eval_totals["event_scenarios"].values()) >= 40
                       and min(eval_totals["positive_pressure_rows"].values()) >= 800)}
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"totals": result["totals"], "stage0_gate": result["stage0_gate"]}, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
