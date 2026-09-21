"""Leakage-resistant multi-seed TRAIN data for weak-family experiments."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json

import joblib
import numpy as np
import yaml

from wdn.dynamic_residual_features import dynamic_residual_features
from wdn.models.blind_reference import BlindPressureReference
from wdn.models.robust_reference import RobustBlindReference
from wdn.probe_blind_reference import residual_features
from wdn.run_expert_redesign import CampaignData
from wdn.sequential_evidence import sequential_evidence


ALLOWED_CONFIG_CHANGES = frozenset({"profile", "output_dir", "seed", "num_scenarios"})


def scenario_uid(seed: int, local_scenario: int) -> int:
    """Stable, human-readable identity for one generator/scenario pair."""
    if seed < 0 or not 0 <= local_scenario < 1000:
        raise ValueError("Seed must be nonnegative and local scenario must be below 1000")
    return int(seed) * 1000 + int(local_scenario)


def distribution_changes(base: dict, candidate: dict) -> set[str]:
    """Return generator fields whose values differ, including missing fields."""
    return {key for key in set(base) | set(candidate)
            if base.get(key) != candidate.get(key)}


class ExpandedTrainData:
    """Expose original TRAIN plus complete new corpora under global identities.

    Loading the original pickle necessarily deserialises its complete file, but
    ``scenario`` only permits the original TRAIN IDs declared in the frozen
    split. No calibration, validation or test feature is extractable here.
    """

    def __init__(self, original_dir, expansion_dirs, splits):
        self.original_dir = Path(original_dir)
        self.expansion_dirs = tuple(Path(path) for path in expansion_dirs)
        original_config = yaml.safe_load((self.original_dir / "generate_config.yaml").read_text())
        self.original_seed = int(original_config["seed"])
        self.original_train = tuple(sorted(map(int, splits["train"])))
        forbidden = set(splits["calibration"] + splits["validation"] + splits["test"])
        if set(self.original_train) & forbidden:
            raise ValueError("Original TRAIN overlaps a protected split")
        if original_config["missing_rate_pressure"] != .5 or original_config["missing_rate_flow"] != .5:
            raise ValueError("Original missing probabilities are not fixed at 0.50")

        directories = (self.original_dir,) + self.expansion_dirs
        self.corpora = {}
        self.local_ids = {}
        self.source_folds = []
        self.config_audit = {}
        for position, directory in enumerate(directories):
            config = yaml.safe_load((directory / "generate_config.yaml").read_text())
            manifest = json.loads((directory / "manifest.json").read_text())
            seed = int(config["seed"])
            if seed in self.corpora:
                raise ValueError("Generator seeds must be unique")
            if config["missing_rate_pressure"] != .5 or config["missing_rate_flow"] != .5:
                raise ValueError("Expanded missing probabilities must remain 0.50")
            changes = distribution_changes(original_config, config)
            if position and not changes <= ALLOWED_CONFIG_CHANGES:
                raise ValueError(f"Expansion changes the data distribution: {sorted(changes)}")
            if position and int(config["num_scenarios"]) != 16:
                raise ValueError("Each frozen expansion seed must contribute all 16 scenarios")
            data = CampaignData(directory)
            found = sorted({int(snapshot.scenario_id) for snapshot in data.snapshots})
            expected = list(range(int(config["num_scenarios"])))
            if found != expected:
                raise ValueError(f"Incomplete scenario corpus for seed {seed}")
            allowed = list(self.original_train) if position == 0 else expected
            if position == 0 and not set(allowed).isdisjoint(forbidden):
                raise ValueError("Protected original scenarios entered expanded TRAIN")
            self.corpora[seed] = data
            self.local_ids[seed] = tuple(allowed)
            self.source_folds.append([scenario_uid(seed, sid) for sid in allowed])
            self.config_audit[str(seed)] = {
                "directory": str(directory), "changed_fields": sorted(changes),
                "scenario_count_used": len(allowed), "generated_scenario_count": len(found),
                "pressure_missing_probability": config["missing_rate_pressure"],
                "flow_missing_probability": config["missing_rate_flow"],
                "empirical_pressure_missing_rate": manifest["pressure"]["missing_rate"],
                "empirical_flow_missing_rate": manifest["flow"]["missing_rate"],
                "manifest_events": manifest["events"],
            }

        self.scenario_ids = tuple(uid for fold in self.source_folds for uid in fold)
        if len(set(self.scenario_ids)) != len(self.scenario_ids):
            raise ValueError("Global scenario identities collide")
        self._lookup = {scenario_uid(seed, sid): (seed, sid)
                        for seed, ids in self.local_ids.items() for sid in ids}
        self._scenario_cache = {}
        self.events = []
        self.events_by_scenario = defaultdict(list)
        for seed, data in self.corpora.items():
            allowed = set(self.local_ids[seed])
            for event in data.events:
                local = int(event["scenario_id"])
                if local not in allowed:
                    continue
                converted = {**event, "source_seed": seed,
                             "local_scenario_id": local,
                             "scenario_id": scenario_uid(seed, local)}
                event_id = len(self.events)
                self.events.append(converted)
                self.events_by_scenario[converted["scenario_id"]].append((event_id, converted))

    def scenario(self, uid):
        uid = int(uid)
        if uid not in self._lookup:
            raise ValueError("Scenario is outside the expanded TRAIN allowlist")
        if uid not in self._scenario_cache:
            seed, local = self._lookup[uid]
            arrays = self.corpora[seed].scenario(local)
            arrays["scenario"] = uid
            arrays["source"] = seed
            self._scenario_cache[uid] = arrays
        return self._scenario_cache[uid]

    def reference(self, scenario_ids):
        values, masks = [], []
        for uid in scenario_ids:
            arrays = self.scenario(uid)
            normal = arrays["families"] == 0
            values.append(arrays["values"][normal])
            masks.append(arrays["mask"][normal])
        values, masks = np.concatenate(values), np.concatenate(masks)
        base = BlindPressureReference(rank=16).fit(values, masks)
        return RobustBlindReference(base).calibrate_scale(values, masks)

    def features(self, scenario_ids, reference):
        keys = ("X", "labels", "families", "event", "scenario", "source", "timestep", "node")
        collected = {key: [] for key in keys}
        errors = []
        for uid in sorted(map(int, scenario_ids)):
            arrays = self.scenario(uid)
            values, mask = arrays["values"], arrays["mask"]
            prediction, support, disagreement = reference.predict_details(values, mask)
            scale = reference.noise_scale_
            base, base_names = residual_features(values, mask, prediction, support, scale)
            dynamic, dynamic_names = dynamic_residual_features(values, mask, prediction, scale)
            sequential, sequential_names = sequential_evidence(values, mask, prediction, scale)
            features = np.concatenate(
                [base, dynamic, (disagreement[15:] / scale)[..., None], sequential], axis=-1)
            names = base_names + dynamic_names + ["reference_disagreement"] + sequential_names
            endpoint = mask[15:]
            labels = arrays["labels"][15:]
            times = arrays["timestep"][15:]
            family = np.broadcast_to(arrays["families"][15:, None], endpoint.shape)
            event_grid = np.full(labels.shape, -1, dtype=np.int32)
            for event_id, event in self.events_by_scenario[uid]:
                active = ((times >= event["start_timestep"])
                          & (times < event["start_timestep"] + event["actual_steps"]))
                event_grid[active] = event_id
            if np.any(event_grid[labels > 0] < 0):
                raise ValueError("Positive label has no expanded TRAIN event identity")
            collected["X"].append(features[endpoint].astype(np.float32))
            collected["labels"].append(np.asarray(labels[endpoint], dtype=np.int8))
            collected["families"].append(family[endpoint].astype(np.int8))
            collected["event"].append(event_grid[endpoint])
            count = int(endpoint.sum())
            collected["scenario"].append(np.full(count, uid, dtype=np.int64))
            collected["source"].append(np.full(count, arrays["source"], dtype=np.int32))
            collected["timestep"].append(
                np.broadcast_to(times[:, None], endpoint.shape)[endpoint].astype(np.int16))
            collected["node"].append(
                np.broadcast_to(np.arange(labels.shape[1]), labels.shape)[endpoint].astype(np.int16))
            normal = endpoint & (family == 0)
            errors.extend(np.abs(values[15:] - prediction[15:])[normal].tolist())
        result = {key: np.concatenate(parts) for key, parts in collected.items()}
        if not np.isfinite(result["X"]).all():
            raise ValueError("Nonfinite expanded TRAIN features")
        result["normal_reference_mae_m"] = np.asarray(np.mean(errors), dtype=np.float64)
        return result, names

    def save_reference(self, path, scenario_ids):
        reference = self.reference(scenario_ids)
        joblib.dump(reference, path)
        return reference
