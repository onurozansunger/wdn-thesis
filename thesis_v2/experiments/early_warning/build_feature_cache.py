"""Build and cache the 116-column pressure feature bank for a named corpus.

One cache file per generator source, so a partial run resumes instead of
rebuilding everything. The cache is keyed by a configuration hash: if the
generator config, the reference, or the feature schema changes, the run
*refuses* the stale cache rather than quietly overwriting it.

The reference is loaded frozen and never refitted here. Nothing in this module
reads a label, an event boundary, an attack parameter, or a clean hydraulic
value: labels and event ids travel alongside the features for supervised
training and evaluation only.

    /opt/miniconda3/bin/python thesis_v2/experiments/early_warning/build_feature_cache.py \
        --corpus modena_calibration
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import yaml

from wdn.latency_deployment import specialist_bank
from wdn.run_expert_redesign import CampaignData

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/thesis_v2"
CAMPAIGN = ROOT / "runs/operational/early_warning_multiseed_v1"
SPLITS = ROOT / "runs/operational/blind_reference_probe_rank16/splits.json"
MODENA_REFERENCE = ROOT / "runs/operational/seasonal_family_deployment_v2/full/reference.joblib"
NAMES = ROOT / "runs/operational/seasonal_family_deployment_v2/feature_names.json"
SEASONAL_NAMES = ROOT / "runs/operational/seasonal_family_deployment_v2/seasonal_feature_names.json"

#: Frozen corpus definitions. ``pieces`` is (directory, generator seed, scenarios).
CORPORA = {
    "modena_calibration": {
        "reference": MODENA_REFERENCE,
        "network": "modena",
        "pieces": [("operational_modena_seed811", 811, "split:calibration")]
        + [(f"operational_calibration_expansion_seed{seed}", seed, "all:24")
           for seed in (12811, 13811, 14811, 15811)],
    },
    "modena_train": {
        "reference": MODENA_REFERENCE,
        "network": "modena",
        # Scenario counts differ by expansion generation: seeds 1811-3811 were
        # generated with 16 scenarios, seeds 10811/11811 with 24. 14 + 48 + 48 = 110.
        "pieces": [("operational_modena_seed811", 811, "split:train")]
        + [(f"operational_train_expansion_seed{seed}", seed, "all:16")
           for seed in (1811, 2811, 3811)]
        + [(f"operational_train_expansion2_seed{seed}", seed, "all:24")
           for seed in (10811, 11811)],
    },
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def atomic_npz(path, **arrays):
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def resolve_scenarios(spec, splits):
    if spec.startswith("split:"):
        return sorted(int(x) for x in splits[spec.split(":", 1)[1]])
    if spec.startswith("all:"):
        return list(range(int(spec.split(":", 1)[1])))
    raise ValueError(f"Unknown scenario specification {spec!r}")


def config_of(directory):
    return yaml.safe_load((directory / "generate_config.yaml").read_text())


def check_distribution(directory, reference_config):
    """Refuse a corpus whose generator distribution differs from the benchmark."""
    config = config_of(directory)
    allowed = {"profile", "output_dir", "seed", "num_scenarios", "network_inp"}
    changed = {key for key in set(config) | set(reference_config)
               if config.get(key) != reference_config.get(key)}
    if not changed <= allowed:
        raise ValueError(f"{directory.name} changes the distribution: {sorted(changed - allowed)}")
    if config["missing_rate_pressure"] != .5 or config["missing_rate_flow"] != .5:
        raise ValueError(f"{directory.name} does not fix both missing probabilities at 0.50")
    if config["duration_hours"] != 168 or config["timestep_minutes"] != 60:
        raise ValueError(f"{directory.name} uses a different observation process")
    return config


def build(corpus_name, data_root=DATA, output_root=None):
    spec = CORPORA[corpus_name]
    output = (output_root or CAMPAIGN / "features") / corpus_name
    output.mkdir(parents=True, exist_ok=True)
    splits = json.loads(SPLITS.read_text())
    bank = json.loads(NAMES.read_text()) + json.loads(SEASONAL_NAMES.read_text())
    reference_path = Path(spec["reference"])
    benchmark = config_of(data_root / "operational_modena_seed811")

    signature = {
        "corpus": corpus_name, "network": spec["network"],
        "reference_sha256": sha(reference_path),
        "feature_count": len(bank),
        "feature_names_sha256": hashlib.sha256(json.dumps(bank).encode()).hexdigest(),
        "pieces": [{"directory": name, "seed": seed, "scenarios": scenarios}
                   for name, seed, scenarios in spec["pieces"]],
    }
    signature_path = output / "signature.json"
    if signature_path.exists():
        stored = json.loads(signature_path.read_text())
        if stored != signature:
            raise SystemExit(
                f"Cached feature signature for {corpus_name} is incompatible with the "
                f"current configuration. Refusing to overwrite {output}. "
                "Inspect the difference and choose a new cache directory if intended.")
    else:
        write_json(signature_path, signature)

    reference = joblib.load(reference_path)
    started = time.monotonic()
    manifest = {"corpus": corpus_name, "network": spec["network"],
                "feature_names": bank, "pieces": [], "reference_sha256": signature["reference_sha256"]}

    for directory_name, seed, scenario_spec in spec["pieces"]:
        directory = data_root / directory_name
        config = check_distribution(directory, benchmark)
        scenarios = resolve_scenarios(scenario_spec, splits)
        cache = output / f"bank_seed{seed}.npz"
        if not cache.exists():
            print(f"building {corpus_name} features: seed {seed}, "
                  f"{len(scenarios)} scenarios", flush=True)
            data = CampaignData(directory)
            arrays, found = specialist_bank(data, scenarios, reference)
            if found != bank:
                raise ValueError(f"Feature schema differs on {directory_name}")
            rows = len(arrays["labels"])
            payload = {
                "X": np.asarray(arrays["X"], dtype=np.float32),
                "labels": np.asarray(arrays["labels"], dtype=np.int8),
                "families": np.asarray(arrays["families"], dtype=np.int8),
                "event": np.asarray(arrays["event"], dtype=np.int32),
                "scenario": np.asarray(arrays["scenario"], dtype=np.int64) + seed * 1000,
                "source": np.full(rows, seed, dtype=np.int64),
                "timestep": np.asarray(arrays["timestep"], dtype=np.int32),
                "node": np.asarray(arrays["node"], dtype=np.int32),
            }
            atomic_npz(cache, **payload)
            del data, arrays, payload
            gc.collect()
        with np.load(cache) as loaded:
            rows = int(len(loaded["labels"]))
        manifest["pieces"].append({
            "directory": directory_name, "seed": seed,
            "scenarios": scenarios, "rows": rows,
            "cache": str(cache.relative_to(ROOT)),
            "config_pressure_missing": config["missing_rate_pressure"],
            "config_flow_missing": config["missing_rate_flow"],
            "cache_sha256": sha(cache),
        })
        print(f"  seed {seed}: {rows} observed endpoints", flush=True)

    manifest["total_rows"] = sum(piece["rows"] for piece in manifest["pieces"])
    manifest["total_scenarios"] = sum(len(piece["scenarios"]) for piece in manifest["pieces"])
    manifest["elapsed_seconds"] = time.monotonic() - started
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"corpus": corpus_name, "scenarios": manifest["total_scenarios"],
                      "rows": manifest["total_rows"],
                      "elapsed_seconds": round(manifest["elapsed_seconds"], 1)}, indent=2))
    return manifest


def globalise_events(event, source):
    """Make per-generator event ids unique across sources.

    Each corpus directory numbers its events from zero, so concatenating two
    generator seeds silently merges event 0 of one with event 0 of the other.
    ``training_weights`` equalises risk per ``(family, event)`` group, so that
    merge would quietly reweight the training set. -1 means "no event" and stays
    -1.
    """
    event = np.asarray(event, dtype=np.int64)
    source = np.asarray(source, dtype=np.int64)
    return np.where(event >= 0, source * 10**6 + event, -1)


def load_corpus(corpus_name, columns=None, output_root=None):
    """Concatenate a cached corpus, optionally keeping only named feature columns."""
    output = (output_root or CAMPAIGN / "features") / corpus_name
    manifest = json.loads((output / "manifest.json").read_text())
    bank = manifest["feature_names"]
    keep = None if columns is None else [bank.index(name) for name in columns]
    parts = []
    for piece in manifest["pieces"]:
        with np.load(ROOT / piece["cache"]) as loaded:
            entry = {key: loaded[key] for key in loaded.files if key != "X"}
            entry["X"] = loaded["X"] if keep is None else loaded["X"][:, keep]
        entry["event"] = globalise_events(entry["event"], entry["source"])
        parts.append(entry)
    corpus = {key: np.concatenate([part[key] for part in parts], axis=0)
              for key in parts[0]}
    corpus["_feature_names"] = bank if columns is None else list(columns)
    return corpus, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, choices=sorted(CORPORA))
    build(parser.parse_args().corpus)


if __name__ == "__main__":
    main()
