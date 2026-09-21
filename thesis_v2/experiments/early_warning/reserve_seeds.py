"""Reserve generator seeds for the early-warning campaign, collision-checked.

Collisions are checked against every generator configuration and dataset
manifest that exists on disk, not against filenames. A filename says what
someone meant to write; ``generate_config.yaml`` says what was actually
generated, and those have disagreed before.

    /opt/miniconda3/bin/python thesis_v2/experiments/early_warning/reserve_seeds.py
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "runs/operational/early_warning_multiseed_v1/seed_manifest.json"

# Frozen in EARLY_WARNING_MULTISEED_PROTOCOL.md section 8.
RESERVATION = {
    "modena_eval": [40811, 41811, 42811, 43811, 44811, 45811],
    "ltown_eval": [50811, 51811, 52811, 53811, 54811, 55811],
    "ltown_train": [60811, 61811, 62811, 63811, 64811],
    "ltown_calibration": [70811, 71811, 72811, 73811],
    # Amended 2026-09-04, before any generation of the full L-Town corpora:
    # one 2-scenario smoke corpus was not enough to measure reference-fit
    # cost, which is superlinear in rows. See protocol section 11.
    "ltown_pilot": [90811, 91811, 92811],
}

# Never reused: locked test source, existing TRAIN/calibration, EVAL-1/2/3.
KNOWN_FORBIDDEN = (
    [811, 1811, 2811, 3811, 10811, 11811, 12811, 13811, 14811, 15811]
    + list(range(4811, 9812, 1000))
    + list(range(20811, 25812, 1000))
    + list(range(30811, 35812, 1000))
)

# Model-fitting randomness for the five repeated training runs. These are not
# generator seeds and share no namespace with them.
TRAINING_SEEDS = [701, 702, 703, 704, 705]


def observed_seeds():
    """Every generator seed that any config or manifest on disk actually used."""
    found = {}

    def record(seed, origin):
        found.setdefault(int(seed), []).append(str(origin.relative_to(ROOT)))

    for path in sorted((ROOT / "configs").rglob("*.yaml")):
        try:
            config = yaml.safe_load(path.read_text())
        except yaml.YAMLError:
            continue
        if isinstance(config, dict) and "seed" in config:
            record(config["seed"], path)

    for path in sorted((ROOT / "data").rglob("generate_config.yaml")):
        config = yaml.safe_load(path.read_text())
        if isinstance(config, dict) and "seed" in config:
            record(config["seed"], path)

    for path in sorted((ROOT / "data").rglob("manifest.json")):
        try:
            manifest = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(manifest, dict) and isinstance(manifest.get("seed"), int):
            record(manifest["seed"], path)

    return found


def main():
    used = observed_seeds()
    reserved = [seed for seeds in RESERVATION.values() for seed in seeds]
    if len(set(reserved)) != len(reserved):
        raise ValueError("The reservation itself contains a duplicate seed")

    collisions = {}
    already_generated = {}
    for purpose, seeds in RESERVATION.items():
        for seed in seeds:
            origins = list(used.get(seed, []))
            if seed in KNOWN_FORBIDDEN:
                origins.append("protocol forbidden list")
            if not origins:
                continue
            # A seed this campaign already generated for this same purpose is not
            # a collision: re-running the reservation must stay idempotent. The
            # directory name carries the purpose, so the match is checked, not
            # assumed.
            mine = f"ew_{{}}_{purpose}_seed{seed}".format
            if all(any(mine(tag) in origin for tag in ("modena", "ltown"))
                   for origin in origins):
                already_generated[seed] = origins
                continue
            collisions[seed] = origins
    if collisions:
        raise SystemExit(f"Seed collision, refusing to reserve: {collisions}")

    if set(TRAINING_SEEDS) & set(reserved):
        raise ValueError("Training seeds must not collide with generator seeds")

    manifest = {
        "campaign": "early_warning_multiseed_v1",
        "reserved_before_generation": True,
        "generator_seeds": RESERVATION,
        "training_seeds": TRAINING_SEEDS,
        "collision_check": {
            "sources_scanned": ["configs/**/*.yaml", "data/**/generate_config.yaml",
                                "data/**/manifest.json", "protocol forbidden list"],
            "distinct_seeds_already_in_use": sorted(used),
            "forbidden_listed": sorted(set(KNOWN_FORBIDDEN)),
            "collisions": {},
            "already_generated_by_this_campaign": already_generated,
        },
        "notes": (
            "Generator seeds are reserved before any data is generated. Training "
            "seeds drive model-fitting randomness only and never change split "
            "membership or the data distribution."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(OUTPUT)
    print(json.dumps({"reserved": RESERVATION, "training_seeds": TRAINING_SEEDS,
                      "already_in_use": sorted(used)}, indent=2))


if __name__ == "__main__":
    main()
