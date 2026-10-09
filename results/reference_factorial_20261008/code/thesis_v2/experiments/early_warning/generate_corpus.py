"""Generate an operational corpus for this campaign, on either network.

Every field of the frozen benchmark (protocol section 3) is fixed here. Only
``profile``, ``network_inp``, ``output_dir``, ``seed`` and ``num_scenarios``
vary. The seed must appear in the campaign's reservation manifest, and the
output directory is never overwritten: ``OperationalConfig``'s own
``mkdir(exist_ok=False)`` refuses a second write, and this script refuses before
that with a clearer message.

    /opt/miniconda3/bin/python thesis_v2/experiments/early_warning/generate_corpus.py \
        --network ltown --purpose ltown_pilot --seed 90811 --scenarios 2
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import yaml

from wdn.operational_data import OperationalConfig

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / "runs/operational/early_warning_multiseed_v1"
SEED_MANIFEST = CAMPAIGN / "seed_manifest.json"
CONFIG_DIR = ROOT / "configs/early_warning"
DATA = ROOT / "data/thesis_v2"

NETWORKS = {
    "modena": {"inp": "data/modena.inp", "tag": "modena"},
    "ltown": {"inp": "data/L-Town.inp", "tag": "ltown"},
}

#: Frozen benchmark. Copied from the protocol, not from a previous config file.
BENCHMARK = dict(
    duration_hours=168, timestep_minutes=60,
    missing_rate_pressure=0.5, missing_rate_flow=0.5,
    pressure_noise_sigma_m=0.1, flow_noise_sigma_m3s=0.0001,
    attack_fraction=0.05, clean_gap_hours=(48, 96),
    attack_duration_hours=(6, 18), pressure_bias_m=(0.5, 2.0),
    flow_bias_m3s=(0.0005, 0.002), drift_ramp_hours=(6, 18),
    replay_lag_hours=(2, 6), injected_noise_factor=(3, 8),
    demand_variation=0.2, demand_pattern_amplitude=1.0,
)


def reserved_seeds():
    manifest = json.loads(SEED_MANIFEST.read_text())
    return {purpose: set(seeds) for purpose, seeds in manifest["generator_seeds"].items()}


def config_for(network, purpose, seed, scenarios):
    reservations = reserved_seeds()
    if purpose not in reservations:
        raise SystemExit(f"Unknown purpose {purpose!r}; reserved purposes: {sorted(reservations)}")
    if seed not in reservations[purpose]:
        raise SystemExit(
            f"Seed {seed} is not reserved for {purpose}. Reserved: "
            f"{sorted(reservations[purpose])}. Amend the seed manifest before generating.")
    tag = NETWORKS[network]["tag"]
    output_dir = DATA / f"ew_{tag}_{purpose}_seed{seed}"
    config = OperationalConfig(
        profile=f"early_warning_multiseed_v1_{tag}_{purpose}",
        network_inp=NETWORKS[network]["inp"],
        output_dir=str(output_dir.relative_to(ROOT)),
        seed=seed, num_scenarios=scenarios, **BENCHMARK)
    config.validate()
    return config, output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", required=True, choices=sorted(NETWORKS))
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scenarios", type=int, default=24)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config, output_dir = config_for(args.network, args.purpose, args.seed, args.scenarios)
    if output_dir.exists():
        print(f"{output_dir} already exists; not regenerating", flush=True)
        return
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_path = CONFIG_DIR / f"{output_dir.name}.yaml"
    config_path.write_text(yaml.safe_dump(asdict(config)))
    print(f"wrote {config_path.relative_to(ROOT)}", flush=True)
    if args.dry_run:
        return

    started = time.monotonic()
    subprocess.run([sys.executable, "-m", "wdn.operational_data",
                    "--config", str(config_path)], cwd=ROOT, check=True)
    elapsed = time.monotonic() - started
    timing = CAMPAIGN / "generation_timings.json"
    record = json.loads(timing.read_text()) if timing.exists() else {}
    record[output_dir.name] = {
        "network": args.network, "purpose": args.purpose, "seed": args.seed,
        "scenarios": args.scenarios, "seconds": round(elapsed, 1),
        "bytes": sum(p.stat().st_size for p in output_dir.iterdir() if p.is_file()),
    }
    timing.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record[output_dir.name], indent=2), flush=True)


if __name__ == "__main__":
    main()
