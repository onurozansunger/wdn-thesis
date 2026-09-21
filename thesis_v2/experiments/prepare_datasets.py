"""Prepare and optionally generate the thesis-v2 dataset matrix.

Examples:

    # Write the six full config files without generating data.
    python thesis_v2/experiments/prepare_datasets.py --prepare-only

    # Generate small Modena and L-Town smoke datasets.
    python thesis_v2/experiments/prepare_datasets.py --smoke --network all

    # Generate all three full Modena data seeds.
    python thesis_v2/experiments/prepare_datasets.py --network modena
"""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path

import yaml


V2 = Path(__file__).resolve().parents[1]
ROOT = V2.parent
MATRIX_PATH = V2 / "experiments" / "configs" / "dataset_matrix.yaml"
GENERATED_CONFIGS = V2 / "experiments" / "configs" / "generated"


def deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def build_config(matrix: dict, network: str, seed: int, smoke: bool) -> dict:
    cfg = deep_merge(matrix["common"], matrix["networks"][network])
    cfg["num_scenarios"] = int(
        matrix["smoke_scenarios"] if smoke else matrix["full_scenarios"]
    )
    cfg["seed"] = int(seed)
    suffix = "smoke" if smoke else f"seed{seed}"
    cfg["output_dir"] = f"data/thesis_v2/{network}_episode_{suffix}"
    return cfg


def write_config(cfg: dict, network: str, seed: int, smoke: bool) -> Path:
    GENERATED_CONFIGS.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if smoke else f"seed{seed}"
    path = GENERATED_CONFIGS / f"{network}_episode_{suffix}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", choices=["all", "modena", "ltown"], default="all")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    matrix = yaml.safe_load(MATRIX_PATH.read_text())
    networks = list(matrix["networks"]) if args.network == "all" else [args.network]
    seeds = args.seeds or list(matrix["seeds"])
    if args.smoke:
        # A smoke dataset checks the pipeline, not data-seed uncertainty.
        seeds = [seeds[0]]

    prepared: list[Path] = []
    for network in networks:
        for seed in seeds:
            cfg = build_config(matrix, network, seed, args.smoke)
            path = write_config(cfg, network, seed, args.smoke)
            prepared.append(path)
            print(f"prepared {path.relative_to(ROOT)} -> {cfg['output_dir']}")

    if args.prepare_only:
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env.setdefault("MPLCONFIGDIR", "/tmp/wdn-thesis-mpl")
    for path in prepared:
        print(f"\ngenerating from {path.relative_to(ROOT)}", flush=True)
        subprocess.run(
            [sys.executable, "-m", "wdn.generate", "--config", str(path)],
            cwd=ROOT,
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
