"""CLI entrypoint for data generation.

Usage:
    python -m wdn.generate --config configs/generate.yaml
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

from wdn.config import GenerateConfig, load_config, save_config
from wdn.data_generation import generate_dataset
from wdn.corruption import corrupt_all_snapshots


def main():
    parser = argparse.ArgumentParser(description="Generate WDN dataset")
    parser.add_argument("--config", type=str, default="configs/generate.yaml",
                        help="Path to generation config YAML")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config, GenerateConfig)

    t0 = time.time()

    # Step 1: Generate clean snapshots
    print("=" * 60)
    print("Step 1: Generating clean snapshots via WNTR simulation")
    print("=" * 60)
    graph, snapshots = generate_dataset(cfg)

    # Step 2: Apply corruption
    print("\n" + "=" * 60)
    print("Step 2: Applying corruption (missing data + noise)")
    print("=" * 60)
    corrupted = corrupt_all_snapshots(snapshots, cfg.corruption, seed=cfg.seed)
    n_attacked = sum(1 for c in corrupted if c.pressure_anomaly.sum() > 0 or c.flow_anomaly.sum() > 0)
    print(f"Corrupted {len(corrupted)} snapshots "
          f"(missing_p={cfg.corruption.missing_rate_pressure}, "
          f"missing_q={cfg.corruption.missing_rate_flow}, "
          f"noise_p={cfg.corruption.noise_sigma_pressure}, "
          f"noise_q={cfg.corruption.noise_sigma_flow})")
    if cfg.corruption.attack_enabled:
        print(f"Attacks enabled: type={cfg.corruption.attack_type}, "
              f"fraction={cfg.corruption.attack_fraction}, "
              f"{n_attacked}/{len(corrupted)} snapshots contain attacks")

    # Step 3: Save everything
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save graph
    with open(out_dir / "graph.pkl", "wb") as f:
        pickle.dump(graph, f)

    # Save snapshots and corrupted data
    with open(out_dir / "snapshots.pkl", "wb") as f:
        pickle.dump(snapshots, f)

    with open(out_dir / "corrupted.pkl", "wb") as f:
        pickle.dump(corrupted, f)

    # Save config for reproducibility
    save_config(cfg, out_dir / "generate_config.yaml")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Done! Generated {len(snapshots)} snapshots in {elapsed:.1f}s")
    print(f"Saved to: {out_dir}")
    print(f"  - graph.pkl          ({graph.num_nodes} nodes, {graph.num_edges} edges)")
    print(f"  - snapshots.pkl      ({len(snapshots)} clean snapshots)")
    print(f"  - corrupted.pkl      ({len(corrupted)} corrupted snapshots)")
    print(f"  - generate_config.yaml")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
