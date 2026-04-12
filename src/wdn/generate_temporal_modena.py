"""Generate temporal data for Modena by adding diurnal demand patterns.

Modena's .inp file has no time patterns (duration=0, steady-state only).
This script adds a realistic diurnal demand curve before running the
simulation, producing 25 timesteps per scenario (24h at 1h intervals).

Usage:
    python -m wdn.generate_temporal_modena
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import wntr

from wdn.data_generation import build_graph, simulate_scenario
from wdn.corruption import corrupt_all_snapshots
from wdn.config import CorruptionConfig


# Diurnal demand pattern: 24 multipliers (one per hour)
# Realistic residential water usage: low at night, peaks morning & evening
DIURNAL_PATTERN = [
    0.3, 0.3, 0.3, 0.4, 0.5, 0.7,   # 00:00 - 05:00 (night)
    1.0, 1.3, 1.4, 1.2, 1.0, 0.9,    # 06:00 - 11:00 (morning peak)
    0.9, 0.8, 0.8, 0.9, 1.0, 1.1,    # 12:00 - 17:00 (afternoon)
    1.3, 1.2, 1.0, 0.8, 0.5, 0.3,    # 18:00 - 23:00 (evening peak, then drop)
]


def add_diurnal_pattern(wn: wntr.network.WaterNetworkModel, pattern_name="diurnal"):
    """Add a diurnal demand pattern to the network and assign it to all junctions."""
    # Set simulation duration to 24 hours
    wn.options.time.duration = 24 * 3600  # 24 hours in seconds
    wn.options.time.hydraulic_timestep = 3600  # 1 hour
    wn.options.time.pattern_timestep = 3600

    # Add the diurnal pattern
    wn.add_pattern(pattern_name, DIURNAL_PATTERN)
    pattern = wn.get_pattern(pattern_name)

    # Assign pattern to all junctions
    for jname in wn.junction_name_list:
        junction = wn.get_node(jname)
        if len(junction.demand_timeseries_list) > 0:
            junction.demand_timeseries_list[0]._pattern = pattern


def main():
    output_dir = Path("data/modena_temporal")
    output_dir.mkdir(parents=True, exist_ok=True)

    inp_path = "data/modena.inp"
    num_scenarios = 200  # 200 scenarios × 25 timesteps = 5000 snapshots
    demand_variation = 0.3
    seed = 42

    rng = np.random.default_rng(seed)

    # Build graph from base network
    wn_base = wntr.network.WaterNetworkModel(inp_path)
    graph = build_graph(wn_base)

    n_junctions = int((graph.node_types == 0).sum())
    print(f"Network: {graph.num_nodes} nodes, {graph.num_edges} edges")
    print(f"Generating {num_scenarios} temporal scenarios (24h each)...")

    all_snapshots = []

    for s in range(num_scenarios):
        # Reload fresh network each scenario
        wn = wntr.network.WaterNetworkModel(inp_path)

        # Add diurnal pattern
        add_diurnal_pattern(wn)

        # Apply random demand variation
        low = 1.0 - demand_variation
        high = 1.0 + demand_variation
        multipliers = rng.uniform(low, high, size=n_junctions).astype(np.float32)

        try:
            snapshots = simulate_scenario(wn, graph, scenario_id=s, demand_multipliers=multipliers)
            all_snapshots.extend(snapshots)
        except Exception as e:
            print(f"  Warning: scenario {s} failed: {e}")
            continue

        if (s + 1) % 20 == 0 or s == 0:
            print(f"  Scenario {s + 1}/{num_scenarios} done ({len(snapshots)} timesteps)")

    print(f"Total snapshots: {len(all_snapshots)}")
    timesteps_per_scenario = len(all_snapshots) // num_scenarios if num_scenarios > 0 else 0
    print(f"Timesteps per scenario: {timesteps_per_scenario}")

    # Corrupt
    corruption_cfg = CorruptionConfig(
        missing_rate_pressure=0.5,
        missing_rate_flow=0.5,
        noise_sigma_pressure=0.5,
        noise_sigma_flow=0.2,
        attack_enabled=True,
        attack_fraction=0.15,
        attack_bias=3.0,
        attack_scale=1.5,
        attack_type="mixed",
    )

    print("Corrupting snapshots...")
    corrupted = corrupt_all_snapshots(all_snapshots, corruption_cfg, seed=seed)

    # Save
    print(f"Saving to {output_dir}...")
    with open(output_dir / "graph.pkl", "wb") as f:
        pickle.dump(graph, f)
    with open(output_dir / "snapshots.pkl", "wb") as f:
        pickle.dump(all_snapshots, f)
    with open(output_dir / "corrupted.pkl", "wb") as f:
        pickle.dump(corrupted, f)

    print("Done!")


if __name__ == "__main__":
    main()
