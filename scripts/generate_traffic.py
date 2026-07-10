"""Generate a road-traffic sensor-network attack-detection dataset in the
same format as the water and power datasets, so the temporal
Mixture-of-Experts GNN and the corruption pipeline run unchanged.

Third cross-domain setting for the thesis (water -> power -> traffic).
A traffic sensor network is a graph of loop detectors on a road network;
each detector reports a speed time-series with strong rush-hour cycles
and spatial correlation between neighbouring roads.

    water / power             traffic
    -------------             -------
    junction / bus     ->     road sensor (loop detector)
    pipe / line        ->     road segment
    pressure / voltage ->     sensor speed (km/h)
    flow               ->     segment traffic volume
    mass conservation  ->     flow conservation at junctions

The 5 hand-crafted attacks map directly: a stealthy bias on a speed
sensor is a data-spoofing attack that hides congestion (or fabricates
it) — the canonical intelligent-transport-system threat.

The road graph is a random geometric graph (sensors placed in 2-D,
connected within a radius) — a standard planar model of a road network.
Speeds follow a daily free-flow/rush-hour curve, are smoothed over the
graph so neighbours correlate, and carry a slow temporal component so a
replayed reading is near the noise floor (the same reason replay is hard
on water and power).

Usage:
    python3 scripts/generate_traffic.py --nodes 200 --scenarios 50 \
        --timesteps 24 --seed 99
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from wdn.dataset import Snapshot, WDNGraph
from wdn.config import CorruptionConfig
from wdn.corruption import corrupt_all_snapshots


# ---------------------------------------------------------------------------
# Road graph (random geometric graph, guaranteed connected)
# ---------------------------------------------------------------------------

def build_graph(n_nodes: int, radius: float, rng) -> WDNGraph:
    coords = rng.uniform(0, 1, size=(n_nodes, 2)).astype(np.float32)

    # Connect pairs within `radius`; then stitch components with nearest
    # links so the whole network is one connected road system.
    def pair_dist(i, j):
        return float(np.linalg.norm(coords[i] - coords[j]))

    edges = set()
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if pair_dist(i, j) <= radius:
                edges.add((i, j))

    # Union-find to find components.
    parent = list(range(n_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for i, j in edges:
        union(i, j)
    # Connect components by nearest cross-component node.
    changed = True
    while changed:
        changed = False
        roots = {}
        for i in range(n_nodes):
            roots.setdefault(find(i), []).append(i)
        if len(roots) == 1:
            break
        comp_list = list(roots.values())
        base = comp_list[0]
        best = None
        for other in comp_list[1:]:
            for a in base:
                for b in other:
                    d = pair_dist(a, b)
                    if best is None or d < best[0]:
                        best = (d, a, b)
        if best:
            edges.add((min(best[1], best[2]), max(best[1], best[2])))
            union(best[1], best[2])
            changed = True

    edge_list = sorted(edges)
    NE = len(edge_list)
    edge_index = np.array(edge_list, dtype=np.int64).T  # (2, NE)

    # Road class per sensor: 0 local, 1 arterial, 2 highway (by degree).
    deg = np.zeros(n_nodes, dtype=np.int64)
    for a, b in edge_list:
        deg[a] += 1
        deg[b] += 1
    node_types = np.zeros(n_nodes, dtype=np.int64)
    node_types[deg >= np.quantile(deg, 0.66)] = 1
    node_types[deg >= np.quantile(deg, 0.90)] = 2

    free_flow = rng.uniform(50, 100, size=n_nodes).astype(np.float32)

    # Edge attributes: length (Euclidean), capacity, lanes.
    lengths = np.array([pair_dist(a, b) for a, b in edge_list], dtype=np.float32)
    capacity = rng.uniform(0.5, 1.0, size=NE).astype(np.float32)
    lanes = rng.integers(1, 4, size=NE).astype(np.float32)
    edge_types = np.zeros(NE, dtype=np.int64)  # single road type slot

    incidence = np.zeros((n_nodes, NE), dtype=np.float32)
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for j, (a, b) in enumerate(edge_list):
        incidence[a, j] = +1.0
        incidence[b, j] = -1.0
        adjacency[a, b] = 1.0
        adjacency[b, a] = 1.0

    return WDNGraph(
        node_names=[f"sensor_{i}" for i in range(n_nodes)],
        node_types=node_types,
        node_elevations=free_flow,           # reuse slot: free-flow speed
        node_base_demands=deg.astype(np.float32),  # reuse slot: degree
        node_coordinates=coords,
        edge_names=[f"road_{j}" for j in range(NE)],
        edge_types=edge_types,
        edge_lengths=lengths,
        edge_diameters=capacity,
        edge_roughness=lanes,
        edge_index=edge_index,
        incidence_matrix=incidence,
        adjacency_matrix=adjacency,
    )


# ---------------------------------------------------------------------------
# Static features (5 & 6 cols) + bidirectional edges
# ---------------------------------------------------------------------------

def _norm(a):
    a = a.astype(np.float32).copy()
    rng = a.max() - a.min()
    return (a - a.min()) / rng if rng > 0 else np.zeros_like(a)


def build_static(graph: WDNGraph):
    node_type_oh = np.eye(3, dtype=np.float32)[graph.node_types]
    node_static = np.column_stack([
        _norm(graph.node_elevations),     # free-flow speed
        _norm(graph.node_base_demands),   # degree
        node_type_oh,
    ])
    edge_type_oh = np.eye(3, dtype=np.float32)[graph.edge_types]
    edge_static = np.column_stack([
        _norm(graph.edge_lengths),        # length
        _norm(graph.edge_diameters),      # capacity
        _norm(graph.edge_roughness),      # lanes
        edge_type_oh,
    ])
    return (torch.tensor(node_static, dtype=torch.float32),
            torch.tensor(edge_static, dtype=torch.float32))


def make_bidirectional(edge_index, num_edges):
    rev = np.stack([edge_index[1], edge_index[0]], axis=0)
    bi = np.concatenate([edge_index, rev], axis=1)
    edge_map = np.concatenate([np.arange(num_edges), np.arange(num_edges)])
    return (torch.tensor(bi, dtype=torch.long),
            torch.tensor(edge_map, dtype=torch.long))


# ---------------------------------------------------------------------------
# Speed dynamics
# ---------------------------------------------------------------------------

def congestion_curve(n_steps: int) -> np.ndarray:
    """Daily speed multiplier in ~[0.45, 1.0]: dips at the morning and
    evening rush hours, free-flow overnight."""
    hours = np.linspace(0, 24, n_steps, endpoint=False)
    morning = np.exp(-0.5 * ((hours - 8) / 1.4) ** 2)
    evening = np.exp(-0.5 * ((hours - 18) / 1.6) ** 2)
    congestion = 0.55 * (morning + evening)          # 0..~0.55 drop
    return (1.0 - np.clip(congestion, 0, 0.55)).astype(np.float32)


def graph_smooth(field, adjacency, iters=2):
    """Smooth a per-node field over the graph so neighbours correlate."""
    deg = adjacency.sum(1, keepdims=True)
    deg[deg == 0] = 1.0
    P = adjacency / deg
    out = field.copy()
    for _ in range(iters):
        out = 0.5 * out + 0.5 * (P @ out)
    return out


def simulate(graph, n_scenarios, n_steps, rng):
    node_static, edge_static = build_static(graph)
    bi_edge_index, edge_map = make_bidirectional(graph.edge_index, graph.num_edges)
    curve = congestion_curve(n_steps)
    free_flow = graph.node_elevations.copy()          # (N,)
    adjacency = graph.adjacency_matrix
    ei = graph.edge_index

    snapshots = []
    for sc in range(n_scenarios):
        # Per-scenario congestion severity + a smooth spatial "incident" field.
        severity = rng.uniform(0.8, 1.2)
        spatial = graph_smooth(rng.normal(0, 0.08, size=free_flow.shape),
                               adjacency, iters=3)
        for t in range(n_steps):
            # Slow temporal noise (AR-like via small step) keeps replay hard.
            base = free_flow * (1.0 - severity * (1.0 - curve[t]))
            speed = base * (1.0 + spatial) * (
                1.0 + rng.normal(0, 0.015, size=free_flow.shape))
            speed = np.clip(speed, 5.0, 120.0).astype(np.float32)

            # Edge volume: higher when the road is congested (low speed),
            # scaled by capacity — a congestion-volume proxy.
            s_src = speed[ei[0]]
            s_dst = speed[ei[1]]
            avg_speed = 0.5 * (s_src + s_dst)
            vol = graph.edge_diameters * (120.0 - avg_speed)   # capacity * slowness
            vol = (vol + rng.normal(0, 1.0, size=vol.shape)).astype(np.float32)

            snapshots.append(Snapshot(
                pressure_true=torch.tensor(speed, dtype=torch.float32),
                flow_true=torch.tensor(vol, dtype=torch.float32),
                node_static=node_static,
                edge_static=edge_static,
                edge_index=bi_edge_index,
                edge_map=edge_map,
                scenario_id=sc,
                timestep=t,
            ))
    return snapshots


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nodes", type=int, default=200)
    p.add_argument("--radius", type=float, default=0.11)
    p.add_argument("--scenarios", type=int, default=50)
    p.add_argument("--timesteps", type=int, default=24)
    p.add_argument("--seed", type=int, default=99)
    p.add_argument("--out_dir", default="data/temporal_moe_traffic")
    p.add_argument("--missing_rate", type=float, default=0.5)
    p.add_argument("--attack_fraction", type=float, default=0.15)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"Building road graph: {args.nodes} sensors, radius {args.radius} ...")
    graph = build_graph(args.nodes, args.radius, rng)
    print(f"Road network: {graph.num_nodes} sensors, {graph.num_edges} segments "
          f"(avg degree {2 * graph.num_edges / graph.num_nodes:.1f})")

    print(f"Simulating {args.scenarios} days x {args.timesteps} steps ...")
    snapshots = simulate(graph, args.scenarios, args.timesteps, rng)
    sp0 = snapshots[0].pressure_true
    print(f"Generated {len(snapshots)} snapshots "
          f"(speed range {sp0.min():.1f}–{sp0.max():.1f} km/h)")

    # Speed ~50–100 km/h; ~3 km/h sensor noise (~4%), stealthy bias ~10 km/h.
    corr_cfg = CorruptionConfig(
        missing_rate_pressure=args.missing_rate,
        missing_rate_flow=args.missing_rate,
        noise_sigma_pressure=3.0,
        noise_sigma_flow=3.0,
        attack_enabled=True,
        attack_fraction=args.attack_fraction,
        attack_bias=10.0,
        attack_scale=1.5,
        attack_type="mixed",
    )
    print("Injecting attacks (same 5-class corruption pipeline) ...")
    corrupted = corrupt_all_snapshots(snapshots, corr_cfg, seed=args.seed)

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "graph.pkl", "wb") as f:
        pickle.dump(graph, f)
    with open(out / "snapshots.pkl", "wb") as f:
        pickle.dump(snapshots, f)
    with open(out / "corrupted.pkl", "wb") as f:
        pickle.dump(corrupted, f)

    ids = np.array([c.attack_type_id for c in corrupted])
    names = ["clean", "random", "replay", "stealthy", "noise", "targeted"]
    hist = {names[i]: int((ids == i).sum()) for i in range(6)}
    print(f"Saved to {out}")
    print(f"Attack distribution: {hist}")


if __name__ == "__main__":
    main()
