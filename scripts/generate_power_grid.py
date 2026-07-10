"""Generate a power-grid attack-detection dataset in the exact same
format as the water-network datasets, so the temporal Mixture-of-Experts
GNN and the corruption pipeline run on it unchanged.

Cross-domain validation for the thesis: the supervisors asked to test
the approach beyond water infrastructure. A power grid is the closest
sibling — both are physics-constrained cyber-physical systems with
sensor time-series on a graph:

    water                     power grid
    -----                     ----------
    junction / node    ->     bus
    pipe / link        ->     transmission line / transformer
    pressure_true      ->     bus voltage magnitude (vm_pu)
    flow_true          ->     line active-power flow (p_from_mw)
    mass conservation  ->     Kirchhoff current law (KCL)

The 5 hand-crafted attacks (random / replay / stealthy / noise /
targeted) are injected by the *same* corruption pipeline — a stealthy
bias on voltage measurements is exactly a false-data-injection attack
(FDIA), the canonical smart-grid threat.

Output (data/temporal_moe_power/):
    graph.pkl       — WDNGraph-compatible topology
    snapshots.pkl   — list[Snapshot] (clean truth)
    corrupted.pkl   — list[CorruptedSnapshot] (attacks injected)

Usage:
    python3 scripts/generate_power_grid.py \
        --case case118 --scenarios 50 --timesteps 24 --seed 99
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

import pandapower as pp
import pandapower.networks as nw

from wdn.dataset import Snapshot, WDNGraph
from wdn.config import CorruptionConfig
from wdn.corruption import corrupt_all_snapshots


# ---------------------------------------------------------------------------
# Topology → WDNGraph
# ---------------------------------------------------------------------------

def build_graph(net) -> tuple[WDNGraph, np.ndarray, list[str]]:
    """Build a WDNGraph from a pandapower net.

    Edges are lines followed by transformers. Returns the graph, an
    ``edge_kind`` array (0=line, 1=trafo) and the ordered edge names so
    the simulator can extract flows in the same order.
    """
    N = len(net.bus)
    # Buses are 0..N-1 contiguous for the standard IEEE cases; map to be safe.
    bus_ids = list(net.bus.index)
    bus_pos = {b: i for i, b in enumerate(bus_ids)}

    # --- Edges: lines then transformers ---
    src, dst, edge_kind, edge_names = [], [], [], []
    x_vals, r_vals, rate_vals = [], [], []
    for idx, row in net.line.iterrows():
        src.append(bus_pos[int(row.from_bus)])
        dst.append(bus_pos[int(row.to_bus)])
        edge_kind.append(0)
        edge_names.append(f"line_{idx}")
        x_vals.append(float(row.x_ohm_per_km) * float(row.length_km))
        r_vals.append(float(row.r_ohm_per_km) * float(row.length_km))
        rate_vals.append(float(row.get("max_i_ka", 1.0) or 1.0))
    for idx, row in net.trafo.iterrows():
        src.append(bus_pos[int(row.hv_bus)])
        dst.append(bus_pos[int(row.lv_bus)])
        edge_kind.append(1)
        edge_names.append(f"trafo_{idx}")
        # Transformer reactance proxy from vk%, no length.
        x_vals.append(float(row.get("vk_percent", 10.0)) / 100.0)
        r_vals.append(0.0)
        rate_vals.append(float(row.get("sn_mva", 100.0)) / 100.0)

    NE = len(src)
    edge_index = np.stack([np.array(src), np.array(dst)], axis=0).astype(np.int64)
    edge_kind = np.array(edge_kind, dtype=np.int64)

    # --- Node types: slack=1 (source), gen=2 (controllable), load/other=0 ---
    node_types = np.zeros(N, dtype=np.int64)
    for b in net.gen.bus.tolist():
        node_types[bus_pos[int(b)]] = 2
    for b in net.ext_grid.bus.tolist():
        node_types[bus_pos[int(b)]] = 1  # slack overrides gen

    # --- Node base load (MW) per bus, for the static feature ---
    base_load = np.zeros(N, dtype=np.float32)
    for _, row in net.load.iterrows():
        base_load[bus_pos[int(row.bus)]] += float(row.p_mw)
    base_gen = np.zeros(N, dtype=np.float32)
    for _, row in net.gen.iterrows():
        base_gen[bus_pos[int(row.bus)]] += float(row.p_mw)

    # --- Incidence (signed) + adjacency ---
    incidence = np.zeros((N, NE), dtype=np.float32)
    adjacency = np.zeros((N, N), dtype=np.float32)
    for j in range(NE):
        s, d = edge_index[0, j], edge_index[1, j]
        incidence[s, j] = +1.0
        incidence[d, j] = -1.0
        adjacency[s, d] = 1.0
        adjacency[d, s] = 1.0

    # Store extra arrays on the graph via the existing WDN fields:
    #   node_elevations  <- base_gen   (a per-bus scalar, only used for a
    #                        normalised static feature downstream)
    #   node_base_demands<- base_load
    #   edge_lengths     <- reactance x, edge_diameters <- rating,
    #   edge_roughness   <- resistance r
    graph = WDNGraph(
        node_names=[str(b) for b in bus_ids],
        node_types=node_types,
        node_elevations=base_gen,
        node_base_demands=base_load,
        node_coordinates=np.zeros((N, 2), dtype=np.float32),
        edge_names=edge_names,
        edge_types=edge_kind,
        edge_lengths=np.array(x_vals, dtype=np.float32),
        edge_diameters=np.array(rate_vals, dtype=np.float32),
        edge_roughness=np.array(r_vals, dtype=np.float32),
        edge_index=edge_index,
        incidence_matrix=incidence,
        adjacency_matrix=adjacency,
    )
    return graph, edge_kind, edge_names


# ---------------------------------------------------------------------------
# Static feature tensors (mirror data_generation._build_*), 5 & 6 cols
# ---------------------------------------------------------------------------

def _norm(a):
    a = a.astype(np.float32).copy()
    rng = a.max() - a.min()
    return (a - a.min()) / rng if rng > 0 else np.zeros_like(a)


def build_static(graph: WDNGraph):
    N, NE = graph.num_nodes, graph.num_edges
    node_type_oh = np.eye(3, dtype=np.float32)[graph.node_types]
    node_static = np.column_stack([
        _norm(graph.node_base_demands),   # base load
        _norm(graph.node_elevations),     # base gen
        node_type_oh,
    ])
    edge_type_oh = np.eye(3, dtype=np.float32)[graph.edge_types]
    edge_static = np.column_stack([
        _norm(graph.edge_lengths),        # reactance x
        _norm(graph.edge_diameters),      # rating
        _norm(graph.edge_roughness),      # resistance r
        edge_type_oh,
    ])
    return (torch.tensor(node_static, dtype=torch.float32),
            torch.tensor(edge_static, dtype=torch.float32))


def make_bidirectional(edge_index: np.ndarray, num_edges: int):
    reversed_index = np.stack([edge_index[1], edge_index[0]], axis=0)
    bi = np.concatenate([edge_index, reversed_index], axis=1)
    edge_map = np.concatenate([np.arange(num_edges), np.arange(num_edges)])
    return (torch.tensor(bi, dtype=torch.long),
            torch.tensor(edge_map, dtype=torch.long))


# ---------------------------------------------------------------------------
# Daily load curve
# ---------------------------------------------------------------------------

def daily_curve(n_steps: int) -> np.ndarray:
    """A smooth double-peak daily demand multiplier in ~[0.6, 1.1]."""
    t = np.linspace(0, 2 * np.pi, n_steps, endpoint=False)
    # Morning + evening peaks.
    curve = (0.85
             + 0.12 * np.sin(t - np.pi / 2)
             + 0.10 * np.sin(2 * t - np.pi / 3))
    return curve.astype(np.float32)


# ---------------------------------------------------------------------------
# Simulate scenarios
# ---------------------------------------------------------------------------

def simulate(net, graph, edge_kind, n_scenarios, n_steps, rng):
    node_static, edge_static = build_static(graph)
    bi_edge_index, edge_map = make_bidirectional(graph.edge_index, graph.num_edges)
    curve = daily_curve(n_steps)

    base_load_p = net.load.p_mw.values.copy()
    base_load_q = net.load.q_mvar.values.copy()
    n_lines = len(net.line)

    snapshots = []
    n_fail = 0
    for sc in range(n_scenarios):
        # Per-scenario demand level and per-bus load offsets.
        level = 1.0 + rng.uniform(-0.20, 0.20)
        bus_offset = 1.0 + rng.uniform(-0.10, 0.10, size=base_load_p.shape)
        for t in range(n_steps):
            mult = level * curve[t] * bus_offset * (
                1.0 + rng.normal(0, 0.02, size=base_load_p.shape))
            net.load.p_mw = base_load_p * mult
            net.load.q_mvar = base_load_q * mult
            try:
                pp.runpp(net, numba=False)
            except Exception:
                n_fail += 1
                continue

            vm = net.res_bus.vm_pu.values.astype(np.float32)          # (N,)
            line_p = net.res_line.p_from_mw.values.astype(np.float32)  # (n_lines,)
            trafo_p = net.res_trafo.p_hv_mw.values.astype(np.float32)  # (n_trafo,)
            flow = np.concatenate([line_p, trafo_p]).astype(np.float32)

            snapshots.append(Snapshot(
                pressure_true=torch.tensor(vm, dtype=torch.float32),
                flow_true=torch.tensor(flow, dtype=torch.float32),
                node_static=node_static,
                edge_static=edge_static,
                edge_index=bi_edge_index,
                edge_map=edge_map,
                scenario_id=sc,
                timestep=t,
            ))
    if n_fail:
        print(f"  ({n_fail} power-flow steps failed to converge and were skipped)")
    return snapshots


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="case118")
    p.add_argument("--scenarios", type=int, default=50)
    p.add_argument("--timesteps", type=int, default=24)
    p.add_argument("--seed", type=int, default=99)
    p.add_argument("--out_dir", default="data/temporal_moe_power")
    # Corruption config — mirror the Modena water dataset defaults.
    p.add_argument("--missing_rate", type=float, default=0.5)
    p.add_argument("--attack_fraction", type=float, default=0.15)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"Loading {args.case} ...")
    net = getattr(nw, args.case)()
    graph, edge_kind, _ = build_graph(net)
    print(f"Grid: {graph.num_nodes} buses, {graph.num_edges} edges "
          f"({int((edge_kind == 0).sum())} lines, "
          f"{int((edge_kind == 1).sum())} transformers)")

    print(f"Simulating {args.scenarios} scenarios x {args.timesteps} steps ...")
    snapshots = simulate(net, graph, edge_kind,
                         args.scenarios, args.timesteps, rng)
    print(f"Generated {len(snapshots)} clean snapshots "
          f"(vm range {snapshots[0].pressure_true.min():.3f}"
          f"–{snapshots[0].pressure_true.max():.3f} pu)")

    # Voltage magnitudes hug 1.0 pu; scale sensor noise to that range.
    # In the water dataset pressure ~24 m with 0.5 m noise (~2%); here
    # voltage ~1.0 pu, so use ~0.01 pu (~1%) so the attack difficulty is
    # comparable, and flows are ~tens of MW.
    corr_cfg = CorruptionConfig(
        missing_rate_pressure=args.missing_rate,
        missing_rate_flow=args.missing_rate,
        noise_sigma_pressure=0.01,
        noise_sigma_flow=2.0,
        attack_enabled=True,
        attack_fraction=args.attack_fraction,
        attack_bias=0.06,     # ~6% voltage bias for stealthy FDIA
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

    # Attack-class histogram for a sanity check.
    ids = np.array([c.attack_type_id for c in corrupted])
    names = ["clean", "random", "replay", "stealthy", "noise", "targeted"]
    hist = {names[i]: int((ids == i).sum()) for i in range(6)}
    print(f"Saved to {out}")
    print(f"Attack distribution: {hist}")


if __name__ == "__main__":
    main()
