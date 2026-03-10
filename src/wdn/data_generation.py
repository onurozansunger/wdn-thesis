"""Data generation pipeline: WNTR simulation → graph snapshots.

Loads an EPANET .inp network, runs hydraulic simulation, and produces
graph-structured snapshots with node features (pressure) and edge features (flow).
"""

from __future__ import annotations

import numpy as np
import torch
import wntr
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from wdn.config import GenerateConfig


# ---------------------------------------------------------------------------
# Network graph representation
# ---------------------------------------------------------------------------

@dataclass
class WDNGraph:
    """Static graph structure of a water distribution network.

    This is computed once from the .inp file and shared across all snapshots.
    """

    # Node info
    node_names: list[str]              # ordered list of node IDs
    node_types: np.ndarray             # (N,) int: 0=junction, 1=reservoir, 2=tank
    node_elevations: np.ndarray        # (N,)
    node_base_demands: np.ndarray      # (N,)
    node_coordinates: np.ndarray       # (N, 2) x,y positions for visualization

    # Edge info
    edge_names: list[str]              # ordered list of link IDs
    edge_types: np.ndarray             # (NE,) int: 0=pipe, 1=pump, 2=valve
    edge_lengths: np.ndarray           # (NE,)
    edge_diameters: np.ndarray         # (NE,)
    edge_roughness: np.ndarray         # (NE,)
    edge_index: np.ndarray             # (2, NE) source→target node indices

    # Graph matrices (for physics-informed loss & baselines)
    incidence_matrix: np.ndarray       # (N, NE) signed incidence matrix
    adjacency_matrix: np.ndarray       # (N, N) binary adjacency

    @property
    def num_nodes(self) -> int:
        return len(self.node_names)

    @property
    def num_edges(self) -> int:
        return len(self.edge_names)


@dataclass
class Snapshot:
    """A single timestep of the WDN simulation.

    Contains ground-truth values + static features as tensors
    ready for the model.
    """

    # Ground truth (what we want to reconstruct)
    pressure_true: torch.Tensor        # (N,) true pressure at each node
    flow_true: torch.Tensor            # (NE,) true flow at each edge

    # Static node features
    node_static: torch.Tensor          # (N, F_node) elevation, demand, type one-hot
    # Static edge features
    edge_static: torch.Tensor          # (NE, F_edge) length, diameter, roughness, type one-hot

    # Graph connectivity (bidirectional for message passing)
    edge_index: torch.Tensor           # (2, 2*NE) bidirectional edges

    # Mapping from bidirectional edge index back to original edge index
    # (needed to get flow predictions for the original directed edges)
    edge_map: torch.Tensor             # (2*NE,) index into original NE edges

    # Scenario and timestep metadata
    scenario_id: int = 0
    timestep: int = 0


# ---------------------------------------------------------------------------
# Build graph from WNTR network
# ---------------------------------------------------------------------------

def build_graph(wn: wntr.network.WaterNetworkModel) -> WDNGraph:
    """Extract the static graph structure from a WNTR network model."""

    # --- Nodes ---
    node_names = list(wn.node_name_list)
    node_name_to_idx = {name: i for i, name in enumerate(node_names)}
    N = len(node_names)

    node_types = np.zeros(N, dtype=np.int64)
    node_elevations = np.zeros(N, dtype=np.float32)
    node_base_demands = np.zeros(N, dtype=np.float32)
    node_coordinates = np.zeros((N, 2), dtype=np.float32)

    for i, name in enumerate(node_names):
        node = wn.get_node(name)
        coords = node.coordinates
        node_coordinates[i] = [coords[0], coords[1]] if coords else [0, 0]

        if isinstance(node, wntr.network.Junction):
            node_types[i] = 0
            node_elevations[i] = node.elevation
            # WNTR 1.4+: use demand_timeseries_list
            if len(node.demand_timeseries_list) > 0:
                node_base_demands[i] = node.demand_timeseries_list[0].base_value
            else:
                node_base_demands[i] = 0.0
        elif isinstance(node, wntr.network.Reservoir):
            node_types[i] = 1
            node_elevations[i] = node.base_head
        elif isinstance(node, wntr.network.Tank):
            node_types[i] = 2
            node_elevations[i] = node.elevation

    # --- Edges ---
    edge_names = list(wn.link_name_list)
    NE = len(edge_names)

    edge_types = np.zeros(NE, dtype=np.int64)
    edge_lengths = np.zeros(NE, dtype=np.float32)
    edge_diameters = np.zeros(NE, dtype=np.float32)
    edge_roughness = np.zeros(NE, dtype=np.float32)
    edge_index = np.zeros((2, NE), dtype=np.int64)

    for j, name in enumerate(edge_names):
        link = wn.get_link(name)
        src = node_name_to_idx[link.start_node_name]
        dst = node_name_to_idx[link.end_node_name]
        edge_index[0, j] = src
        edge_index[1, j] = dst

        if isinstance(link, wntr.network.Pipe):
            edge_types[j] = 0
            edge_lengths[j] = link.length
            edge_diameters[j] = link.diameter
            edge_roughness[j] = link.roughness
        elif isinstance(link, wntr.network.Pump):
            edge_types[j] = 1
            # Pumps don't have length/diameter/roughness in the same way
            edge_lengths[j] = 0.0
            edge_diameters[j] = 0.0
            edge_roughness[j] = 0.0
        elif isinstance(link, wntr.network.Valve):
            edge_types[j] = 2
            edge_lengths[j] = 0.0
            edge_diameters[j] = link.diameter
            edge_roughness[j] = 0.0

    # --- Incidence matrix (signed: +1 for outgoing, -1 for incoming) ---
    incidence = np.zeros((N, NE), dtype=np.float32)
    for j in range(NE):
        src, dst = edge_index[0, j], edge_index[1, j]
        incidence[src, j] = +1.0
        incidence[dst, j] = -1.0

    # --- Adjacency matrix ---
    adjacency = np.zeros((N, N), dtype=np.float32)
    for j in range(NE):
        src, dst = edge_index[0, j], edge_index[1, j]
        adjacency[src, dst] = 1.0
        adjacency[dst, src] = 1.0

    return WDNGraph(
        node_names=node_names,
        node_types=node_types,
        node_elevations=node_elevations,
        node_base_demands=node_base_demands,
        node_coordinates=node_coordinates,
        edge_names=edge_names,
        edge_types=edge_types,
        edge_lengths=edge_lengths,
        edge_diameters=edge_diameters,
        edge_roughness=edge_roughness,
        edge_index=edge_index,
        incidence_matrix=incidence,
        adjacency_matrix=adjacency,
    )


# ---------------------------------------------------------------------------
# Build static feature tensors
# ---------------------------------------------------------------------------

def _build_node_static(graph: WDNGraph) -> torch.Tensor:
    """Build static node feature matrix: [elevation, base_demand, type_onehot(3)]."""
    N = graph.num_nodes
    # Normalize elevation and demand to [0, 1] range
    elev = graph.node_elevations.copy()
    if elev.max() - elev.min() > 0:
        elev = (elev - elev.min()) / (elev.max() - elev.min())

    demand = graph.node_base_demands.copy()
    if demand.max() - demand.min() > 0:
        demand = (demand - demand.min()) / (demand.max() - demand.min())

    # One-hot node type (junction=0, reservoir=1, tank=2)
    type_onehot = np.eye(3, dtype=np.float32)[graph.node_types]

    features = np.column_stack([elev, demand, type_onehot])
    return torch.tensor(features, dtype=torch.float32)


def _build_edge_static(graph: WDNGraph) -> torch.Tensor:
    """Build static edge feature matrix: [length, diameter, roughness, type_onehot(3)]."""
    NE = graph.num_edges
    # Normalize
    length = graph.edge_lengths.copy()
    if length.max() - length.min() > 0:
        length = (length - length.min()) / (length.max() - length.min())

    diam = graph.edge_diameters.copy()
    if diam.max() - diam.min() > 0:
        diam = (diam - diam.min()) / (diam.max() - diam.min())

    rough = graph.edge_roughness.copy()
    if rough.max() - rough.min() > 0:
        rough = (rough - rough.min()) / (rough.max() - rough.min())

    type_onehot = np.eye(3, dtype=np.float32)[graph.edge_types]

    features = np.column_stack([length, diam, rough, type_onehot])
    return torch.tensor(features, dtype=torch.float32)


def _make_bidirectional(edge_index: np.ndarray, num_edges: int):
    """Convert directed edges to bidirectional for GNN message passing.

    Returns:
        bi_edge_index: (2, 2*NE) — original edges + reversed edges
        edge_map: (2*NE,) — maps each bidirectional edge to its original edge index
    """
    # Original: src→dst
    # Reversed: dst→src
    reversed_index = np.stack([edge_index[1], edge_index[0]], axis=0)
    bi_edge_index = np.concatenate([edge_index, reversed_index], axis=1)

    # edge_map: first NE entries map to 0..NE-1, next NE entries also map to 0..NE-1
    edge_map = np.concatenate([np.arange(num_edges), np.arange(num_edges)])

    return (
        torch.tensor(bi_edge_index, dtype=torch.long),
        torch.tensor(edge_map, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Run simulation and extract snapshots
# ---------------------------------------------------------------------------

def simulate_scenario(
    wn: wntr.network.WaterNetworkModel,
    graph: WDNGraph,
    scenario_id: int = 0,
    demand_multipliers: Optional[np.ndarray] = None,
) -> list[Snapshot]:
    """Run one hydraulic simulation and return a list of Snapshots (one per timestep).

    Args:
        wn: WNTR network model (will be modified in-place for demand variation).
        graph: Pre-built static graph.
        scenario_id: ID for this scenario.
        demand_multipliers: (N_junctions,) multipliers for base demands.
            If None, uses the original demands.

    Returns:
        List of Snapshot objects, one per simulation timestep.
    """
    # Apply demand multipliers if provided
    if demand_multipliers is not None:
        junction_idx = 0
        for name in graph.node_names:
            node = wn.get_node(name)
            if isinstance(node, wntr.network.Junction):
                if junction_idx < len(demand_multipliers):
                    # WNTR 1.4+: base_demand is read-only, use demand_timeseries_list
                    if len(node.demand_timeseries_list) > 0:
                        base = node.demand_timeseries_list[0].base_value
                        new_val = float(base * demand_multipliers[junction_idx])
                        node.demand_timeseries_list[0].base_value = new_val
                junction_idx += 1

    # Run simulation
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    # Extract pressure (nodes) and flow (links)
    # results.node["pressure"] is a DataFrame: index=time, columns=node_names
    pressure_df = results.node["pressure"]
    flow_df = results.link["flowrate"]

    # Pre-compute static features and bidirectional edges (shared across timesteps)
    node_static = _build_node_static(graph)
    edge_static = _build_edge_static(graph)
    bi_edge_index, edge_map = _make_bidirectional(graph.edge_index, graph.num_edges)

    # Reindex DataFrames once for vectorized access
    pressure_df = pressure_df[graph.node_names]
    flow_df = flow_df[graph.edge_names]

    snapshots = []
    for t_idx, time_sec in enumerate(pressure_df.index):
        p_values = pressure_df.iloc[t_idx].values.astype(np.float32)
        q_values = flow_df.iloc[t_idx].values.astype(np.float32)

        snap = Snapshot(
            pressure_true=torch.tensor(p_values, dtype=torch.float32),
            flow_true=torch.tensor(q_values, dtype=torch.float32),
            node_static=node_static,
            edge_static=edge_static,
            edge_index=bi_edge_index,
            edge_map=edge_map,
            scenario_id=scenario_id,
            timestep=t_idx,
        )
        snapshots.append(snap)

    return snapshots


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_dataset(cfg: GenerateConfig) -> tuple[WDNGraph, list[Snapshot]]:
    """Generate the full dataset: multiple scenarios × timesteps.

    Returns:
        graph: The static WDN graph structure.
        snapshots: List of all snapshots across all scenarios.
    """
    rng = np.random.default_rng(cfg.seed)

    # Load network
    inp_path = Path(cfg.network_inp)
    if not inp_path.exists():
        raise FileNotFoundError(f"Network file not found: {inp_path}")

    wn = wntr.network.WaterNetworkModel(str(inp_path))

    # Build static graph
    graph = build_graph(wn)

    # Count junctions (for demand variation)
    n_junctions = int((graph.node_types == 0).sum())

    all_snapshots: list[Snapshot] = []

    print(f"Generating {cfg.num_scenarios} scenarios on {inp_path.name} "
          f"({graph.num_nodes} nodes, {graph.num_edges} edges)...")

    for s in range(cfg.num_scenarios):
        # Reload network fresh each scenario (to reset demands)
        wn = wntr.network.WaterNetworkModel(str(inp_path))

        # Random demand multipliers
        if cfg.demand_variation > 0:
            low = 1.0 - cfg.demand_variation
            high = 1.0 + cfg.demand_variation
            multipliers = rng.uniform(low, high, size=n_junctions).astype(np.float32)
        else:
            multipliers = None

        snapshots = simulate_scenario(wn, graph, scenario_id=s, demand_multipliers=multipliers)
        all_snapshots.extend(snapshots)

        if (s + 1) % 10 == 0 or s == 0:
            print(f"  Scenario {s + 1}/{cfg.num_scenarios} done "
                  f"({len(snapshots)} timesteps each)")

    print(f"Total snapshots: {len(all_snapshots)}")
    return graph, all_snapshots
