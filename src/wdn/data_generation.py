from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import wntr

from wdn.config import GenerateConfig
from wdn.corruption import apply_corruption
from wdn.utils import set_seed


NODE_TYPES = {"Junction": 0, "Reservoir": 1, "Tank": 2}
LINK_TYPES = {"Pipe": 0, "Pump": 1, "Valve": 2}


def _one_hot(index: int, size: int) -> List[float]:
    vec = [0.0] * size
    if index >= 0:
        vec[index] = 1.0
    return vec


def load_network(inp_path: str) -> wntr.network.model.WaterNetworkModel:
    return wntr.network.io.read_inpfile(inp_path)


def build_graph(wn: wntr.network.model.WaterNetworkModel) -> Tuple[List[str], List[str], np.ndarray]:
    node_names = list(wn.node_name_list)
    link_names = list(wn.link_name_list)

    node_index = {name: i for i, name in enumerate(node_names)}
    edges = []
    for link_name in link_names:
        link = wn.get_link(link_name)
        start = node_index[link.start_node_name]
        end = node_index[link.end_node_name]
        edges.append((start, end))
    edge_index = np.array(edges, dtype=np.int64).T
    return node_names, link_names, edge_index


def node_static_features(wn: wntr.network.model.WaterNetworkModel, node_names: List[str]) -> np.ndarray:
    features = []
    for name in node_names:
        node = wn.get_node(name)
        elevation = getattr(node, "elevation", 0.0)
        base_demand = 0.0
        if hasattr(node, "base_demand") and node.base_demand is not None:
            base_demand = float(node.base_demand)
        node_type_idx = NODE_TYPES.get(node.node_type, -1)
        features.append([elevation, base_demand] + _one_hot(node_type_idx, len(NODE_TYPES)))
    return np.asarray(features, dtype=np.float32)


def edge_static_features(wn: wntr.network.model.WaterNetworkModel, link_names: List[str]) -> np.ndarray:
    features = []
    for name in link_names:
        link = wn.get_link(name)
        length = getattr(link, "length", 0.0)
        diameter = getattr(link, "diameter", 0.0)
        roughness = getattr(link, "roughness", 0.0)
        link_type_idx = LINK_TYPES.get(link.link_type, -1)
        features.append([length, diameter, roughness] + _one_hot(link_type_idx, len(LINK_TYPES)))
    return np.asarray(features, dtype=np.float32)


def run_simulation(wn: wntr.network.model.WaterNetworkModel) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    pressures = results.node["pressure"]
    flows = results.link["flowrate"]
    return pressures, flows


def generate_dataset(config: GenerateConfig) -> Dict[str, np.ndarray]:
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    wn = load_network(config.inp_path)
    node_names, link_names, edge_index = build_graph(wn)
    node_static = node_static_features(wn, node_names)
    edge_static = edge_static_features(wn, link_names)

    pressures, flows = run_simulation(wn)

    times = list(range(config.time_start, config.time_end + 1, config.time_step))
    time_seconds = [t * 3600 for t in times]

    p_true_list = []
    q_true_list = []
    p_obs_list = []
    q_obs_list = []
    p_mask_list = []
    q_mask_list = []
    p_anom_list = []
    q_anom_list = []

    node_index = {name: i for i, name in enumerate(node_names)}
    link_index = {name: i for i, name in enumerate(link_names)}

    target_node_idx = [node_index[n] for n in config.corruption.target_nodes if n in node_index]
    target_edge_idx = [link_index[e] for e in config.corruption.target_edges if e in link_index]

    time_window = None
    if config.corruption.time_window and len(config.corruption.time_window) == 2:
        time_window = (config.corruption.time_window[0], config.corruption.time_window[1])

    snapshot_indices = []
    time_used = []
    for i, t in enumerate(time_seconds):
        if t not in pressures.index:
            continue
        p_true = pressures.loc[t, node_names].to_numpy(dtype=np.float32)
        q_true = flows.loc[t, link_names].to_numpy(dtype=np.float32)

        corr = apply_corruption(
            rng=rng,
            p_true=p_true,
            q_true=q_true,
            missing_p=config.corruption.missing_p,
            missing_q=config.corruption.missing_q,
            noise_sigma_p=config.corruption.noise_sigma_p,
            noise_sigma_q=config.corruption.noise_sigma_q,
            attack_enabled=config.corruption.attack_enabled,
            attack_fraction=config.corruption.attack_fraction,
            attack_bias_p=config.corruption.attack_bias_p,
            attack_bias_q=config.corruption.attack_bias_q,
            attack_scale_p=config.corruption.attack_scale_p,
            attack_scale_q=config.corruption.attack_scale_q,
            targeted=config.corruption.targeted,
            target_node_idx=target_node_idx,
            target_edge_idx=target_edge_idx,
            time_index=i,
            time_window=time_window,
        )

        p_true_list.append(p_true)
        q_true_list.append(q_true)
        p_obs_list.append(corr.p_obs)
        q_obs_list.append(corr.q_obs)
        p_mask_list.append(corr.p_mask)
        q_mask_list.append(corr.q_mask)
        p_anom_list.append(corr.p_anom)
        q_anom_list.append(corr.q_anom)
        snapshot_indices.append(i)
        time_used.append(t)

    network_name = wn.name if wn.name else Path(config.inp_path).stem
    dataset = {
        "node_names": np.array(node_names, dtype=object),
        "edge_names": np.array(link_names, dtype=object),
        "edge_index": edge_index.astype(np.int64),
        "node_static": node_static.astype(np.float32),
        "edge_static": edge_static.astype(np.float32),
        "times": np.array(time_used, dtype=np.int64),
        "snapshot_index": np.array(snapshot_indices, dtype=np.int64),
        "network_name": np.array([network_name] * len(p_true_list), dtype=object),
        "seed": np.array([config.seed] * len(p_true_list), dtype=np.int64),
        "P_true": np.stack(p_true_list, axis=0),
        "Q_true": np.stack(q_true_list, axis=0),
        "P_obs": np.stack(p_obs_list, axis=0),
        "Q_obs": np.stack(q_obs_list, axis=0),
        "P_mask": np.stack(p_mask_list, axis=0),
        "Q_mask": np.stack(q_mask_list, axis=0),
        "P_anom": np.stack(p_anom_list, axis=0),
        "Q_anom": np.stack(q_anom_list, axis=0),
    }
    return dataset


def save_dataset(dataset: Dict[str, np.ndarray], config: GenerateConfig) -> Path:
    output_dir = Path(config.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / config.output_name
    np.savez_compressed(output_path, **dataset)

    meta_path = output_dir / f"{Path(config.output_name).stem}_meta.yaml"
    meta = asdict(config)
    with open(meta_path, "w", encoding="utf-8") as f:
        import yaml

        yaml.safe_dump(meta, f, sort_keys=False)

    return output_path
