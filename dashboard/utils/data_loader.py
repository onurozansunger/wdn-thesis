"""Cached data loading for the dashboard."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent

NETWORKS = {
    "Net1": {"label": "Net1 (11 nodes, 13 pipes)", "nodes": 11, "edges": 13},
    "Modena": {"label": "Modena (272 nodes, 317 pipes)", "nodes": 272, "edges": 317},
}


def network_selector(key="network_selector"):
    """Render a network selector radio and return the chosen network name."""
    return st.radio(
        "Network", list(NETWORKS.keys()),
        format_func=lambda k: NETWORKS[k]["label"],
        horizontal=True, key=key,
    )


# ── Net1 loaders ──

@st.cache_resource
def load_graph_net1():
    with open(PROJECT_ROOT / "data" / "generated_attacks" / "graph.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_test_results_net1():
    with open(PROJECT_ROOT / "runs" / "multitask" / "20260310_201113" / "test_results.json") as f:
        return json.load(f)


@st.cache_data
def load_history_net1():
    with open(PROJECT_ROOT / "runs" / "multitask" / "20260310_201113" / "history.json") as f:
        return json.load(f)


@st.cache_data
def load_demo_snapshots_net1():
    p = PROJECT_ROOT / "dashboard" / "data" / "demo_snapshots.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


@st.cache_data
def load_attack_analysis_net1():
    p = PROJECT_ROOT / "dashboard" / "data" / "attack_analysis.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


# ── Modena loaders ──

@st.cache_data
def _load_modena_demo_raw():
    p = PROJECT_ROOT / "dashboard" / "data" / "modena_demo.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


@st.cache_resource
def load_graph_modena():
    """Build a WDNGraph-like object from the Modena demo JSON."""
    raw = _load_modena_demo_raw()
    if raw is None:
        return None
    g = raw["graph"]
    return SimpleNamespace(
        node_names=g["node_names"],
        node_types=np.array(g["node_types"]),
        node_elevations=np.zeros(g["num_nodes"]),
        node_base_demands=np.zeros(g["num_nodes"]),
        node_coordinates=np.array(g["node_coordinates"]),
        edge_names=g["edge_names"],
        edge_types=np.zeros(g["num_edges"], dtype=int),
        edge_lengths=np.zeros(g["num_edges"]),
        edge_diameters=np.zeros(g["num_edges"]),
        edge_roughness=np.zeros(g["num_edges"]),
        edge_index=np.array(g["edge_index"]),
        num_nodes=g["num_nodes"],
        num_edges=g["num_edges"],
    )


@st.cache_data
def load_test_results_modena():
    raw = _load_modena_demo_raw()
    return raw["test_results"] if raw else None


@st.cache_data
def load_history_modena():
    raw = _load_modena_demo_raw()
    return raw["history"] if raw else None


@st.cache_data
def load_demo_snapshots_modena():
    raw = _load_modena_demo_raw()
    if raw is None:
        return None
    g = raw["graph"]
    return {"node_names": g["node_names"], "edge_names": g["edge_names"], "snapshots": raw["snapshots"]}


@st.cache_data
def load_attack_analysis_modena():
    p = PROJECT_ROOT / "dashboard" / "data" / "attack_analysis_modena.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


# ── Unified dispatchers ──

def load_graph(network="Net1"):
    return load_graph_net1() if network == "Net1" else load_graph_modena()


def load_test_results(network="Net1"):
    return load_test_results_net1() if network == "Net1" else load_test_results_modena()


def load_history(network="Net1"):
    return load_history_net1() if network == "Net1" else load_history_modena()


def load_demo_snapshots(network="Net1"):
    return load_demo_snapshots_net1() if network == "Net1" else load_demo_snapshots_modena()


def load_attack_analysis(network="Net1"):
    return load_attack_analysis_net1() if network == "Net1" else load_attack_analysis_modena()


# ── Temporal model loaders ──

TEMPORAL_RUNS = {
    "Net1": "20260412_114523",
    "Modena": "20260412_170210",
}


@st.cache_data
def load_temporal_results(network="Net1"):
    run_id = TEMPORAL_RUNS.get(network)
    if not run_id:
        return None
    p = PROJECT_ROOT / "runs" / "temporal" / run_id / "test_results.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


@st.cache_data
def load_temporal_history(network="Net1"):
    run_id = TEMPORAL_RUNS.get(network)
    if not run_id:
        return None
    p = PROJECT_ROOT / "runs" / "temporal" / run_id / "history.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


# ── Net1-only loaders (no Modena equivalent yet) ──

@st.cache_data
def load_architecture_comparison():
    with open(PROJECT_ROOT / "data" / "architecture_comparison.json") as f:
        return json.load(f)


@st.cache_data
def load_baseline_comparison():
    with open(PROJECT_ROOT / "data" / "comparison_30_50.json") as f:
        comp = json.load(f)
    with open(PROJECT_ROOT / "data" / "generated" / "baseline_results.json") as f:
        baselines = json.load(f)
    return {"comparison": comp, "baselines": baselines}


@st.cache_data
def load_sensor_oracle():
    oracle_dir = PROJECT_ROOT / "runs" / "sensor_oracle" / "20260325_204635"
    results = {}
    for name in ("node_ranking.json", "greedy_placement.json"):
        p = oracle_dir / name
        if p.exists():
            with open(p) as f:
                results[name.replace(".json", "")] = json.load(f)
    return results
