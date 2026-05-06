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
    "Net1":   {"label": "Net1 (11 nodes, 13 pipes)",     "nodes": 11,  "edges": 13},
    "Net3":   {"label": "Net3 (97 nodes, 117 pipes)",    "nodes": 97,  "edges": 117},
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


# ── Net3 loaders ──
# The export scripts (dashboard/precompute/export_net3_demo.py and
# export_attack_analysis_net3.py) write the same JSON shape as the
# Modena ones, so the existing pages work out of the box.

@st.cache_data
def _load_net3_demo_raw():
    p = PROJECT_ROOT / "dashboard" / "data" / "net3_demo.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


@st.cache_resource
def load_graph_net3():
    """Build a WDNGraph-like object from the Net3 demo JSON, falling
    back to the raw graph.pkl if the precomputed demo isn't there."""
    raw = _load_net3_demo_raw()
    if raw is not None:
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
    p = PROJECT_ROOT / "data" / "temporal_moe_net3" / "graph.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


@st.cache_data
def load_test_results_net3():
    raw = _load_net3_demo_raw()
    if raw and "test_results" in raw:
        return raw["test_results"]
    p = PROJECT_ROOT / "runs" / "temporal_moe" / "20260505_150656" / "test_results.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


@st.cache_data
def load_history_net3():
    raw = _load_net3_demo_raw()
    if raw and "history" in raw:
        return raw["history"]
    p = PROJECT_ROOT / "runs" / "temporal_moe" / "20260505_150656" / "history.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


@st.cache_data
def load_demo_snapshots_net3():
    raw = _load_net3_demo_raw()
    if raw is None:
        return None
    g = raw["graph"]
    return {"node_names": g["node_names"], "edge_names": g["edge_names"],
            "snapshots": raw["snapshots"]}


@st.cache_data
def load_attack_analysis_net3():
    p = PROJECT_ROOT / "dashboard" / "data" / "attack_analysis_net3.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


# ── Unified dispatchers ──

_DISPATCH = {
    "Net1":   {"graph": load_graph_net1,
               "test_results": load_test_results_net1,
               "history": load_history_net1,
               "demo": load_demo_snapshots_net1,
               "attack_analysis": load_attack_analysis_net1},
    "Net3":   {"graph": load_graph_net3,
               "test_results": load_test_results_net3,
               "history": load_history_net3,
               "demo": load_demo_snapshots_net3,
               "attack_analysis": load_attack_analysis_net3},
    "Modena": {"graph": load_graph_modena,
               "test_results": load_test_results_modena,
               "history": load_history_modena,
               "demo": load_demo_snapshots_modena,
               "attack_analysis": load_attack_analysis_modena},
}


def load_graph(network="Net1"):
    return _DISPATCH.get(network, _DISPATCH["Net1"])["graph"]()


def load_test_results(network="Net1"):
    return _DISPATCH.get(network, _DISPATCH["Net1"])["test_results"]()


def load_history(network="Net1"):
    return _DISPATCH.get(network, _DISPATCH["Net1"])["history"]()


def load_demo_snapshots(network="Net1"):
    return _DISPATCH.get(network, _DISPATCH["Net1"])["demo"]()


def load_attack_analysis(network="Net1"):
    return _DISPATCH.get(network, _DISPATCH["Net1"])["attack_analysis"]()


# ── Temporal model loaders ──

TEMPORAL_RUNS = {
    "Net1": "20260412_114523",
    "Net3": "20260505_150656",   # temporal-MoE Net3 (no separate temporal run)
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


# ── MoE loaders ──

@st.cache_data
def load_moe_results(network="Net1", variant="spatial"):
    """Load MoE test results for a given network and variant.

    variant: "spatial" -> runs/moe/<run_id>/
             "temporal" -> runs/temporal_moe/<run_id>/
    """
    if variant == "spatial":
        run_map = {"Net1": "20260414_162712", "Modena": "20260414_185219"}
        base = PROJECT_ROOT / "runs" / "moe"
    else:
        base = PROJECT_ROOT / "runs" / "temporal_moe"
        # Best runs picked manually by per-attack F1 sweep:
        #  Net1 prefers the smaller 3-feature setup (less overfitting on
        #  the tiny network), Modena gets the bigger 6-feature one.
        run_map = {
            "Net1":   "20260425_161000",
            "Net3":   "20260505_150656",
            "Modena": "20260425_170314",
        }

    run_id = run_map.get(network)
    if not run_id:
        return None
    p = base / run_id / "test_results.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


@st.cache_data
def load_moe_history(network="Net1", variant="spatial"):
    if variant == "spatial":
        run_map = {"Net1": "20260414_162712", "Modena": "20260414_185219"}
        base = PROJECT_ROOT / "runs" / "moe"
    else:
        base = PROJECT_ROOT / "runs" / "temporal_moe"
        run_map = {
            "Net1":   "20260425_161000",
            "Net3":   "20260505_150656",
            "Modena": "20260425_170314",
        }

    run_id = run_map.get(network)
    if not run_id:
        return None
    p = base / run_id / "history.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None



# ── Self-play loaders ──

SELFPLAY_RUNS = {
    "Modena": {
        "single":      "20260505_223529",
        "scratch":     "20260505_215710",
        "attacker_moe":"20260506_110630",
        "seed1":       "20260506_003725",
        "seed2":       "20260506_005944",
        "seed3":       "20260506_012245",
    },
    "Net1":   {"single": "20260505_233503"},
    "Net3":   {"single": "20260505_232120"},
}


@st.cache_data
def load_selfplay_history(network="Modena", variant="single"):
    """Load the per-epoch ``history.json`` for a self-play run."""
    run_id = SELFPLAY_RUNS.get(network, {}).get(variant)
    if not run_id:
        return None
    p = PROJECT_ROOT / "runs" / "selfplay" / run_id / "history.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


@st.cache_data
def load_selfplay_args(network="Modena", variant="single"):
    run_id = SELFPLAY_RUNS.get(network, {}).get(variant)
    if not run_id:
        return None
    p = PROJECT_ROOT / "runs" / "selfplay" / run_id / "args.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


# GNN backbone ablation — best (network, gnn_type) -> multitask run id.
# Picked by per-network max F1 from the today's queue.
GNN_ABLATION_RUNS = {
    "Net1": {
        "GraphSAGE":   "20260505_162622",
        "GAT":         "20260505_162827",
        "GCN":         "20260505_163009",
        "Transformer": "20260505_163157",
    },
    "Net3": {
        "GraphSAGE":   "20260505_163339",
        "GAT":         "20260505_163110",
        "GCN":         "20260505_163256",
        "Transformer": "20260505_163453",
    },
    "Modena": {
        "GraphSAGE":   "20260505_153539",
        "GAT":         "20260505_153737",
        "GCN":         "20260505_153918",
        "Transformer": "20260505_154140",
    },
}


@st.cache_data
def load_gnn_ablation():
    """Return {network: {gnn: test_results_dict}}."""
    out = {}
    for net, gnn_map in GNN_ABLATION_RUNS.items():
        out[net] = {}
        for gnn, run_id in gnn_map.items():
            p = PROJECT_ROOT / "runs" / "multitask" / run_id / "test_results.json"
            if p.exists():
                with open(p) as f:
                    out[net][gnn] = json.load(f)
    return out


@st.cache_data
def load_selfplay_summary():
    """Load the aggregate JSON files produced by scripts/eval_*.py."""
    out = {}
    for name, fname in (
        ("atkmoe",    "eval_atkmoe.json"),
        ("multiseed", "eval_multiseed.json"),
        ("three",     "eval_all.json"),
        ("heldout",   "eval_heldout.json"),
    ):
        p = PROJECT_ROOT / "runs" / "selfplay" / fname
        if p.exists():
            with open(p) as f:
                out[name] = json.load(f)
    return out
