"""Cached data loading for the dashboard."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent


@st.cache_resource
def load_graph():
    with open(PROJECT_ROOT / "data" / "generated_attacks" / "graph.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_test_results():
    with open(PROJECT_ROOT / "runs" / "multitask" / "20260310_201113" / "test_results.json") as f:
        return json.load(f)


@st.cache_data
def load_history():
    with open(PROJECT_ROOT / "runs" / "multitask" / "20260310_201113" / "history.json") as f:
        return json.load(f)


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


@st.cache_data
def load_demo_snapshots():
    p = PROJECT_ROOT / "dashboard" / "data" / "demo_snapshots.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None
