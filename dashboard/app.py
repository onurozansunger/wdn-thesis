"""WDN Thesis Dashboard — Landing page."""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.theme import GLOBAL_CSS, BLUE, GREEN, ORANGE, CYAN

st.set_page_config(
    page_title="WDN State Reconstruction & Anomaly Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ── Header ──
st.markdown("""
<div style="text-align:center; padding: 1rem 0 0.5rem 0;">
    <h1 style="font-size:2.2rem; font-weight:700; margin-bottom:0.2rem;">
        WDN State Reconstruction & Anomaly Detection
    </h1>
    <p style="opacity:0.5; font-size:0.95rem; margin-top:0;">
        Graph Neural Networks for Water Distribution Network Monitoring
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Key Results ──
c1, c2, c3, c4 = st.columns(4)
c1.metric("Pressure MAE", "1.42 m", delta="-17.0 vs baseline", delta_color="inverse")
c2.metric("Anomaly AUROC", "0.883")
c3.metric("Improvement over WLS", "12.7x")
c4.metric("Anomaly Precision", "93.6%")

st.divider()

# ── Content ──
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown(f"""
    #### The Problem
    Water utilities deploy pressure and flow sensors across their pipe networks.
    In practice, **50% or more of these sensors can be offline** at any given time
    due to hardware failures, communication drops, or maintenance. Worse, active
    sensors can be **compromised by cyber-attacks** — injecting false readings
    that mask leaks, contamination, or theft.

    #### Our Approach
    We model the pipe network as a **graph** and apply Graph Neural Networks to
    jointly solve two tasks from the same shared backbone:

    1. **State reconstruction** — predict pressure at every node and flow at every
       pipe, even where sensors are missing, using spatial context from neighboring
       nodes through message passing
    2. **Anomaly detection** — classify each observed sensor as clean or compromised
       by learning the difference between normal reconstruction residuals and
       attack-induced deviations

    #### Key Design Choices
    - **Physics-informed training**: a mass-conservation penalty (Kirchhoff's law)
      regularizes predicted flows so they remain physically plausible
    - **MC Dropout uncertainty**: running multiple stochastic forward passes at
      inference time gives per-node confidence intervals — nodes with high variance
      are candidates for new sensor installations
    - **Multi-task learning**: the shared GNN backbone means better reconstruction
      also improves anomaly detection, and vice versa
    """)

with right:
    st.markdown("""
    #### Dashboard Pages
    | Page | Description |
    |------|-------------|
    | **Network Overview** | Topology and properties of the water network |
    | **Reconstruction** | Side-by-side comparison of observed, predicted, and true states |
    | **Anomaly Detection** | Interactive attack detection with adjustable threshold |
    | **Model Comparison** | GNN architectures benchmarked against analytical baselines |
    | **Sensor Oracle** | Optimal placement strategy based on uncertainty |
    | **Training History** | Loss curves and validation metrics over training |
    """)

    st.markdown("""
    #### Technical Details
    | | |
    |---|---|
    | Network | EPANET Net1 (11 nodes, 13 edges) |
    | Data | 50 scenarios, 24h each, 1250 snapshots |
    | Missing rate | 50% of sensors (Bernoulli) |
    | Attacks | 5 types (replay, stealthy bias, falsification, noise, targeted) |
    | Best architecture | GraphSAGE (30k params, 95s training) |
    | Uncertainty | MC Dropout, 30 forward passes |
    """)

st.divider()
st.caption("EPANET Net1 benchmark network — synthetic data generated with WNTR")
