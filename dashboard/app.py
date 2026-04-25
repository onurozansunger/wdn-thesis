"""WDN Thesis Dashboard — Landing page."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from utils.theme import GLOBAL_CSS
from utils.data_loader import (
    load_test_results_net1, load_test_results_modena,
    load_temporal_results, load_moe_results,
)

st.set_page_config(
    page_title="WDN State Reconstruction",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

st.title("WDN State Reconstruction & Anomaly Detection")
st.caption("Graph Neural Networks for cyber-physical monitoring of water distribution networks")

st.divider()

# ── Headline numbers (best model on each network) ──
net1 = load_test_results_net1()
mod = load_test_results_modena()
net1_moe_t = load_moe_results("Net1", "temporal")
mod_moe_t = load_moe_results("Modena", "temporal")


def _replay_f1(res):
    if not res:
        return None
    pa = res.get("per_attack_pressure", {}).get("replay", {})
    return pa.get("f1")


col_n, col_m = st.columns(2)
with col_n:
    st.markdown("**Net1** — 11 nodes, 13 pipes")
    a, b, c = st.columns(3)
    a.metric("P MAE (unobs)", f"{net1['reconstruction']['pressure_unobs']['mae']:.2f} m")
    b.metric("Anomaly F1", f"{net1['anomaly_detection']['pressure']['f1']:.3f}")
    rep = _replay_f1(net1_moe_t)
    c.metric("Replay F1 (MoE)", f"{rep:.3f}" if rep is not None else "—")

with col_m:
    st.markdown("**Modena** — 272 nodes, 317 pipes")
    a, b, c = st.columns(3)
    a.metric("P MAE (unobs)", f"{mod['reconstruction']['pressure_unobs']['mae']:.2f} m")
    b.metric("Anomaly F1", f"{mod['anomaly_detection']['pressure']['f1']:.3f}")
    rep = _replay_f1(mod_moe_t)
    c.metric("Replay F1 (MoE)", f"{rep:.3f}" if rep is not None else "—")

st.divider()

col_l, col_r = st.columns([3, 2], gap="large")

with col_l:
    st.markdown(
        """
        **Approach.** The pipe network is modelled as a graph and a
        Graph Neural Network jointly reconstructs missing sensor values
        and flags compromised sensors.

        - **Physics-informed.** A mass-conservation penalty keeps
          predictions hydraulically plausible.
        - **Multi-task.** A shared backbone serves both reconstruction
          and anomaly detection.
        - **Temporal.** A GRU over a 6-step sliding window captures
          time-dependent attack patterns (notably replay).
        - **Mixture-of-Experts.** Six attack-specialised experts plus a
          learned router target each cyber-attack class on its own
          terms.
        - **Explainable.** GNNExplainer surfaces which nodes and
          features drive each prediction.
        """
    )

with col_r:
    st.markdown(
        """
        | Page | Content |
        |------|---------|
        | Network Overview | Topology and per-node properties |
        | Reconstruction | True vs predicted pressure / flow |
        | Attack Analysis | Per-attack-type detection curves |
        | Anomaly Detection | Interactive thresholding demo |
        | Model Comparison | Spatial → Temporal → MoE benchmarks |
        | Explainability | Feature and node-importance maps |
        """
    )
