"""WDN Thesis Dashboard — Landing page."""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.theme import GLOBAL_CSS
from utils.data_loader import load_test_results_net1, load_test_results_modena, load_temporal_results

st.set_page_config(
    page_title="WDN State Reconstruction",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

st.title("WDN State Reconstruction & Anomaly Detection")
st.caption("GNN-based monitoring for water distribution networks")

st.divider()

# ── Results overview ──
net1 = load_test_results_net1()
modena = load_test_results_modena()
temp_net1 = load_temporal_results("Net1")
temp_modena = load_temporal_results("Modena")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Net1** — 11 nodes, 13 pipes")
    c1, c2, c3 = st.columns(3)
    c1.metric("P MAE (unobs)", f"{net1['reconstruction']['pressure_unobs']['mae']:.2f} m")
    c2.metric("Anomaly F1", f"{net1['anomaly_detection']['pressure']['f1']:.3f}")
    c3.metric("AUROC", f"{net1['anomaly_detection']['pressure']['auroc']:.3f}")

with col2:
    st.markdown("**Modena** — 272 nodes, 317 pipes")
    c1, c2, c3 = st.columns(3)
    c1.metric("P MAE (unobs)", f"{modena['reconstruction']['pressure_unobs']['mae']:.2f} m")
    c2.metric("Anomaly F1", f"{modena['anomaly_detection']['pressure']['f1']:.3f}")
    c3.metric("AUROC", f"{modena['anomaly_detection']['pressure']['auroc']:.3f}")

if temp_net1 and temp_modena:
    st.divider()
    st.markdown("##### Temporal Model (GNN + GRU)")
    col1, col2 = st.columns(2)
    with col1:
        c1, c2, c3 = st.columns(3)
        c1.metric("P MAE", f"{temp_net1['reconstruction']['pressure_unobs']['mae']:.3f} m",
                   delta=f"{(net1['reconstruction']['pressure_unobs']['mae'] - temp_net1['reconstruction']['pressure_unobs']['mae']):.3f} m")
        c2.metric("F1", f"{temp_net1['anomaly_detection']['pressure']['f1']:.3f}")
        c3.metric("AUROC", f"{temp_net1['anomaly_detection']['pressure']['auroc']:.3f}")
    with col2:
        c1, c2, c3 = st.columns(3)
        c1.metric("P MAE", f"{temp_modena['reconstruction']['pressure_unobs']['mae']:.3f} m",
                   delta=f"{(modena['reconstruction']['pressure_unobs']['mae'] - temp_modena['reconstruction']['pressure_unobs']['mae']):.3f} m")
        c2.metric("F1", f"{temp_modena['anomaly_detection']['pressure']['f1']:.3f}")
        c3.metric("AUROC", f"{temp_modena['anomaly_detection']['pressure']['auroc']:.3f}")

st.divider()

col_l, col_r = st.columns([3, 2], gap="large")

with col_l:
    st.markdown("""
    **Approach**: Model the pipe network as a graph and apply Graph Neural Networks
    to jointly reconstruct missing sensor values and detect compromised sensors.

    - **Physics-informed**: mass conservation penalty keeps predictions physically plausible
    - **Multi-task**: shared backbone improves both reconstruction and anomaly detection
    - **Temporal**: GRU over 6-hour sliding windows captures time-dependent attack patterns
    - **Explainable**: GNNExplainer reveals which nodes and features drive predictions
    """)

with col_r:
    st.markdown("""
    | Page | What it shows |
    |------|--------------|
    | Network Overview | Topology and properties |
    | Reconstruction | True vs predicted states |
    | Attack Analysis | Per-attack detection |
    | Anomaly Detection | Interactive threshold |
    | Model Comparison | Benchmarks and baselines |
    | Training History | Loss curves and metrics |
    | Explainability | Feature/node importance |
    """)
