"""Page 2: State Reconstruction Demo."""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_graph, load_demo_snapshots, load_test_results
from utils.network_viz import (
    build_network_figure, PRESSURE_COLORSCALE, ERROR_COLORSCALE, TYPE_NAMES,
)
from utils.theme import GLOBAL_CSS, plotly_layout, BLUE, ORANGE, GREEN, DIM

st.set_page_config(page_title="Reconstruction", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

st.title("State Reconstruction")
st.caption("Recovering full network pressures from 50% missing, noisy sensor data")

graph = load_graph()
demo = load_demo_snapshots()
test_results = load_test_results()

if demo is None:
    st.error("Demo data not found. Run: `python dashboard/precompute/export_demo_data.py`")
    st.stop()

snapshots = demo["snapshots"]
node_names = demo["node_names"]

# ── Metrics ──
c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE (all nodes)", f"{test_results['reconstruction']['pressure_all']['mae']:.2f} m")
c2.metric("MAE (unobserved)", f"{test_results['reconstruction']['pressure_unobs']['mae']:.2f} m")
c3.metric("RMSE (unobserved)", f"{test_results['reconstruction']['pressure_unobs']['rmse']:.2f} m")
c4.metric("Missing Rate", "50%")

st.divider()

# ── Snapshot selector ──
snap_idx = st.selectbox(
    "Select a test snapshot to inspect",
    range(len(snapshots)),
    format_func=lambda i: f"Snapshot {i+1}  (test index {snapshots[i]['index']})",
)

snap = snapshots[snap_idx]
p_true = np.array(snap["pressure_true"])
p_pred = np.array(snap["pressure_pred"])
p_obs = np.array(snap["pressure_obs"])
p_mask = np.array(snap["pressure_mask"])
p_error = np.array(snap["pressure_error"])

vmin = min(p_true.min(), p_pred.min())
vmax = max(p_true.max(), p_pred.max())

# ── Three-panel comparison ──
col1, col2, col3 = st.columns(3)

with col1:
    fig = build_network_figure(
        graph, node_values=p_true, colorscale=PRESSURE_COLORSCALE,
        title="Ground Truth", color_label="Pressure (m)",
        height=420, node_size=30,
    )
    fig.update_traces(marker=dict(cmin=vmin, cmax=vmax), selector=dict(mode="markers+text"))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Observed: dim missing nodes
    obs_text = []
    for i in range(graph.num_nodes):
        t = TYPE_NAMES.get(graph.node_types[i], "?")
        if p_mask[i] > 0:
            obs_text.append(f"<b>Node {node_names[i]}</b><br>Type: {t}<br>Reading: {p_obs[i]:.2f} m")
        else:
            obs_text.append(f"<b>Node {node_names[i]}</b><br>Type: {t}<br><i>Sensor offline</i>")

    fig = build_network_figure(graph, title="Observed (50% missing)", height=420, node_size=30)
    marker_colors = [p_obs[i] if p_mask[i] > 0 else vmin for i in range(graph.num_nodes)]
    marker_opacities = [1.0 if p_mask[i] > 0 else 0.15 for i in range(graph.num_nodes)]
    fig.update_traces(
        marker=dict(
            color=marker_colors, colorscale=PRESSURE_COLORSCALE,
            cmin=vmin, cmax=vmax, opacity=marker_opacities,
            showscale=True,
            colorbar=dict(title=dict(text="Pressure (m)", font=dict(size=11, color=DIM)),
                          thickness=14, len=0.55, outlinewidth=0, bgcolor="rgba(0,0,0,0)",
                          tickfont=dict(size=10, color=DIM)),
        ),
        hovertext=obs_text,
        selector=dict(mode="markers+text"),
    )
    st.plotly_chart(fig, use_container_width=True)

with col3:
    fig = build_network_figure(
        graph, node_values=p_pred, colorscale=PRESSURE_COLORSCALE,
        title="GNN Prediction", color_label="Pressure (m)",
        height=420, node_size=30,
    )
    fig.update_traces(marker=dict(cmin=vmin, cmax=vmax), selector=dict(mode="markers+text"))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Error analysis ──
st.markdown("##### Reconstruction Error")

col_map, col_chart = st.columns([1, 1])

with col_map:
    err_text = []
    for i in range(graph.num_nodes):
        t = TYPE_NAMES.get(graph.node_types[i], "?")
        status = "Observed" if p_mask[i] > 0 else "Missing"
        err_text.append(
            f"<b>Node {node_names[i]}</b><br>Type: {t}<br>Status: {status}"
            f"<br>Error: {p_error[i]:.3f} m"
            f"<br>True: {p_true[i]:.2f} m &nbsp;|&nbsp; Pred: {p_pred[i]:.2f} m"
        )
    fig = build_network_figure(
        graph, node_values=p_error, node_text=err_text,
        colorscale=ERROR_COLORSCALE,
        title="Absolute Error Map",
        color_label="Error (m)", height=420, node_size=30,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_chart:
    obs_color = [BLUE if p_mask[i] > 0 else ORANGE for i in range(graph.num_nodes)]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=node_names, y=p_error,
        marker=dict(color=obs_color, line=dict(width=0)),
        hovertemplate="Node %{x}<br>Error: %{y:.3f} m<extra></extra>",
    ))
    # Invisible legend entries
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=BLUE, name="Observed sensor", showlegend=True))
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=ORANGE, name="Missing sensor", showlegend=True))
    fig.update_layout(**plotly_layout(
        title=dict(text="Per-Node Absolute Error"),
        xaxis_title="Node", yaxis_title="Error (m)",
        height=420, showlegend=True,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.08),
    ))
    st.plotly_chart(fig, use_container_width=True)

# ── Summary ──
n_missing = int((p_mask == 0).sum())
avg_obs = p_error[p_mask > 0].mean() if (p_mask > 0).any() else 0
avg_miss = p_error[p_mask == 0].mean() if (p_mask == 0).any() else 0
st.info(
    f"**{n_missing} of {graph.num_nodes} sensors offline** — "
    f"mean error on observed nodes: **{avg_obs:.3f} m** — "
    f"mean error on missing nodes: **{avg_miss:.3f} m**"
)
