"""Page 2: State Reconstruction Demo."""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import network_selector, load_graph, load_demo_snapshots, load_test_results
from utils.network_viz import (
    build_network_figure, PRESSURE_COLORSCALE, ERROR_COLORSCALE, TYPE_NAMES,
)
from utils.theme import GLOBAL_CSS, plotly_layout, BLUE, ORANGE, GREEN, DIM, TEXT_DIM

st.set_page_config(page_title="Reconstruction", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

st.title("State Reconstruction")
network = network_selector(key="recon_net")
st.caption("Recovering full network pressures from 50% missing, noisy sensor data")

graph = load_graph(network)
demo = load_demo_snapshots(network)
test_results = load_test_results(network)

if demo is None:
    st.error("Demo data not found. Run the appropriate precompute script.")
    st.stop()

snapshots = demo["snapshots"]
node_names = demo["node_names"]
N = graph.num_nodes

# ── Metrics ──
c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE (all nodes)", f"{test_results['reconstruction']['pressure_all']['mae']:.3f} m")
c2.metric("MAE (unobserved)", f"{test_results['reconstruction']['pressure_unobs']['mae']:.3f} m")
c3.metric("RMSE (unobserved)", f"{test_results['reconstruction']['pressure_unobs']['rmse']:.3f} m")
c4.metric("Missing Rate", "50%")

st.divider()

snap_idx = st.selectbox(
    "Select a test snapshot",
    range(len(snapshots)),
    format_func=lambda i: f"Snapshot {i+1}  (test index {snapshots[i]['index']})",
)

snap = snapshots[snap_idx]
p_true = np.array(snap["pressure_true"])
p_pred = np.array(snap["pressure_pred"])
p_mask = np.array(snap["pressure_mask"])
p_error = np.array(snap["pressure_error"])
p_obs = np.array(snap.get("pressure_obs", p_true * p_mask))

vmin = min(p_true.min(), p_pred.min())
vmax = max(p_true.max(), p_pred.max())

# ── Helper for large networks ──
def _build_scatter_map(values, title, colorscale, color_label, cmin=None, cmax=None):
    """Build a Plotly scatter map for large networks."""
    coords = graph.node_coordinates
    edge_index = graph.edge_index
    edge_x, edge_y = [], []
    for j in range(edge_index.shape[1]):
        src, dst = edge_index[0, j], edge_index[1, j]
        edge_x += [coords[src, 0], coords[dst, 0], None]
        edge_y += [coords[src, 1], coords[dst, 1], None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.8, color="rgba(128,128,128,0.15)"),
        hoverinfo="none", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=coords[:, 0].tolist(), y=coords[:, 1].tolist(),
        mode="markers",
        marker=dict(
            size=5, color=values.tolist(), colorscale=colorscale,
            cmin=cmin, cmax=cmax, showscale=True,
            colorbar=dict(
                title=dict(text=color_label, font=dict(size=10, color=DIM)),
                thickness=12, len=0.6, tickfont=dict(size=9, color=DIM),
                outlinewidth=0, bgcolor="rgba(0,0,0,0)",
            ), line=dict(width=0),
        ),
        hovertext=[f"Node {node_names[i]}<br>{color_label}: {values[i]:.3f}" for i in range(N)],
        hoverinfo="text", showlegend=False,
    ))
    fig.update_layout(**plotly_layout(
        title=dict(text=title, font=dict(size=13)),
        height=400, showlegend=False, hovermode="closest",
        margin=dict(l=5, r=5, t=40, b=5),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False,
                   scaleanchor="x", scaleratio=1),
    ))
    return fig


# ── Three-panel comparison ──
col1, col2, col3 = st.columns(3)

pressure_cs = PRESSURE_COLORSCALE
error_cs = ERROR_COLORSCALE

if network == "Net1":
    with col1:
        fig = build_network_figure(graph, node_values=p_true, colorscale=pressure_cs,
                                   title="Ground Truth", color_label="Pressure (m)", height=420, node_size=30)
        fig.update_traces(marker=dict(cmin=vmin, cmax=vmax), selector=dict(mode="markers"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        obs_text = []
        for i in range(N):
            t = TYPE_NAMES.get(graph.node_types[i], "?")
            if p_mask[i] > 0:
                obs_text.append(f"<b>Node {node_names[i]}</b><br>Type: {t}<br>Reading: {p_obs[i]:.2f} m")
            else:
                obs_text.append(f"<b>Node {node_names[i]}</b><br>Type: {t}<br><i>Sensor offline</i>")
        fig = build_network_figure(graph, title="Observed (50% missing)", height=420, node_size=30)
        marker_colors = [p_obs[i] if p_mask[i] > 0 else vmin for i in range(N)]
        marker_opacities = [1.0 if p_mask[i] > 0 else 0.15 for i in range(N)]
        fig.update_traces(
            marker=dict(color=marker_colors, colorscale=pressure_cs, cmin=vmin, cmax=vmax,
                        opacity=marker_opacities, showscale=True,
                        colorbar=dict(title=dict(text="Pressure (m)", font=dict(size=11, color=DIM)),
                                      thickness=14, len=0.55, outlinewidth=0, bgcolor="rgba(0,0,0,0)",
                                      tickfont=dict(size=10, color=DIM))),
            hovertext=obs_text, selector=dict(mode="markers"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        fig = build_network_figure(graph, node_values=p_pred, colorscale=pressure_cs,
                                   title="GNN Prediction", color_label="Pressure (m)", height=420, node_size=30)
        fig.update_traces(marker=dict(cmin=vmin, cmax=vmax), selector=dict(mode="markers"))
        st.plotly_chart(fig, use_container_width=True)
else:
    # Large network: use scatter maps
    light_pressure_cs = [[0, "#a8d0f0"], [0.25, "#6baed6"], [0.5, "#3787c0"],
                         [0.75, "#2166ac"], [1.0, "#08306b"]]
    light_error_cs = [[0, "#f0f0f0"], [0.3, "#f4a090"], [0.6, "#e05040"],
                      [0.8, "#c0392b"], [1.0, "#8b1a1a"]]
    with col1:
        st.plotly_chart(_build_scatter_map(p_true, "Ground Truth", light_pressure_cs,
                                           "Pressure (m)", vmin, vmax), use_container_width=True)
    with col2:
        st.plotly_chart(_build_scatter_map(p_pred, "GNN Prediction", light_pressure_cs,
                                           "Pressure (m)", vmin, vmax), use_container_width=True)
    with col3:
        st.plotly_chart(_build_scatter_map(p_error, "Absolute Error", light_error_cs,
                                           "Error (m)", 0, max(p_error.max(), 0.5)), use_container_width=True)

st.divider()

# ── Error analysis ──
st.markdown("##### Reconstruction Error")

if network == "Net1":
    col_map, col_chart = st.columns([1, 1])
    with col_map:
        err_text = []
        for i in range(N):
            t = TYPE_NAMES.get(graph.node_types[i], "?")
            status = "Observed" if p_mask[i] > 0 else "Missing"
            err_text.append(
                f"<b>Node {node_names[i]}</b><br>Type: {t}<br>Status: {status}"
                f"<br>Error: {p_error[i]:.3f} m<br>True: {p_true[i]:.2f} m | Pred: {p_pred[i]:.2f} m"
            )
        fig = build_network_figure(graph, node_values=p_error, node_text=err_text,
                                   colorscale=ERROR_COLORSCALE, title="Absolute Error Map",
                                   color_label="Error (m)", height=420, node_size=30)
        st.plotly_chart(fig, use_container_width=True)

    with col_chart:
        obs_color = [BLUE if p_mask[i] > 0 else ORANGE for i in range(N)]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=node_names, y=p_error.tolist(),
                             marker=dict(color=obs_color, line=dict(width=0)),
                             hovertemplate="Node %{x}<br>Error: %{y:.3f} m<extra></extra>"))
        fig.add_trace(go.Bar(x=[None], y=[None], marker_color=BLUE, name="Observed", showlegend=True))
        fig.add_trace(go.Bar(x=[None], y=[None], marker_color=ORANGE, name="Missing", showlegend=True))
        fig.update_layout(**plotly_layout(
            title=dict(text="Per-Node Absolute Error"), xaxis_title="Node", yaxis_title="Error (m)",
            height=420, showlegend=True, legend=dict(orientation="h", x=0.5, xanchor="center", y=1.08),
        ))
        st.plotly_chart(fig, use_container_width=True)
else:
    col_hist, col_box = st.columns(2)
    with col_hist:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=p_error.tolist(), nbinsx=40, marker_color=BLUE, opacity=0.85))
        fig.update_layout(**plotly_layout(
            title=dict(text="Error Distribution"), xaxis_title="Absolute Error (m)",
            yaxis_title="Number of Nodes", height=350,
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col_box:
        all_maes = [np.mean(s["pressure_error"]) for s in snapshots]
        fig = go.Figure()
        fig.add_trace(go.Box(y=all_maes, name="Snapshot MAE", marker_color=GREEN, boxmean=True))
        fig.update_layout(**plotly_layout(
            title=dict(text="MAE Across Test Snapshots"), yaxis_title="MAE (m)",
            height=350, showlegend=False,
        ))
        st.plotly_chart(fig, use_container_width=True)

# ── Summary ──
n_missing = int((p_mask == 0).sum())
avg_obs = float(p_error[p_mask > 0].mean()) if (p_mask > 0).any() else 0
avg_miss = float(p_error[p_mask == 0].mean()) if (p_mask == 0).any() else 0
st.info(
    f"**{n_missing} of {N} sensors offline** — "
    f"mean error on observed nodes: **{avg_obs:.3f} m** — "
    f"mean error on missing nodes: **{avg_miss:.3f} m**"
)
