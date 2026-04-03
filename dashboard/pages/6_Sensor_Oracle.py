"""Page 6: Sensor Oracle — Optimal placement via uncertainty (Net1 only)."""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_graph, load_sensor_oracle
from utils.network_viz import build_network_figure, UNCERTAINTY_COLORSCALE, TYPE_NAMES
from utils.theme import GLOBAL_CSS, plotly_layout, GREEN, ORANGE, RED, BLUE, DIM, CYAN

st.set_page_config(page_title="Sensor Oracle", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

st.title("Sensor Oracle")
st.caption("Determining optimal sensor locations using MC Dropout prediction uncertainty (Net1)")

graph = load_graph("Net1")
oracle = load_sensor_oracle()
ranking_data = oracle.get("node_ranking", {})
greedy_data = oracle.get("greedy_placement", {})

uncertainties = ranking_data.get("mean_uncertainties", [0] * graph.num_nodes)
ranking = ranking_data.get("ranking", [])

# ── Uncertainty map ──
st.markdown("##### Prediction Uncertainty Across the Network")
st.markdown(
    "<span style='opacity:0.5; font-size:0.85rem;'>"
    "Brighter nodes have higher uncertainty — the model is least confident about their "
    "state when sensors are missing.</span>", unsafe_allow_html=True,
)

unc_text = []
for i in range(graph.num_nodes):
    t = TYPE_NAMES.get(graph.node_types[i], "?")
    r = next((r for r in ranking if r["node_index"] == i), None)
    rank_str = f"<br>Placement priority: #{r['rank']}" if r else ""
    obs_rate = f"{r['observation_rate']:.0%}" if r else "?"
    unc_text.append(
        f"<b>Node {graph.node_names[i]}</b><br>Type: {t}"
        f"<br>Mean uncertainty: {uncertainties[i]:.5f}"
        f"<br>Observation rate: {obs_rate}{rank_str}"
    )

fig = build_network_figure(
    graph, node_values=uncertainties, node_text=unc_text,
    colorscale=UNCERTAINTY_COLORSCALE,
    title="", color_label="Uncertainty (\u03c3)",
    height=500, node_size=38,
)
st.plotly_chart(fig, use_container_width=True)

# ── Ranking ──
st.divider()
col_table, col_rec = st.columns([3, 2])

with col_table:
    st.markdown("##### Placement Priority Ranking")
    df = pd.DataFrame(ranking)
    df = df[["rank", "node_name", "node_type", "mean_uncertainty", "max_uncertainty",
             "mean_error", "observation_rate"]]
    df.columns = ["Priority", "Node", "Type", "Mean Unc.", "Max Unc.", "Mean Error", "Obs Rate"]
    df["Obs Rate"] = df["Obs Rate"].apply(lambda x: f"{x:.0%}")
    df["Mean Unc."] = df["Mean Unc."].apply(lambda x: f"{x:.5f}")
    df["Max Unc."] = df["Max Unc."].apply(lambda x: f"{x:.5f}")
    df["Mean Error"] = df["Mean Error"].apply(lambda x: f"{x:.4f}")
    st.dataframe(df, use_container_width=True, hide_index=True, height=350)

with col_rec:
    st.markdown("##### Top Recommendations")
    for r in ranking[:3]:
        st.markdown(
            f"**#{r['rank']} &mdash; Node {r['node_name']}** ({r['node_type']})<br>"
            f"<span style='opacity:0.5; font-size:0.85rem;'>"
            f"Mean uncertainty: {r['mean_uncertainty']:.5f} &nbsp;|&nbsp; "
            f"Observation rate: {r['observation_rate']:.0%}</span>",
            unsafe_allow_html=True,
        )
        st.markdown("")
    st.markdown(
        "<span style='opacity:0.5; font-size:0.85rem;'>"
        "These nodes have the highest prediction uncertainty because their state is "
        "difficult to infer from neighboring sensors alone. Installing sensors here "
        "would maximally reduce the network's overall reconstruction error.</span>",
        unsafe_allow_html=True,
    )

# ── Greedy placement ──
st.divider()
st.markdown("##### Greedy Placement Simulation")
st.markdown(
    "<span style='opacity:0.5; font-size:0.85rem;'>"
    "Drag the slider to add sensors one at a time at the optimal locations.</span>",
    unsafe_allow_html=True,
)

if greedy_data:
    details = greedy_data.get("placement_details", [])
    unc_curve = greedy_data.get("uncertainty_curve", [])
    err_curve = greedy_data.get("error_curve", [])

    n_placed = st.slider("Number of new sensors", 0, len(details), 0, key="placement_slider")

    placed_set = set(greedy_data["placement_order"][:n_placed])
    max_unc = max(uncertainties) if max(uncertainties) > 0 else 1

    node_viz_colors, hover_placed = [], []
    for i in range(graph.num_nodes):
        t = TYPE_NAMES.get(graph.node_types[i], "?")
        if i in placed_set:
            step = greedy_data["placement_order"].index(i) + 1
            node_viz_colors.append(GREEN)
            hover_placed.append(f"<b>Node {graph.node_names[i]}</b><br>Type: {t}"
                                f"<br>New sensor (step {step})")
        else:
            norm = uncertainties[i] / max_unc
            r = int(50 + 205 * norm)
            g = int(50 + 80 * (1 - norm))
            b = int(60 + 40 * (1 - norm))
            node_viz_colors.append(f"rgb({r},{g},{b})")
            hover_placed.append(f"<b>Node {graph.node_names[i]}</b><br>Type: {t}"
                                f"<br>Uncertainty: {uncertainties[i]:.5f}")

    col_net, col_curve = st.columns([1, 1])

    with col_net:
        fig = build_network_figure(
            graph, discrete_colors=node_viz_colors, node_text=hover_placed,
            title=f"{n_placed} sensor{'s' if n_placed != 1 else ''} placed",
            height=460, node_size=36,
        )
        fig.add_trace(dict(type="scatter", x=[None], y=[None], mode="markers",
                           marker=dict(size=10, color=GREEN, line=dict(width=1, color="rgba(128,128,128,0.3)")),
                           name="New sensor", showlegend=True))
        fig.add_trace(dict(type="scatter", x=[None], y=[None], mode="markers",
                           marker=dict(size=10, color="rgb(200,80,70)", line=dict(width=1, color="rgba(128,128,128,0.3)")),
                           name="High uncertainty", showlegend=True))
        fig.update_layout(showlegend=True,
                          legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.02, bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig, use_container_width=True)

    with col_curve:
        x_vals = list(range(len(err_curve)))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_vals, y=err_curve, mode="lines+markers", name="Reconstruction Error",
            line=dict(color=BLUE, width=3),
            marker=dict(size=7, color=BLUE, line=dict(width=1, color="rgba(128,128,128,0.3)")),
            fill="tozeroy", fillcolor="rgba(77,166,255,0.08)",
        ))
        if n_placed < len(err_curve):
            fig.add_trace(go.Scatter(
                x=[n_placed], y=[err_curve[n_placed]], mode="markers",
                marker=dict(size=14, color=ORANGE, symbol="diamond",
                            line=dict(width=2, color="rgba(128,128,128,0.5)")),
                name="Current", showlegend=True,
            ))
        fig.update_layout(**plotly_layout(
            title=dict(text="Error vs Sensors Placed"),
            xaxis_title="Sensors Placed", yaxis_title="Mean Error",
            height=460, legend=dict(orientation="h", x=0.5, xanchor="center", y=1.08),
        ))
        st.plotly_chart(fig, use_container_width=True)

    if n_placed > 0 and n_placed <= len(unc_curve) - 1:
        unc_red = ((unc_curve[0] - unc_curve[n_placed]) / unc_curve[0]) * 100
        err_red = ((err_curve[0] - err_curve[n_placed]) / err_curve[0]) * 100
        st.info(f"With **{n_placed} sensor{'s' if n_placed != 1 else ''}**: "
                f"uncertainty reduced by **{unc_red:.1f}%**, error reduced by **{err_red:.1f}%**.")
