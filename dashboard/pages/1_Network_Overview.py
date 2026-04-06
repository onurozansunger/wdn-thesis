"""Page 1: Network Overview."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import network_selector, load_graph
from utils.network_viz import build_network_figure, TYPE_NAMES, TYPE_COLORS
from utils.theme import (
    GLOBAL_CSS, plotly_layout,
    NODE_JUNCTION, NODE_RESERVOIR, NODE_TANK,
    TEXT_DIM,
)

st.set_page_config(page_title="Network Overview", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

st.title("Network Overview")
network = network_selector(key="overview_net")

graph = load_graph(network)

desc = {
    "Net1": "EPANET Net1 — a benchmark water distribution network with 11 nodes and 13 links",
    "Modena": "Modena — a real-world benchmark with 272 nodes and 317 pipes (Bragalli et al., 2008)",
}
st.caption(desc.get(network, ""))

# ── Stats ──
n_junctions = int((graph.node_types == 0).sum())
n_reservoirs = int((graph.node_types == 1).sum())
n_tanks = int((graph.node_types == 2).sum())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Nodes", graph.num_nodes)
c2.metric("Edges", graph.num_edges)
c3.metric("Junctions", n_junctions)
c4.metric("Reservoirs", n_reservoirs)
c5.metric("Tanks", n_tanks)

st.divider()

# ── Network graph ──
TYPE_SYMBOLS = {0: "circle", 1: "diamond", 2: "square"}
TYPE_COLORS_MAP = {0: NODE_JUNCTION, 1: NODE_RESERVOIR, 2: NODE_TANK}

if network == "Net1":
    # Use build_network_figure for Net1 (small network, shows labels)
    fig = build_network_figure(graph, title="", node_size=38, height=580, edge_width=3.5)
    for ntype, color in TYPE_COLORS.items():
        fig.add_trace(dict(
            type="scatter", x=[None], y=[None], mode="markers",
            marker=dict(size=11, color=color, line=dict(width=1.5, color="rgba(128,128,128,0.3)")),
            name=TYPE_NAMES[ntype], showlegend=True,
        ))
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.02,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
    )
else:
    # Modena: scatter plot (too many nodes for labels)
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
        line=dict(width=1.5, color="rgba(128,128,128,0.30)"),
        hoverinfo="none", showlegend=False,
    ))
    for ntype in [0, 1, 2]:
        mask = graph.node_types == ntype
        if not mask.any():
            continue
        idxs = np.where(mask)[0]
        fig.add_trace(go.Scatter(
            x=coords[idxs, 0].tolist(), y=coords[idxs, 1].tolist(),
            mode="markers",
            marker=dict(size=9 if ntype == 0 else 16,
                        color=TYPE_COLORS_MAP[ntype],
                        symbol=TYPE_SYMBOLS[ntype],
                        line=dict(width=1, color="rgba(128,128,128,0.4)")),
            name=TYPE_NAMES[ntype],
            hovertext=[f"Node {graph.node_names[i]}<br>Type: {TYPE_NAMES[ntype]}" for i in idxs],
            hoverinfo="text",
        ))
    fig.update_layout(**plotly_layout(
        height=580, showlegend=True, hovermode="closest",
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False,
                   scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.02),
    ))

st.plotly_chart(fig, use_container_width=True)

# ── Tables ──
st.divider()
tab_nodes, tab_edges = st.tabs(["Node Properties", "Edge Properties"])

with tab_nodes:
    rows = []
    for i in range(graph.num_nodes):
        rows.append({
            "Name": graph.node_names[i],
            "Type": TYPE_NAMES.get(graph.node_types[i], "?"),
            "Elevation (m)": round(float(graph.node_elevations[i]), 1),
            "Base Demand (m\u00b3/s)": round(float(graph.node_base_demands[i]), 6),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                 height=min(400, 35 * graph.num_nodes + 40))

with tab_edges:
    etypes = {0: "Pipe", 1: "Pump", 2: "Valve"}
    rows = []
    for j in range(graph.num_edges):
        sn = graph.node_names[graph.edge_index[0, j]]
        dn = graph.node_names[graph.edge_index[1, j]]
        rows.append({
            "Name": graph.edge_names[j],
            "Type": etypes.get(int(graph.edge_types[j]), "?"),
            "From": sn, "To": dn,
            "Length (m)": round(float(graph.edge_lengths[j]), 0),
            "Diameter (m)": round(float(graph.edge_diameters[j]), 4),
            "Roughness": round(float(graph.edge_roughness[j]), 1),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                 height=min(400, 35 * graph.num_edges + 40))
