"""Page 1: Network Overview."""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_graph
from utils.network_viz import build_network_figure, TYPE_NAMES, TYPE_COLORS
from utils.theme import GLOBAL_CSS, plotly_layout

st.set_page_config(page_title="Network Overview", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

st.title("Network Overview")
st.caption("EPANET Net1 — a benchmark water distribution network with 11 nodes and 13 links")

graph = load_graph()

# ── Stats ──
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Nodes", graph.num_nodes)
c2.metric("Edges", graph.num_edges)
c3.metric("Junctions", int((graph.node_types == 0).sum()))
c4.metric("Reservoirs", int((graph.node_types == 1).sum()))
c5.metric("Tanks", int((graph.node_types == 2).sum()))

st.divider()

# ── Network graph ──
fig = build_network_figure(graph, title="", node_size=38, height=580, edge_width=3.5)

# Legend entries
for ntype, color in TYPE_COLORS.items():
    fig.add_trace(dict(
        type="scatter", x=[None], y=[None], mode="markers",
        marker=dict(size=11, color=color, line=dict(width=1.5, color="rgba(255,255,255,0.2)")),
        name=TYPE_NAMES[ntype], showlegend=True,
    ))
fig.update_layout(
    showlegend=True,
    legend=dict(
        orientation="h", x=0.5, xanchor="center", y=-0.02,
        bgcolor="rgba(0,0,0,0)", font=dict(size=12),
    ),
)
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
            "Elevation (m)": round(graph.node_elevations[i], 1),
            "Base Demand (m\u00b3/s)": round(graph.node_base_demands[i], 6),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab_edges:
    etypes = {0: "Pipe", 1: "Pump", 2: "Valve"}
    rows = []
    for j in range(graph.num_edges):
        sn = graph.node_names[graph.edge_index[0, j]]
        dn = graph.node_names[graph.edge_index[1, j]]
        rows.append({
            "Name": graph.edge_names[j],
            "Type": etypes.get(graph.edge_types[j], "?"),
            "From": sn, "To": dn,
            "Length (m)": round(graph.edge_lengths[j], 0),
            "Diameter (m)": round(graph.edge_diameters[j], 4),
            "Roughness": round(graph.edge_roughness[j], 1),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
