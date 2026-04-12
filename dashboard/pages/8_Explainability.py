"""Page 8: GNN Explainability — understanding model decisions."""

import streamlit as st
import plotly.graph_objects as go
import json
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_graph_net1, load_graph_modena, network_selector
from utils.network_viz import build_network_figure
from utils.theme import GLOBAL_CSS, plotly_layout, BLUE, GREEN, ORANGE, RED, PURPLE, CYAN, DIM

st.set_page_config(page_title="Explainability", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

st.title("GNN Explainability")
st.caption("Understanding which nodes, edges, and features drive model decisions (GNNExplainer)")

# ── Network selector ──
network = network_selector(key="explain_net")

# Load data based on network
data_dir = Path(__file__).parent.parent / "data"
if network == "Net1":
    data_path = data_dir / "explainability.json"
else:
    data_path = data_dir / "explainability_modena.json"

if not data_path.exists():
    st.warning(f"Explainability data not available for {network}. Run: `python -m wdn.explainability`")
    st.stop()

with open(data_path) as f:
    data = json.load(f)

graph = load_graph_net1() if network == "Net1" else load_graph_modena()
if graph is None:
    st.warning(f"Graph data not available for {network}.")
    st.stop()

node_names = data["node_names"]
feature_names = data["feature_names"]

# ── Target selector ──
target = st.radio("Explanation target", ["Reconstruction", "Anomaly Detection"],
                  horizontal=True, key="explain_target")
target_key = "reconstruction" if target == "Reconstruction" else "anomaly_detection"
results = data[target_key]

st.divider()

# ── Node Importance ──
st.markdown("##### Node Importance")
st.markdown(f"<span style='opacity:0.5; font-size:0.85rem;'>"
            f"Which nodes contribute most to the model's {target.lower()} decisions "
            f"(averaged over {results['n_explained']} test snapshots)</span>",
            unsafe_allow_html=True)

node_imp = results["node_importance"]

col_map, col_bar = st.columns([1, 1])

with col_map:
    imp_array = np.array(node_imp)

    IMPORTANCE_CS = [[0, "#f0f0f0"], [0.25, "#a8d0f0"], [0.5, "#3787c0"],
                     [0.75, "#1a5599"], [1.0, "#08306b"]]

    hover_text = [
        f"<b>Node {node_names[i]}</b><br>Importance: {node_imp[i]:.3f}"
        for i in range(len(node_names))
    ]

    node_size = 35 if network == "Net1" else 9
    fig = build_network_figure(
        graph, node_values=imp_array, node_text=hover_text,
        colorscale=IMPORTANCE_CS,
        title="Node Importance Map", color_label="Importance",
        height=450, node_size=node_size,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_bar:
    sorted_pairs = sorted(zip(node_names, node_imp), key=lambda x: x[1], reverse=True)

    # For Modena, show top 20 nodes
    if network == "Modena":
        sorted_pairs = sorted_pairs[:20]
        bar_title = "Top 20 Nodes by Importance"
    else:
        bar_title = "Node Importance Ranking"

    sorted_names, sorted_imp = zip(*sorted_pairs)
    bar_colors = [GREEN if v > 0.7 else BLUE if v > 0.3 else "rgba(128,128,128,0.5)" for v in sorted_imp]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(sorted_names), y=list(sorted_imp),
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"{v:.2f}" for v in sorted_imp],
        textposition="outside", textfont=dict(size=11),
    ))
    fig.update_layout(**plotly_layout(
        title=dict(text=bar_title),
        xaxis_title="Node", yaxis_title="Importance (normalized)",
        height=450, yaxis=dict(range=[0, 1.15]),
    ))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Feature Importance ──
st.markdown("##### Feature Importance")
st.markdown("<span style='opacity:0.5; font-size:0.85rem;'>"
            "Which input features are most important for the model's predictions</span>",
            unsafe_allow_html=True)

feat_imp = results["feature_importance"]

col_feat, col_insight = st.columns([3, 2])

with col_feat:
    feat_colors = [GREEN if v > 0.7 else BLUE if v > 0.3 else "rgba(128,128,128,0.5)" for v in feat_imp]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=feature_names, y=feat_imp,
        marker=dict(color=feat_colors, line=dict(width=0)),
        text=[f"{v:.2f}" for v in feat_imp],
        textposition="outside", textfont=dict(size=12),
    ))
    fig.update_layout(**plotly_layout(
        title=dict(text="Input Feature Importance"),
        xaxis_title="Feature", yaxis_title="Importance (normalized)",
        height=400, yaxis=dict(range=[0, 1.25]),
    ))
    st.plotly_chart(fig, use_container_width=True)

with col_insight:
    st.markdown("<br>", unsafe_allow_html=True)

    sorted_feats = sorted(zip(feature_names, feat_imp), key=lambda x: x[1], reverse=True)
    st.markdown("**Most important features:**")
    for name, imp in sorted_feats[:3]:
        st.markdown(f"- **{name}**: {imp:.2f}")

    st.markdown("<br>", unsafe_allow_html=True)

    if target == "Reconstruction":
        st.info(
            "For **reconstruction**, the model relies most heavily on "
            "observed pressure readings and the observation mask to infer "
            "missing sensor values through graph message passing."
        )
    else:
        st.info(
            "For **anomaly detection**, the model uses the residual between "
            "observed and predicted values — high discrepancy signals a "
            "potential attack on that sensor."
        )

st.divider()

# ── Edge Importance ──
st.markdown("##### Edge Importance")

edge_imp = results["edge_importance"]
NE = graph.num_edges
orig_edge_imp = edge_imp[:NE]

sorted_edges = sorted(
    zip(graph.edge_names, orig_edge_imp), key=lambda x: x[1], reverse=True
)

# For Modena, show top 20 edges
if network == "Modena":
    sorted_edges = sorted_edges[:20]
    edge_bar_title = "Top 20 Pipes by Importance"
else:
    edge_bar_title = "Edge (Pipe) Importance Ranking"

col_ebar, col_etable = st.columns([3, 2])

with col_ebar:
    e_names, e_imp = zip(*sorted_edges)
    e_colors = [GREEN if v > 0.7 else BLUE if v > 0.3 else "rgba(128,128,128,0.5)" for v in e_imp]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(e_names), y=list(e_imp),
        marker=dict(color=e_colors, line=dict(width=0)),
        text=[f"{v:.2f}" for v in e_imp],
        textposition="outside", textfont=dict(size=11),
    ))
    fig.update_layout(**plotly_layout(
        title=dict(text=edge_bar_title),
        xaxis_title="Pipe", yaxis_title="Importance (normalized)",
        height=400, yaxis=dict(range=[0, 1.15]),
    ))
    st.plotly_chart(fig, use_container_width=True)

with col_etable:
    rows = []
    for name, imp in sorted_edges:
        idx = graph.edge_names.index(name)
        src = graph.node_names[graph.edge_index[0, idx]]
        dst = graph.node_names[graph.edge_index[1, idx]]
        rows.append({
            "Pipe": name,
            "From": src,
            "To": dst,
            "Importance": f"{imp:.3f}",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True, height=350)
