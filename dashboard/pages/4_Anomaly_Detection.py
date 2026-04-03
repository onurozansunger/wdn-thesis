"""Page 4: Anomaly Detection."""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import network_selector, load_graph, load_demo_snapshots, load_test_results
from utils.network_viz import build_network_figure, TYPE_NAMES
from utils.theme import GLOBAL_CSS, plotly_layout, GREEN, ORANGE, RED, BLUE, GRAY, DIM

st.set_page_config(page_title="Anomaly Detection", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

st.title("Anomaly Detection")
network = network_selector(key="anom_net")
st.caption("Identifying which sensors have been compromised by cyber-attacks")

graph = load_graph(network)
demo = load_demo_snapshots(network)
test_results = load_test_results(network)

if demo is None:
    st.error("Demo data not found.")
    st.stop()

anom = test_results.get("anomaly_detection", {})
p_anom = anom.get("pressure", {})

# ── Metrics ──
c1, c2, c3, c4 = st.columns(4)
c1.metric("Precision", f"{p_anom.get('precision', 0):.1%}")
c2.metric("Recall", f"{p_anom.get('recall', 0):.1%}")
c3.metric("F1 Score", f"{p_anom.get('f1', 0):.3f}")
c4.metric("AUROC", f"{p_anom.get('auroc', 0):.3f}")

st.divider()

# ── Controls ──
snapshots = demo["snapshots"]
node_names = demo["node_names"]
N = graph.num_nodes
attacked_indices = [i for i, s in enumerate(snapshots) if sum(s["pressure_anomaly_true"]) > 0]

col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 1, 1])
with col_ctrl1:
    show_attacked = st.checkbox("Only show snapshots with attacks", value=True)
    display_indices = attacked_indices if show_attacked and attacked_indices else list(range(len(snapshots)))
    snap_idx = st.selectbox(
        "Select snapshot", display_indices,
        format_func=lambda i: f"Snapshot {i+1}  ({'attacks' if i in attacked_indices else 'clean'})",
    )
with col_ctrl2:
    threshold = st.slider("Detection threshold", 0.0, 1.0, 0.5, 0.01)
with col_ctrl3:
    st.markdown("""<div style="font-size:0.8rem; opacity:0.55; margin-top:1.5rem;">
    Sensors above the threshold are flagged as compromised.
    Lower = more sensitive, more false alarms.</div>""", unsafe_allow_html=True)

st.divider()

snap = snapshots[snap_idx]
p_anom_true = np.array(snap["pressure_anomaly_true"])
p_anom_prob = np.array(snap["pressure_anomaly_prob"])
p_mask = np.array(snap["pressure_mask"])
p_anom_pred = (p_anom_prob > threshold).astype(float)

STATUS = {
    "TP": {"color": GREEN,  "label": "Correctly detected attack"},
    "FP": {"color": ORANGE, "label": "False alarm"},
    "FN": {"color": RED,    "label": "Missed attack"},
    "TN": {"color": BLUE,   "label": "Correctly identified as clean"},
    "NA": {"color": GRAY,   "label": "Sensor offline"},
}

node_colors, hover_text, statuses = [], [], []
tp = fp = fn = tn = 0

for i in range(N):
    t = TYPE_NAMES.get(graph.node_types[i], "?")
    if p_mask[i] == 0:
        s = "NA"
    elif p_anom_true[i] == 1 and p_anom_pred[i] == 1:
        s = "TP"; tp += 1
    elif p_anom_true[i] == 0 and p_anom_pred[i] == 1:
        s = "FP"; fp += 1
    elif p_anom_true[i] == 1 and p_anom_pred[i] == 0:
        s = "FN"; fn += 1
    else:
        s = "TN"; tn += 1
    statuses.append(s)
    node_colors.append(STATUS[s]["color"])
    hover_text.append(
        f"<b>Node {node_names[i]}</b><br>Type: {t}<br>{STATUS[s]['label']}"
        f"<br>Anomaly prob: {p_anom_prob[i]:.3f}"
        f"<br>Truth: {'Attacked' if p_anom_true[i] > 0 else 'Clean'}"
    )

# ── Network + Confusion Matrix ──
col_net, col_side = st.columns([5, 3])

with col_net:
    if network == "Net1":
        fig = build_network_figure(graph, discrete_colors=node_colors, node_text=hover_text,
                                   title="", height=520, node_size=40, edge_width=3)
        for key, info in STATUS.items():
            fig.add_trace(dict(type="scatter", x=[None], y=[None], mode="markers",
                               marker=dict(size=10, color=info["color"],
                                           line=dict(width=1, color="rgba(128,128,128,0.3)")),
                               name=info["label"], showlegend=True))
        fig.update_layout(showlegend=True,
                          legend=dict(orientation="v", x=1.02, y=0.5, yanchor="middle",
                                      bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
                          margin=dict(l=10, r=160, t=30, b=10))
    else:
        # Large network scatter
        coords = graph.node_coordinates
        edge_index = graph.edge_index
        edge_x, edge_y = [], []
        for j in range(edge_index.shape[1]):
            src, dst = edge_index[0, j], edge_index[1, j]
            edge_x += [coords[src, 0], coords[dst, 0], None]
            edge_y += [coords[src, 1], coords[dst, 1], None]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                 line=dict(width=0.8, color="rgba(128,128,128,0.15)"),
                                 hoverinfo="none", showlegend=False))
        for key, info in STATUS.items():
            idxs = [i for i, s in enumerate(statuses) if s == key]
            if not idxs:
                continue
            fig.add_trace(go.Scatter(
                x=[coords[i, 0] for i in idxs], y=[coords[i, 1] for i in idxs],
                mode="markers",
                marker=dict(size=5 if key in ("TN", "NA") else 8, color=info["color"],
                            line=dict(width=0)),
                name=info["label"],
                hovertext=[hover_text[i] for i in idxs], hoverinfo="text",
            ))
        fig.update_layout(**plotly_layout(
            height=520, hovermode="closest",
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False,
                       scaleanchor="x", scaleratio=1),
            legend=dict(orientation="v", x=1.02, y=0.5, yanchor="middle",
                        bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        ))

    st.plotly_chart(fig, use_container_width=True)

with col_side:
    st.markdown("##### Confusion Matrix")
    cm = np.array([[tn, fp], [fn, tp]])
    cm_text = [[str(v) for v in row] for row in cm]
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=["Pred Clean", "Pred Attack"], y=["Actually Clean", "Actually Attacked"],
        text=cm_text, texttemplate="<b>%{text}</b>", textfont=dict(size=20),
        colorscale=[[0, "rgba(77,166,255,0.08)"], [1, "rgba(77,166,255,0.35)"]],
        showscale=False, hoverinfo="skip",
    ))
    fig.update_layout(**plotly_layout(
        height=240, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(side="bottom", tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11), autorange="reversed"),
    ))
    st.plotly_chart(fig, use_container_width=True)

    n_attacks = int(p_anom_true[p_mask > 0].sum())
    n_observed = int(p_mask.sum())
    st.markdown(f"""
    | | |
    |---|---|
    | Observed sensors | **{n_observed}** / {N} |
    | Attacks | **{n_attacks}** |
    | Detected | **{tp}** |
    | Missed | **{fn}** |
    | False alarms | **{fp}** |
    """)

st.divider()

# ── Per-node probability ──
st.markdown("##### Anomaly Probability per Node")

if network == "Net1" or N <= 50:
    bar_colors = [RED if p_anom_true[i] > 0 else "rgba(77,166,255,0.6)" for i in range(N)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=node_names, y=p_anom_prob.tolist(),
                         marker=dict(color=bar_colors, line=dict(width=0)),
                         hovertemplate="Node %{x}<br>Prob: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=threshold, line_dash="dot", line_color=ORANGE, line_width=2,
                  annotation_text=f"threshold = {threshold:.2f}",
                  annotation_position="top right", annotation_font=dict(size=11, color=ORANGE))
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=RED, name="Attacked", showlegend=True))
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color="rgba(77,166,255,0.6)", name="Clean", showlegend=True))
    fig.update_layout(**plotly_layout(
        xaxis_title="Node", yaxis_title="Anomaly Probability",
        yaxis=dict(range=[0, 1.05], gridcolor="rgba(128,128,128,0.12)"),
        height=340, showlegend=True, legend=dict(orientation="h", x=0.5, xanchor="center", y=1.1),
    ))
else:
    # Histogram for large networks
    attacked_probs = p_anom_prob[p_anom_true > 0].tolist()
    clean_probs = p_anom_prob[(p_anom_true == 0) & (p_mask > 0)].tolist()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=clean_probs, nbinsx=30, name="Clean sensors",
                               marker_color="rgba(77,166,255,0.6)", opacity=0.7))
    fig.add_trace(go.Histogram(x=attacked_probs, nbinsx=30, name="Attacked sensors",
                               marker_color=RED, opacity=0.7))
    fig.add_vline(x=threshold, line_dash="dot", line_color=ORANGE, line_width=2,
                  annotation_text=f"threshold", annotation_position="top right",
                  annotation_font=dict(size=11, color=ORANGE))
    fig.update_layout(**plotly_layout(
        title=dict(text="Anomaly Probability Distribution"),
        xaxis_title="Anomaly Probability", yaxis_title="Count",
        height=340, barmode="overlay",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.08),
    ))

st.plotly_chart(fig, use_container_width=True)
