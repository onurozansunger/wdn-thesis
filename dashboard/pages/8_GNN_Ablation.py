"""GNN backbone ablation — defending the GraphSAGE choice empirically.

Four backbones (GraphSAGE, GAT, GCN, Transformer) trained as a spatial
MultiTaskGNN on each of the three benchmark networks. Same data, same
training schedule, only the message-passing layer changes.
"""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

DASHBOARD = Path(__file__).parent.parent
sys.path.insert(0, str(DASHBOARD))

from utils.theme import GLOBAL_CSS, BLUE, PURPLE, GREEN, ORANGE, plotly_layout
from utils.data_loader import load_gnn_ablation


st.set_page_config(page_title="GNN Ablation", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

st.title("GNN Backbone Ablation")
st.caption(
    "Spatial MultiTaskGNN with four different message-passing layers on "
    "three networks. GraphSAGE wins on every network — the mean "
    "aggregator suits smooth pressure / flow signals where attention "
    "adds capacity without a matching signal gain."
)

data = load_gnn_ablation()
networks = ["Net1", "Net3", "Modena"]
backbones = ["GraphSAGE", "GAT", "GCN", "Transformer"]
COLORS = {
    "GraphSAGE": GREEN, "GAT": BLUE, "GCN": PURPLE, "Transformer": ORANGE,
}


def _hex_to_rgba(h: str, alpha: float = 0.12) -> str:
    """Plotly's ``fillcolor`` rejects the 8-char hex form, so we
    convert to an explicit rgba() string."""
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _f1(d):
    return d.get("anomaly_detection", {}).get("pressure", {}).get("f1")


def _auroc(d):
    return d.get("anomaly_detection", {}).get("pressure", {}).get("auroc")


def _mae(d):
    return d.get("reconstruction", {}).get("pressure_unobs", {}).get("mae")


# ── Headline table ────────────────────────────────────────────────────
st.markdown("### Headline — anomaly F1 across networks")

import pandas as pd

table_rows = []
for gnn in backbones:
    row = {"Backbone": gnn}
    for net in networks:
        d = data.get(net, {}).get(gnn)
        row[f"{net} F1"] = f"{_f1(d):.3f}" if d else "—"
    table_rows.append(row)

df = pd.DataFrame(table_rows)
st.dataframe(df, width="stretch", hide_index=True)

# Highlight the winner per column
winner_text = "  ·  ".join([
    f"**{net}**: " + max(
        ((g, _f1(data.get(net, {}).get(g))) for g in backbones if data.get(net, {}).get(g)),
        key=lambda x: x[1] or 0,
    )[0]
    for net in networks
])
st.success(f"Per-network winner — {winner_text}")

st.divider()

# ── Bar chart per network ─────────────────────────────────────────────
st.markdown("### Anomaly F1 per backbone")

cols = st.columns(3)
for col, net in zip(cols, networks):
    f1s = [_f1(data.get(net, {}).get(g)) or 0 for g in backbones]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=backbones, y=f1s,
        marker_color=[COLORS[g] for g in backbones],
        marker_opacity=0.85,
        text=[f"{v:.3f}" for v in f1s], textposition="outside",
    ))
    fig.update_layout(**plotly_layout(
        title=dict(text=net),
        yaxis=dict(range=[0, 0.85], title="F1"),
        height=320, showlegend=False,
    ))
    col.plotly_chart(fig, width="stretch")

st.divider()

# ── Radar chart ───────────────────────────────────────────────────────
st.markdown("### Multi-metric view (F1, AUROC, 1 − normalised MAE)")
st.caption(
    "Radar chart pulls in three metrics at once. Reconstruction is "
    "shown as 1 minus min-max-normalised MAE so 'further from the "
    "centre' means 'better' on every axis."
)

# Compute metrics, then normalise across all (network, backbone) pairs.
all_mae = [_mae(data.get(n, {}).get(g))
           for n in networks for g in backbones
           if data.get(n, {}).get(g) is not None]
mae_min, mae_max = (min(all_mae), max(all_mae)) if all_mae else (0, 1)

cols = st.columns(3)
for col, net in zip(cols, networks):
    fig = go.Figure()
    for g in backbones:
        d = data.get(net, {}).get(g)
        if not d:
            continue
        f1 = _f1(d) or 0
        au = _auroc(d) or 0
        mae = _mae(d) or 0
        recon_score = 1 - (mae - mae_min) / max(mae_max - mae_min, 1e-9)
        fig.add_trace(go.Scatterpolar(
            r=[f1, au, recon_score, f1],
            theta=["F1", "AUROC", "Recon", "F1"],
            mode="lines+markers", name=g,
            line=dict(color=COLORS[g], width=2),
            opacity=0.85, fill="toself",
            fillcolor=_hex_to_rgba(COLORS[g], 0.12),
        ))
    fig.update_layout(
        title=dict(text=net, font=dict(size=14)),
        polar=dict(
            radialaxis=dict(range=[0, 1.05], showticklabels=True,
                            tickfont=dict(size=9)),
            angularaxis=dict(tickfont=dict(size=10)),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(110,110,110,0.9)"),
        showlegend=True,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.05,
                    font=dict(size=10)),
        margin=dict(l=20, r=20, t=40, b=10), height=360,
    )
    col.plotly_chart(fig, width="stretch")

st.divider()

# ── Reading guide ─────────────────────────────────────────────────────
st.markdown(
    """
    ### Why GraphSAGE wins

    The four backbones differ in how a node aggregates information from
    its neighbours:

    | Backbone | Aggregator | What it adds |
    |---|---|---|
    | **GraphSAGE** | Mean of neighbour features + linear projection | Cheap, well-suited to homogeneous neighbourhoods |
    | GAT | Attention-weighted sum | Capacity to focus on individual neighbours |
    | GCN | Symmetric normalised mean | Spectral interpretation, slightly less expressive |
    | Transformer | Multi-head self-attention | Highest capacity, treats all neighbours like tokens |

    Pressure and flow on a water network are **smooth across
    neighbouring junctions** (Tobler's first law: nearby nodes are more
    similar than distant ones). The mean aggregator already captures
    that smoothness; attention spends extra parameters discriminating
    between neighbours that aren't in fact different. On a small graph
    this manifests as overfitting; on Modena it manifests as wasted
    capacity.

    The ablation isn't just a sanity check — it gives the thesis a
    direct empirical answer to *"why this architecture?"* across three
    networks of very different scale.
    """
)
