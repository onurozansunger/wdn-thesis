"""Plotly network graph builder for WDN visualization."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from utils.theme import (
    NODE_JUNCTION, NODE_RESERVOIR, NODE_TANK, GRAY, DIM,
    CHART_BG, TEXT_COLOR, TEXT_DIM, plotly_layout,
)

# Color scales
PRESSURE_COLORSCALE = [
    [0.0, "#0d1b2a"], [0.2, "#1b3a5c"], [0.4, "#2a6f97"],
    [0.6, "#4da6ff"], [0.8, "#89c2ff"], [1.0, "#d4e9ff"],
]
ERROR_COLORSCALE = [
    [0.0, "#1a1a2e"], [0.3, "#5c2d2d"], [0.6, "#c0392b"],
    [0.8, "#e74c3c"], [1.0, "#f8b4b4"],
]
UNCERTAINTY_COLORSCALE = [
    [0.0, "#1a1a2e"], [0.3, "#5c3d1e"], [0.6, "#d4760a"],
    [0.8, "#fb923c"], [1.0, "#fde68a"],
]

# Node type styling
TYPE_SYMBOLS = {0: "circle", 1: "diamond", 2: "square"}
TYPE_NAMES = {0: "Junction", 1: "Reservoir", 2: "Tank"}
TYPE_COLORS = {0: NODE_JUNCTION, 1: NODE_RESERVOIR, 2: NODE_TANK}


def build_network_figure(
    graph,
    node_values=None,
    node_text=None,
    colorscale=PRESSURE_COLORSCALE,
    title="",
    color_label="Value",
    show_colorbar=True,
    node_size=32,
    discrete_colors=None,
    height=500,
    edge_width=3.0,
):
    """Build a Plotly figure of the WDN network."""
    coords = graph.node_coordinates
    edge_index = graph.edge_index
    N = graph.num_nodes

    # --- Edges ---
    edge_x, edge_y = [], []
    for j in range(edge_index.shape[1]):
        src, dst = edge_index[0, j], edge_index[1, j]
        edge_x += [coords[src, 0], coords[dst, 0], None]
        edge_y += [coords[src, 1], coords[dst, 1], None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=edge_width, color="rgba(255,255,255,0.12)"),
        hoverinfo="none",
    )

    # --- Edge midpoint labels ---
    mid_x, mid_y, mid_text = [], [], []
    for j in range(edge_index.shape[1]):
        src, dst = edge_index[0, j], edge_index[1, j]
        mx = (coords[src, 0] + coords[dst, 0]) / 2
        my = (coords[src, 1] + coords[dst, 1]) / 2
        mid_x.append(mx)
        mid_y.append(my)
        mid_text.append(
            f"<b>Pipe {graph.edge_names[j]}</b><br>"
            f"{graph.node_names[src]} → {graph.node_names[dst]}"
        )

    edge_label_trace = go.Scatter(
        x=mid_x, y=mid_y,
        mode="markers",
        marker=dict(size=8, color="rgba(255,255,255,0.0)"),
        hovertext=mid_text,
        hoverinfo="text",
    )

    # --- Nodes ---
    node_x = coords[:, 0].tolist()
    node_y = coords[:, 1].tolist()

    if node_text is None:
        node_text = []
        for i in range(N):
            t = TYPE_NAMES.get(graph.node_types[i], "Unknown")
            txt = f"<b>Node {graph.node_names[i]}</b><br>Type: {t}"
            if node_values is not None:
                txt += f"<br>{color_label}: {node_values[i]:.4f}"
            node_text.append(txt)

    symbols = [TYPE_SYMBOLS.get(graph.node_types[i], "circle") for i in range(N)]

    marker_kwargs = dict(
        size=node_size,
        symbol=symbols,
        line=dict(width=2, color="rgba(255,255,255,0.25)"),
    )

    if discrete_colors is not None:
        marker_kwargs["color"] = discrete_colors
    elif node_values is not None:
        marker_kwargs.update(
            color=list(node_values),
            colorscale=colorscale,
            showscale=show_colorbar,
            colorbar=dict(
                title=dict(text=color_label, font=dict(size=11, color=TEXT_DIM)),
                thickness=14, len=0.55,
                tickfont=dict(size=10, color=TEXT_DIM),
                outlinewidth=0,
                bgcolor="rgba(0,0,0,0)",
            ) if show_colorbar else None,
        )
    else:
        marker_kwargs["color"] = [TYPE_COLORS.get(graph.node_types[i], GRAY) for i in range(N)]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=marker_kwargs,
        text=[graph.node_names[i] for i in range(N)],
        textposition="top center",
        textfont=dict(size=10, color="rgba(255,255,255,0.6)"),
        hovertext=node_text,
        hoverinfo="text",
    )

    fig = go.Figure(data=[edge_trace, edge_label_trace, node_trace])
    fig.update_layout(
        **plotly_layout(
            title=dict(text=title, x=0.5, font=dict(size=14, color=TEXT_DIM)),
            height=height,
            showlegend=False,
            hovermode="closest",
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        ),
    )
    return fig
