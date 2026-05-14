"""Methodology page — full pipeline at a glance.

Walks the reader from raw simulation through every architectural stage
to the self-play loop, with a little bit of prose and a lot of visual.
The diagrams are drawn with Plotly shapes so they stay crisp and
themable from the same colour palette as every other page.
"""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

DASHBOARD = Path(__file__).parent.parent
sys.path.insert(0, str(DASHBOARD))

from utils.theme import (
    GLOBAL_CSS, BLUE, GREEN, ORANGE, RED, PURPLE, CYAN, GRAY, DIM,
)


st.set_page_config(page_title="Methodology", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

st.title("Methodology — How the pieces fit together")
st.caption(
    "End-to-end pipeline, architecture progression, and the "
    "Stackelberg self-play loop on a single page."
)

# ── 1. Pipeline overview ─────────────────────────────────────────────
st.markdown("### Pipeline overview")
st.caption(
    "Data flows left to right. Every step is a distinct module in "
    "`src/wdn/` so each stage can be studied or replaced in isolation."
)


def _box(fig, x, y, w, h, label, color, sublabel=None, text_color="white"):
    fig.add_shape(
        type="rect", x0=x, y0=y, x1=x + w, y1=y + h,
        line=dict(color=color, width=2),
        fillcolor=color, opacity=0.18,
    )
    fig.add_annotation(
        x=x + w / 2, y=y + h / 2 + (0.04 if sublabel else 0),
        text=f"<b>{label}</b>", showarrow=False,
        font=dict(color=text_color, size=12),
    )
    if sublabel:
        fig.add_annotation(
            x=x + w / 2, y=y + h / 2 - 0.06,
            text=f"<span style='font-size:10px'>{sublabel}</span>",
            showarrow=False, font=dict(color=text_color, size=10),
        )


def _arrow(fig, x0, y0, x1, y1, color=GRAY):
    fig.add_annotation(
        x=x1, y=y1, ax=x0, ay=y0,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2,
        arrowcolor=color,
    )


def pipeline_figure():
    fig = go.Figure()
    fig.update_layout(
        xaxis=dict(range=[0, 12], showgrid=False, zeroline=False,
                   visible=False),
        yaxis=dict(range=[0, 1.2], showgrid=False, zeroline=False,
                   visible=False, scaleanchor="x", scaleratio=1),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=20, b=20), height=240,
    )

    boxes = [
        (0.2, "EPANET .inp", BLUE, "Net1 / Net3 / Modena"),
        (2.4, "WNTR sim", CYAN, "Hydraulic snapshots"),
        (4.6, "Corruption", ORANGE, "Missing + 5 attacks"),
        (6.8, "GNN model", GREEN, "Reconstruct + flag"),
        (9.0, "Outputs", PURPLE, "Pressure, flow, anomaly"),
    ]
    for x, label, color, sub in boxes:
        _box(fig, x, 0.4, 1.8, 0.5, label, color, sub)
    for i in range(len(boxes) - 1):
        _arrow(fig, boxes[i][0] + 1.8, 0.65, boxes[i + 1][0], 0.65)

    return fig


st.plotly_chart(pipeline_figure(), width="stretch",
                config={"displayModeBar": False})

st.divider()

# ── 2. Architecture progression ──────────────────────────────────────
st.markdown("### Architecture progression")
st.caption(
    "Five iterations on the same defender. Each row addresses a "
    "specific weakness of the previous one."
)

stages = [
    ("Spatial GNN",
     GREEN,
     "GraphSAGE backbone, 2 layers, 4 prediction heads. Mass-conservation penalty.",
     "Doesn't see time → cannot flag replay attacks."),
    ("+ Temporal GRU",
     CYAN,
     "GRU over a 6-step sliding window with explicit temporal-stability features.",
     "Single model has to compromise across attack classes."),
    ("+ Mixture-of-Experts",
     PURPLE,
     "Six experts, learned router, load-balancing entropy + class supervision.",
     "Pattern-detection signal still tied to a fixed dataset."),
    ("+ Pattern features",
     ORANGE,
     "Lag-1 autocorrelation, adjacent-difference std, noise ratio fed into the anomaly head.",
     "Hand-crafted attack zoo limits robustness to unseen attacks."),
    ("+ Adversarial self-play",
     RED,
     "Attacker GNN learns sparse, physics-aware perturbations; defender catches them in a Stackelberg loop.",
     "→ generalises to held-out novel attacks."),
]

for i, (name, color, body, weakness) in enumerate(stages):
    with st.container():
        c1, c2 = st.columns([1, 4])
        with c1:
            st.markdown(
                f"<div style='background:{color}; opacity:0.85; "
                f"color:white; padding:14px; border-radius:8px; "
                f"text-align:center; font-weight:600;'>"
                f"Stage {i+1}<br><span style='font-size:0.85rem; "
                f"opacity:0.95'>{name}</span></div>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(f"**What it does.** {body}")
            if i < len(stages) - 1:
                st.markdown(
                    f"<span style='opacity:0.55; font-size:0.85rem'>"
                    f"⤷ remaining gap: {weakness}</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<span style='color:{GREEN}; font-size:0.85rem'>"
                    f"✔ {weakness}</span>",
                    unsafe_allow_html=True,
                )

st.divider()

# ── 3. Self-play loop ────────────────────────────────────────────────
st.markdown("### Self-play loop (Stage 5 in detail)")
st.caption(
    "The attacker and defender play a Stackelberg game. The attacker "
    "leads (chooses a perturbation given the current defender); the "
    "defender follows (updates its weights to catch the new attack). "
    "Auto-curriculum grows the budget once the defender starts winning."
)


def selfplay_figure():
    fig = go.Figure()
    fig.update_layout(
        xaxis=dict(range=[0, 12], showgrid=False, zeroline=False,
                   visible=False),
        yaxis=dict(range=[-1, 4], showgrid=False, zeroline=False,
                   visible=False, scaleanchor="x", scaleratio=1),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=20, b=20), height=420,
    )

    # Attacker
    _box(fig, 0.4, 2.2, 3.5, 1.3, "Attacker GNN", RED,
         "Top-k mask · ε-bounded δ · physics-aware")
    # Defender
    _box(fig, 8.1, 2.2, 3.5, 1.3, "Defender (Temporal MoE)", BLUE,
         "Reconstruct · flag compromised sensors")
    # Stealth budget
    _box(fig, 4.4, 2.5, 3.3, 0.7, "Stealth-budget projection",
         ORANGE, "‖δ‖∞ ≤ ε · top-k sensors only")
    # Auto-curriculum
    _box(fig, 4.4, 0.7, 3.3, 0.7, "Auto-curriculum",
         GREEN, "Bump (ε, k) when adv-F1 > θ")

    # Arrows
    _arrow(fig, 3.9, 2.85, 4.4, 2.85, color=RED)
    _arrow(fig, 7.7, 2.85, 8.1, 2.85, color=ORANGE)
    _arrow(fig, 8.1, 2.4, 7.7, 1.05, color=BLUE)   # def → curriculum
    _arrow(fig, 4.4, 1.05, 3.9, 2.4, color=GREEN)  # curriculum → atk

    # Labels along arrows
    fig.add_annotation(x=4.0, y=3.05, text="raw δ + mask",
                       showarrow=False,
                       font=dict(size=9, color=GRAY))
    fig.add_annotation(x=7.9, y=3.05, text="perturbed snapshot",
                       showarrow=False,
                       font=dict(size=9, color=GRAY))
    fig.add_annotation(x=7.6, y=1.7, text="adv-F1",
                       showarrow=False,
                       font=dict(size=9, color=GRAY))
    fig.add_annotation(x=4.4, y=1.7, text="ε ↑, k ↑",
                       showarrow=False,
                       font=dict(size=9, color=GRAY))

    return fig


st.plotly_chart(selfplay_figure(), width="stretch",
                config={"displayModeBar": False})

st.markdown(
    r"""
    **Attacker objective.**
    $\quad L_{\mathrm{atk}} \;=\; L_{\mathrm{stealth}}
    \;-\; \lambda_{\mathrm{recon}} L_{\mathrm{damage}}
    \;+\; \lambda_{\mathrm{phys}} \|B\,q\|^2 \;+\; (\text{MoE aux})$

    - $L_{\mathrm{stealth}}$ — BCE that rewards the attacker when the
      defender does **not** flag the sensors it touched.
    - $L_{\mathrm{damage}}$ — defender reconstruction error on the same
      sensors (negated, so the attacker maximises it).
    - $\|B\,q\|^2$ — soft mass-conservation penalty on the perturbed
      flow vector.

    **Defender objective.** Standard temporal-MoE loss (recon MSE +
    class-weighted BCE + router CE + balance entropy) computed against
    the attacker-injected labels.
    """
)

st.divider()

# ── 4. Where to go next ──────────────────────────────────────────────
st.markdown("### Where to go next on this dashboard")

st.markdown(
    """
    | Page | Question it answers |
    |------|---------------------|
    | **Network Overview** | What does the topology look like? |
    | **Reconstruction** | True vs predicted pressure / flow per snapshot |
    | **Attack Analysis** | How does each attack class look on the wire? |
    | **Anomaly Detection** | Interactive thresholding — precision/recall trade-off |
    | **Model Comparison** | What does each architectural step buy us? |
    | **Explainability** | Which features and nodes drive each prediction? |
    | **Self-Play** | Co-evolution dynamics + held-out generalisation + emergent vocabulary |
    | **GNN Ablation** | Why GraphSAGE? (4 backbones × 3 networks) |
    """
)
