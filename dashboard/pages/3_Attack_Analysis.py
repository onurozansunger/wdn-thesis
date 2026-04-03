"""Page 3: Attack Analysis — Per-type attack impact and detection."""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import network_selector, load_attack_analysis
from utils.theme import (
    GLOBAL_CSS, plotly_layout,
    BLUE, GREEN, ORANGE, RED, PURPLE, CYAN, YELLOW, GRAY, DIM,
)

st.set_page_config(page_title="Attack Analysis", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

st.title("Attack Type Analysis")
network = network_selector(key="attack_net")
st.caption("Impact and detectability comparison across 5 adversarial attack types")

data = load_attack_analysis(network)
if data is None:
    st.warning(f"Attack analysis data not available for {network}. Run the precompute script.")
    st.stop()

results = data["results"]
node_names = data["node_names"]
attack_types = data["attack_types"]

ATTACK_COLORS = {
    "random": RED, "replay": PURPLE, "stealthy": ORANGE,
    "noise": CYAN, "targeted": YELLOW,
}

# ── Overview ──
st.markdown("##### Detection Performance by Attack Type")

col_f1, col_dev = st.columns(2)

with col_f1:
    fig = go.Figure()
    for atype in attack_types:
        r = results[atype]
        fracs = [d["fraction"] * 100 for d in r["fraction_data"]]
        f1s = [d["f1"] for d in r["fraction_data"]]
        fig.add_trace(go.Scatter(
            x=fracs, y=f1s, mode="lines+markers", name=r["label"],
            line=dict(color=ATTACK_COLORS[atype], width=2.5), marker=dict(size=7),
        ))
    fig.update_layout(**plotly_layout(
        title=dict(text="Detection F1 Score vs Attack Fraction"),
        xaxis_title="Attack Fraction (%)", yaxis_title="F1 Score",
        height=400, yaxis=dict(range=[0, 1.05]),
        legend=dict(x=0.02, y=0.02, yanchor="bottom"),
    ))
    st.plotly_chart(fig, use_container_width=True)

with col_dev:
    fig = go.Figure()
    for atype in attack_types:
        r = results[atype]
        fracs = [d["fraction"] * 100 for d in r["fraction_data"]]
        devs = [d["mean_deviation_m"] for d in r["fraction_data"]]
        fig.add_trace(go.Scatter(
            x=fracs, y=devs, mode="lines+markers", name=r["label"],
            line=dict(color=ATTACK_COLORS[atype], width=2.5), marker=dict(size=7),
        ))
    fig.update_layout(**plotly_layout(
        title=dict(text="Mean Pressure Deviation vs Attack Fraction"),
        xaxis_title="Attack Fraction (%)", yaxis_title="Deviation (m)",
        height=400, legend=dict(x=0.02, y=0.98, yanchor="top"),
    ))
    st.plotly_chart(fig, use_container_width=True)

# ── Per-Attack Detail ──
st.divider()
st.markdown("##### Per-Attack-Type Detail")

for atype in attack_types:
    r = results[atype]
    frac_data = r["fraction_data"]

    with st.expander(f"{r['label']}", expanded=False):
        st.markdown(f"<span style='opacity:0.6; font-size:0.85rem;'>{r['description']}</span>",
                    unsafe_allow_html=True)

        rep = next((d for d in frac_data if d["fraction"] == 0.15), frac_data[2])
        col1, col2, col3 = st.columns(3)
        col1.metric("Precision", f"{rep['precision']:.1%}")
        col2.metric("Recall", f"{rep['recall']:.1%}")
        col3.metric("Mean Deviation", f"{rep['mean_deviation_m']:.1f} m")

        col_left, col_right = st.columns(2)

        with col_left:
            fig = go.Figure()
            fracs = [d["fraction"] * 100 for d in frac_data]
            fig.add_trace(go.Scatter(x=fracs, y=[d["precision"] for d in frac_data],
                                     mode="lines+markers", name="Precision",
                                     line=dict(color=BLUE, width=2.5), marker=dict(size=6)))
            fig.add_trace(go.Scatter(x=fracs, y=[d["recall"] for d in frac_data],
                                     mode="lines+markers", name="Recall",
                                     line=dict(color=GREEN, width=2.5), marker=dict(size=6)))
            fig.add_trace(go.Scatter(x=fracs, y=[d["f1"] for d in frac_data],
                                     mode="lines+markers", name="F1",
                                     line=dict(color=ORANGE, width=2.5, dash="dash"), marker=dict(size=6)))
            fig.update_layout(**plotly_layout(
                title=dict(text="Detection Metrics vs Fraction"),
                xaxis_title="Sensors Attacked (%)", yaxis_title="Score",
                height=350, yaxis=dict(range=[0, 1.05]),
                legend=dict(x=0.65, y=0.05, yanchor="bottom"),
            ))
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            pct_affected = [d["pct_affected"] for d in frac_data]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=fracs, y=pct_affected, marker_color=ATTACK_COLORS[atype], opacity=0.85,
                text=[f"{p:.1f}%" for p in pct_affected], textposition="outside", textfont=dict(size=11),
            ))
            fig.update_layout(**plotly_layout(
                title=dict(text="Percentage of Nodes Affected"),
                xaxis_title="Attack Fraction (%)", yaxis_title="Affected Nodes (%)",
                height=350, showlegend=False, yaxis=dict(range=[0, max(pct_affected) * 1.25]),
            ))
            st.plotly_chart(fig, use_container_width=True)

        # Node-level bar chart
        node_analysis = r.get("node_analysis")
        if node_analysis:
            names = [n["node_name"] for n in node_analysis]
            devs = [n["mean_deviation"] for n in node_analysis]
            if max(devs) > 0:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=names, y=devs, marker_color=ATTACK_COLORS[atype], opacity=0.85,
                    text=[f"{d:.1f}" if d > 0 else "" for d in devs],
                    textposition="outside", textfont=dict(size=10),
                ))
                fig.update_layout(**plotly_layout(
                    title=dict(text="Mean Pressure Deviation per Node"),
                    xaxis_title="Node", yaxis_title="Deviation (m)",
                    height=300, showlegend=False,
                ))
                st.plotly_chart(fig, use_container_width=True)

# ── Summary Table ──
st.divider()
st.markdown("##### Summary at 15% Attack Fraction")

summary_rows = []
for atype in attack_types:
    r = results[atype]
    rep = next((d for d in r["fraction_data"] if d["fraction"] == 0.15), r["fraction_data"][2])
    summary_rows.append({
        "Attack Type": r["label"],
        "Affected (%)": f"{rep['pct_affected']:.1f}%",
        "Deviation (m)": f"{rep['mean_deviation_m']:.1f}",
        "Precision": f"{rep['precision']:.1%}",
        "Recall": f"{rep['recall']:.1%}",
        "F1": f"{rep['f1']:.3f}",
        "Detectability": "Hard" if rep['f1'] < 0.4 else "Medium" if rep['f1'] < 0.7 else "Easy",
    })

st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
