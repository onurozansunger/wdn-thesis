"""Page 5: Model Comparison — Spatial vs Temporal vs Mixture-of-Experts."""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import (
    load_test_results_net1, load_test_results_modena,
    load_temporal_results, load_moe_results,
)
from utils.theme import (
    GLOBAL_CSS, plotly_layout,
    BLUE, GREEN, ORANGE, RED, PURPLE, CYAN, DIM,
)

st.set_page_config(page_title="Model Comparison", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

st.title("Model Comparison")
st.caption(
    "Single GNN, then a temporal GRU, then attack-specialised experts — "
    "tracking what each architecture buys us"
)

ATTACK_LABELS = {
    "random": "Random",
    "replay": "Replay",
    "stealthy": "Stealthy",
    "noise": "Noise",
    "targeted": "Targeted",
}
ATTACK_COLORS = {
    "random": RED, "replay": PURPLE, "stealthy": ORANGE,
    "noise": CYAN, "targeted": "#e6c619",
}

net1_spatial = load_test_results_net1()
mod_spatial = load_test_results_modena()
net1_temp = load_temporal_results("Net1")
mod_temp = load_temporal_results("Modena")
net1_moe_s = load_moe_results("Net1", "spatial")
mod_moe_s = load_moe_results("Modena", "spatial")
net1_moe_t = load_moe_results("Net1", "temporal")
mod_moe_t = load_moe_results("Modena", "temporal")


def _recon_mae(r):
    return r["reconstruction"]["pressure_unobs"]["mae"] if r else None


def _anom_f1(r):
    return r["anomaly_detection"]["pressure"]["f1"] if r else None


def _anom_auroc(r):
    return r["anomaly_detection"]["pressure"]["auroc"] if r else None


# ──────────────────────────────────────────────
# Section 1 — Headline summary table
# ──────────────────────────────────────────────
st.markdown("##### Headline Metrics")

rows = []
for net_name, models in [
    ("Net1", [
        ("MultiTaskGNN (spatial)", net1_spatial),
        ("TemporalMultiTaskGNN", net1_temp),
        ("MoE (spatial)", net1_moe_s),
        ("MoE (temporal)", net1_moe_t),
    ]),
    ("Modena", [
        ("MultiTaskGNN (spatial)", mod_spatial),
        ("TemporalMultiTaskGNN", mod_temp),
        ("MoE (spatial)", mod_moe_s),
        ("MoE (temporal)", mod_moe_t),
    ]),
]:
    for label, res in models:
        if not res:
            continue
        rows.append({
            "Network": net_name,
            "Model": label,
            "Params": f"{res.get('n_params', 0):,}",
            "P_MAE (m)": f"{_recon_mae(res):.3f}" if _recon_mae(res) else "—",
            "F1": f"{_anom_f1(res):.3f}" if _anom_f1(res) else "—",
            "AUROC": f"{_anom_auroc(res):.3f}" if _anom_auroc(res) else "—",
        })

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()

# ──────────────────────────────────────────────
# Section 2 — Architecture progression (Net1 vs Modena)
# ──────────────────────────────────────────────
st.markdown("##### Architecture Progression")
st.markdown(
    "<span style='opacity:0.55; font-size:0.85rem;'>"
    "Each step adds one capability: temporal context, then per-attack "
    "expert specialisation.</span>",
    unsafe_allow_html=True,
)

col_n, col_m = st.columns(2)


def _arch_chart(spatial, temporal, moe_t, network_name):
    if not (spatial and temporal and moe_t):
        return None
    models = ["Spatial", "Temporal", "MoE\n(temporal)"]
    f1 = [_anom_f1(spatial), _anom_f1(temporal), _anom_f1(moe_t)]
    auroc = [_anom_auroc(spatial), _anom_auroc(temporal), _anom_auroc(moe_t)]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=models, y=f1, name="F1", marker_color=BLUE,
        text=[f"{v:.3f}" for v in f1], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=models, y=auroc, name="AUROC", marker_color=PURPLE,
        text=[f"{v:.3f}" for v in auroc], textposition="outside",
    ))
    fig.update_layout(**plotly_layout(
        title=dict(text=network_name),
        yaxis_title="Score", height=360,
        yaxis=dict(range=[0, 1.18]), barmode="group",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.10),
    ))
    return fig


with col_n:
    fig = _arch_chart(net1_spatial, net1_temp, net1_moe_t, "Net1 (11 nodes)")
    if fig:
        st.plotly_chart(fig, use_container_width=True)

with col_m:
    fig = _arch_chart(mod_spatial, mod_temp, mod_moe_t, "Modena (272 nodes)")
    if fig:
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ──────────────────────────────────────────────
# Section 3 — The replay story (per-attack F1)
# ──────────────────────────────────────────────
st.markdown("##### Per-Attack F1 — Where Specialisation Pays Off")
st.markdown(
    "<span style='opacity:0.55; font-size:0.85rem;'>"
    "Replay attacks replace a sensor reading with a past legitimate "
    "value. Spatially the value still looks plausible, so a "
    "single-snapshot model has nothing to flag. Only temporal context "
    "exposes the missing variability.</span>",
    unsafe_allow_html=True,
)

attacks = ["random", "replay", "stealthy", "noise", "targeted"]
labels = [ATTACK_LABELS[a] for a in attacks]


def _per_attack_chart(spatial_res, temporal_res, network_name):
    if not (spatial_res and temporal_res):
        return None
    spa_pa = spatial_res.get("per_attack_pressure", {})
    tmp_pa = temporal_res.get("per_attack_pressure", {})
    spa_f1 = [spa_pa.get(a, {}).get("f1", 0) for a in attacks]
    tmp_f1 = [tmp_pa.get(a, {}).get("f1", 0) for a in attacks]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=spa_f1, name="Spatial MoE", marker_color=BLUE,
        marker_opacity=0.55,
        text=[f"{v:.2f}" for v in spa_f1], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=labels, y=tmp_f1, name="Temporal MoE", marker_color=PURPLE,
        text=[f"{v:.2f}" for v in tmp_f1], textposition="outside",
    ))
    fig.update_layout(**plotly_layout(
        title=dict(text=network_name),
        yaxis_title="F1 Score", height=380,
        yaxis=dict(range=[0, 1.18]), barmode="group",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.10),
    ))
    return fig


col_a, col_b = st.columns(2)
with col_a:
    fig = _per_attack_chart(net1_moe_s, net1_moe_t, "Net1")
    if fig:
        st.plotly_chart(fig, use_container_width=True)
with col_b:
    fig = _per_attack_chart(mod_moe_s, mod_moe_t, "Modena")
    if fig:
        st.plotly_chart(fig, use_container_width=True)

# Replay improvement callout
if net1_moe_s and net1_moe_t and mod_moe_s and mod_moe_t:
    n_s = net1_moe_s["per_attack_pressure"]["replay"]["f1"]
    n_t = net1_moe_t["per_attack_pressure"]["replay"]["f1"]
    m_s = mod_moe_s["per_attack_pressure"]["replay"]["f1"]
    m_t = mod_moe_t["per_attack_pressure"]["replay"]["f1"]
    n_lift = n_t / max(n_s, 1e-3)
    m_lift = m_t / max(m_s, 1e-3)
    st.success(
        f"**Replay F1 — Net1**: {n_s:.3f} → **{n_t:.3f}** ({n_lift:.1f}× lift)  ·  "
        f"**Modena**: {m_s:.3f} → **{m_t:.3f}** ({m_lift:.0f}× lift)"
    )

st.divider()

# ──────────────────────────────────────────────
# Section 4 — Router behaviour
# ──────────────────────────────────────────────
st.markdown("##### Attack Router")
st.markdown(
    "<span style='opacity:0.55; font-size:0.85rem;'>"
    "The router is a small GNN that classifies the dominant attack class "
    "of the incoming window. High accuracy means each expert receives "
    "the targeted gradient signal it needs.</span>",
    unsafe_allow_html=True,
)

col_r1, col_r2 = st.columns(2)
with col_r1:
    rows_r = []
    for net_name, spatial, temporal in [
        ("Net1", net1_moe_s, net1_moe_t),
        ("Modena", mod_moe_s, mod_moe_t),
    ]:
        if spatial and temporal:
            rows_r.append({
                "Network": net_name,
                "Spatial MoE Router Acc": f"{spatial.get('router_acc', 0):.3f}",
                "Temporal MoE Router Acc": f"{temporal.get('router_acc', 0):.3f}",
            })
    st.dataframe(pd.DataFrame(rows_r), use_container_width=True, hide_index=True)

with col_r2:
    st.markdown(
        "**Reading the router**\n\n"
        "- ~0.50 on Net1: 11 nodes give the router very little context "
        "to disambiguate attack types.\n"
        "- ~0.97 on Modena: 272 nodes carry enough structure for the "
        "router to identify the active attack reliably."
    )
