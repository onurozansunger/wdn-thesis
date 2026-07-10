"""Page 5: Model Comparison — Spatial vs Temporal vs Mixture-of-Experts."""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import (
    load_test_results_net1, load_test_results_net3, load_test_results_modena,
    load_temporal_results, load_moe_results,
    load_rw_sweep_summary, load_router_diagnostic,
)
from utils.theme import (
    GLOBAL_CSS, plotly_layout,
    BLUE, GREEN, ORANGE, RED, PURPLE, CYAN, DIM, YELLOW,
)

st.set_page_config(page_title="Model Comparison", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

st.title("Model Comparison — Part 1")
st.caption(
    "Single GNN → temporal GRU → attack-specialised experts. "
    "Reproducible over 10 seeds, with the router diagnostic and the "
    "replay ceiling reported honestly."
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
net3_spatial = load_test_results_net3()
mod_spatial = load_test_results_modena()
net1_temp = load_temporal_results("Net1")
net3_temp = load_temporal_results("Net3")
mod_temp = load_temporal_results("Modena")
net1_moe_s = load_moe_results("Net1", "spatial")
mod_moe_s = load_moe_results("Modena", "spatial")
net1_moe_t = load_moe_results("Net1", "temporal")
net3_moe_t = load_moe_results("Net3", "temporal")
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
    ("Net3", [
        ("MultiTaskGNN (spatial)", net3_spatial),
        ("TemporalMultiTaskGNN", net3_temp),
        ("MoE (temporal)", net3_moe_t),
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

st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

st.divider()

# ──────────────────────────────────────────────
# Section 1b — Multi-seed reproducibility (Modena, 10 seeds)
# ──────────────────────────────────────────────
st.markdown("##### Reproducibility — 10-seed Modena (temporal MoE)")
st.markdown(
    "<span style='opacity:0.55; font-size:0.85rem;'>"
    "The four Part-1 upgrades the supervisors asked for — confidence-gated "
    "rerouting, a smaller router with bigger experts, direct expert "
    "supervision, and per-expert reconstruction — evaluated over 10 seeds. "
    "The headline F1 is stable to ±0.004.</span>",
    unsafe_allow_html=True,
)

rw = load_rw_sweep_summary()
if rw and "1.0" in rw:
    base = rw["1.0"]
    c = st.columns(4)
    c[0].metric("Anomaly F1", f"{base['pressure_f1']['mean']:.3f}",
                f"± {base['pressure_f1']['std']:.3f} over "
                f"{base['pressure_f1']['n']} seeds", delta_color="off")
    c[1].metric("AUROC", f"{base['pressure_auroc']['mean']:.3f}",
                f"± {base['pressure_auroc']['std']:.3f}", delta_color="off")
    c[2].metric("Flow F1", f"{base['flow_f1']['mean']:.3f}",
                f"± {base['flow_f1']['std']:.3f}", delta_color="off")
    c[3].metric("Router Acc", f"{base['router_acc']['mean']:.3f}",
                f"± {base['router_acc']['std']:.3f}", delta_color="off")

    # Per-attack F1 with error bars over the 10 seeds.
    pa = base["per_attack"]
    order = ["random", "targeted", "stealthy", "noise", "replay"]
    means = [pa[a]["mean"] for a in order]
    stds = [pa[a]["std"] for a in order]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[ATTACK_LABELS[a] for a in order], y=means,
        error_y=dict(type="data", array=stds, visible=True,
                     color="rgba(128,128,128,0.6)", thickness=1.2),
        marker_color=[ATTACK_COLORS[a] for a in order],
        text=[f"{m:.3f}" for m in means], textposition="outside",
    ))
    fig.update_layout(**plotly_layout(
        title=dict(text="Per-attack F1 (mean ± std, 10 seeds)"),
        yaxis_title="F1 Score", height=340,
        yaxis=dict(range=[0, 1.18]),
    ))
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Random, targeted, stealthy and noise are all detected reliably. "
        "Replay sits near zero — not a training artefact but an "
        "information-theoretic limit, quantified in the replay-ceiling "
        "section below."
    )
else:
    st.info("Run scripts/compare_rw_sweep.py to generate rw_sweep_summary.json.")

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

col_n, col_3, col_m = st.columns(3)


def _arch_chart(spatial, temporal, moe_t, network_name):
    # Need at least the spatial baseline and the MoE final result;
    # the standalone temporal model is shown only when available
    # (Net3 was trained straight to temporal-MoE and has no separate
    # temporal-multitask checkpoint).
    if not (spatial and moe_t):
        return None
    pieces = [("Spatial", spatial)]
    if temporal:
        pieces.append(("Temporal", temporal))
    pieces.append(("MoE\n(temporal)", moe_t))
    models = [p[0] for p in pieces]
    f1 = [_anom_f1(p[1]) for p in pieces]
    auroc = [_anom_auroc(p[1]) for p in pieces]
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
        st.plotly_chart(fig, width="stretch")

with col_3:
    fig = _arch_chart(net3_spatial, net3_temp, net3_moe_t, "Net3 (97 nodes)")
    if fig:
        st.plotly_chart(fig, width="stretch")

with col_m:
    fig = _arch_chart(mod_spatial, mod_temp, mod_moe_t, "Modena (272 nodes)")
    if fig:
        st.plotly_chart(fig, width="stretch")

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


col_a, col_c, col_b = st.columns(3)
with col_a:
    fig = _per_attack_chart(net1_moe_s, net1_moe_t, "Net1")
    if fig:
        st.plotly_chart(fig, width="stretch")
with col_c:
    # Net3 has no spatial-MoE checkpoint; show temporal-MoE alone.
    if net3_moe_t:
        fig = go.Figure()
        net3_pa = net3_moe_t.get("per_attack_pressure", {})
        net3_f1 = [net3_pa.get(a, {}).get("f1", 0) for a in attacks]
        fig.add_trace(go.Bar(
            x=labels, y=net3_f1, name="Temporal MoE",
            marker_color=PURPLE,
            text=[f"{v:.2f}" for v in net3_f1], textposition="outside",
        ))
        fig.update_layout(**plotly_layout(
            title=dict(text="Net3"),
            yaxis_title="F1 Score", height=380,
            yaxis=dict(range=[0, 1.18]), barmode="group",
            legend=dict(orientation="h", x=0.5, xanchor="center", y=1.10),
        ))
        st.plotly_chart(fig, width="stretch")
with col_b:
    fig = _per_attack_chart(mod_moe_s, mod_moe_t, "Modena")
    if fig:
        st.plotly_chart(fig, width="stretch")

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
# Section 4 — Router diagnostic (the supervisors' suspicion)
# ──────────────────────────────────────────────
st.markdown("##### Router Diagnostic — Is the router sending attacks to the right expert?")
st.markdown(
    "<span style='opacity:0.55; font-size:0.85rem;'>"
    "The supervisors suspected the router was mis-routing attacks. We "
    "checked directly. On Modena the router is healthy (96%). On Net3 it "
    "sends <b>77% of stealthy windows to the replay expert</b> — the "
    "suspicion was correct. Both classes look smooth and low-noise, and "
    "Net3's 97 nodes are not enough to separate them.</span>",
    unsafe_allow_html=True,
)

diag = load_router_diagnostic()
CLASS_NAMES = ["clean", "random", "replay", "stealthy", "noise", "targeted"]


def _confusion_heatmap(net_name, d):
    conf = d["confusion"]
    # Drop the empty 'clean' row/col (index 0) for a cleaner 5x5 view.
    idx = list(range(1, 6))
    z = [[conf[i][j] for j in idx] for i in idx]
    labels = [CLASS_NAMES[i] for i in idx]
    # Row-normalise for colour, annotate with raw counts.
    z_norm = []
    for row in z:
        tot = sum(row) or 1
        z_norm.append([v / tot for v in row])
    fig = go.Figure(data=go.Heatmap(
        z=z_norm, x=labels, y=labels,
        text=[[str(v) for v in row] for row in z],
        texttemplate="%{text}", textfont=dict(size=12),
        colorscale=[[0, "rgba(96,165,250,0.05)"], [1, BLUE]],
        showscale=False, hoverinfo="skip",
    ))
    fig.update_layout(**plotly_layout(
        title=dict(text=f"{net_name} · router acc {d['overall_acc']:.0%}"),
        height=360,
        xaxis=dict(title="Routed to expert", side="bottom"),
        yaxis=dict(title="True attack class", autorange="reversed"),
    ))
    return fig


if diag:
    col_m, col_3 = st.columns(2)
    with col_m:
        st.plotly_chart(_confusion_heatmap("Modena", diag["Modena"]),
                        width="stretch")
    with col_3:
        st.plotly_chart(_confusion_heatmap("Net3", diag["Net3"]),
                        width="stretch")

    net3_stealthy = diag["Net3"]["per_class_acc"]["stealthy"]
    mod_stealthy = diag["Modena"]["per_class_acc"]["stealthy"]
    st.warning(
        f"**Stealthy routing accuracy — Modena: {mod_stealthy:.0%}** "
        f"(healthy) · **Net3: {net3_stealthy:.0%}** (77% of stealthy "
        "windows land on the replay expert). "
        "Confidence-gated rerouting was added precisely so a wrong router "
        "call no longer collapses the prediction onto a single expert — "
        "the mix spreads when the router is unsure."
    )
else:
    st.info("Run scripts/diagnose_router.py to generate router_diagnostic.json.")

st.divider()

# ──────────────────────────────────────────────
# Section 5 — The replay ceiling (honest negative result)
# ──────────────────────────────────────────────
st.markdown("##### The Replay Ceiling — an honest negative result")
st.markdown(
    "<span style='opacity:0.55; font-size:0.85rem;'>"
    "Can we rescue replay by weighting it harder in the loss? We swept "
    "the replay weight over 10 seeds. Overall F1 and replay F1 trade off "
    "along a Pareto front — you cannot maximise both. In a six-hour "
    "Modena window pressure varies by ~0.01&nbsp;m while clean sensor "
    "noise is ~0.4&nbsp;m, so a k-step replay is almost indistinguishable "
    "from a clean signal. The ceiling is a property of the data, not the "
    "model.</span>",
    unsafe_allow_html=True,
)

if rw:
    weights = sorted(rw.keys(), key=float)
    overall = [rw[w]["pressure_f1"]["mean"] for w in weights]
    overall_std = [rw[w]["pressure_f1"]["std"] for w in weights]
    replay = [rw[w]["per_attack"]["replay"]["mean"] for w in weights]
    replay_std = [rw[w]["per_attack"]["replay"]["std"] for w in weights]
    n_seeds = [rw[w]["pressure_f1"]["n"] for w in weights]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[float(w) for w in weights], y=overall,
        error_y=dict(type="data", array=overall_std, visible=True,
                     color="rgba(96,165,250,0.35)"),
        name="Overall F1", mode="lines+markers",
        line=dict(color=BLUE, width=2.5), marker=dict(size=9),
    ))
    fig.add_trace(go.Scatter(
        x=[float(w) for w in weights], y=replay,
        error_y=dict(type="data", array=replay_std, visible=True,
                     color="rgba(167,139,250,0.35)"),
        name="Replay F1", mode="lines+markers",
        line=dict(color=PURPLE, width=2.5), marker=dict(size=9),
    ))
    fig.update_layout(**plotly_layout(
        title=dict(text="Replay weight sweep — overall vs replay F1"),
        xaxis_title="Replay loss weight", yaxis_title="F1 Score",
        height=380, yaxis=dict(range=[0, 1.0]),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.12),
    ))
    col_p, col_t = st.columns([3, 2], gap="large")
    with col_p:
        st.plotly_chart(fig, width="stretch")
    with col_t:
        rows_rw = []
        for w in weights:
            rows_rw.append({
                "RW": w,
                "Overall F1": f"{rw[w]['pressure_f1']['mean']:.3f}",
                "Replay F1": f"{rw[w]['per_attack']['replay']['mean']:.3f}",
                "Seeds": rw[w]["pressure_f1"]["n"],
            })
        st.dataframe(pd.DataFrame(rows_rw), width="stretch", hide_index=True)
        d_overall = rw["1.0"]["pressure_f1"]["mean"] - rw["2.5"]["pressure_f1"]["mean"]
        d_replay = rw["2.5"]["per_attack"]["replay"]["mean"] - rw["1.0"]["per_attack"]["replay"]["mean"]
        st.markdown(
            f"<span style='font-size:0.85rem; opacity:0.75;'>"
            f"Pushing replay from rw=1→2.5 lifts replay F1 by only "
            f"<b>{d_replay:+.3f}</b> while overall F1 drops "
            f"<b>{-d_overall:.3f}</b>. The framework reaches the ceiling; "
            f"beating it needs the attacker side (Part 2).</span>",
            unsafe_allow_html=True,
        )
