"""Page 5: Cross-Domain — the same temporal-MoE GNN on water, power and
traffic sensor networks. The flagship generalisation result."""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_crossdomain_summary
from utils.theme import (
    GLOBAL_CSS, plotly_layout, SURFACE,
    BLUE, ORANGE, GREEN, PURPLE, CYAN, TEXT_COLOR, TEXT_DIM, TEXT_MUTED,
    ATTACK_LABELS,
)

st.set_page_config(page_title="Cross-Domain", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ── Domain metadata ──────────────────────────────────────────────────
DOMAINS = [
    ("water",   "Water",   "Modena distribution net", "💧", BLUE,
     "272 buses · 317 pipes", "bus pressure", "pressure barely moves in a "
     "6-hour window"),
    ("power",   "Power",   "IEEE 118-bus grid",        "⚡", ORANGE,
     "118 buses · 186 lines", "bus voltage", "voltage drifts slowly with load"),
    ("traffic", "Traffic", "200-sensor road network",  "🚗", GREEN,
     "200 sensors · 675 roads", "sensor speed", "speed swings hard at rush hour"),
]
DKEYS = {"water": "water_modena_10seed", "power": "power_ieee118_5seed",
         "traffic": "traffic_200_5seed"}
DCOLOR = {"water": BLUE, "power": ORANGE, "traffic": GREEN}
DLABEL = {"water": "Water", "power": "Power", "traffic": "Traffic"}

data = load_crossdomain_summary()

# ── Hero ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .cd-hero {
        background: linear-gradient(135deg,
            rgba(37,99,235,0.08) 0%, rgba(234,88,12,0.06) 50%,
            rgba(5,150,105,0.07) 100%);
        border: 1px solid rgba(31,41,55,0.10);
        border-radius: 14px; padding: 20px 24px; margin-bottom: 4px;
    }
    .cd-eyebrow { font-size: 0.72rem; letter-spacing: 0.16em;
        text-transform: uppercase; opacity: 0.55; margin-bottom: 6px; }
    .cd-title { font-size: 1.5rem; font-weight: 680; line-height: 1.2;
        letter-spacing: -0.02em; margin-bottom: 8px; }
    .cd-sub { font-size: 0.95rem; opacity: 0.78; line-height: 1.55; }
    </style>
    <div class="cd-hero">
      <div class="cd-eyebrow">Part 1 · Generalisation</div>
      <div class="cd-title">One architecture, three physical systems</div>
      <div class="cd-sub">
        The supervisors asked whether the approach holds beyond water.
        The <b>identical</b> temporal Mixture-of-Experts GNN — same 376K
        parameters, same five attacks, same corruption pipeline, not a
        line changed — was retrained on an electrical grid and a road
        network. It reaches the same detection quality on all three, and
        exposes <b>why</b> one attack class behaves so differently across
        domains.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

if not data:
    st.info("Run the cross-domain aggregation to produce "
            "runs/temporal_moe/crossdomain_summary.json.")
    st.stop()


def stat(key, metric):
    m, s = data[DKEYS[key]][metric]
    return m, s


# ── Domain stat cards ────────────────────────────────────────────────
cols = st.columns(3)
for col, (key, name, desc, emoji, color, topo, sensor, why) in zip(cols, DOMAINS):
    f1, f1s = stat(key, "F1")
    col.markdown(
        f"""
        <div style="border:1px solid rgba(31,41,55,0.10); border-radius:13px;
             padding:16px 18px; background:linear-gradient(180deg,
             {color}0d, {color}03); height:100%;">
          <div style="font-size:1.5rem; line-height:1;">{emoji}</div>
          <div style="font-weight:670; font-size:1.05rem; margin-top:6px;
               color:{color};">{name}</div>
          <div style="font-size:0.78rem; opacity:0.6; margin-bottom:10px;">
               {desc} · {topo}</div>
          <div style="font-size:1.9rem; font-weight:700; letter-spacing:-0.02em;">
               {f1:.3f}<span style="font-size:0.9rem; font-weight:500;
               opacity:0.5;"> ±{f1s:.3f}</span></div>
          <div style="font-size:0.72rem; letter-spacing:0.05em;
               text-transform:uppercase; opacity:0.5;">detection F1</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
st.divider()

# ── Section 1 — Overall detection, side by side ──────────────────────
st.markdown("##### Same detection quality everywhere")
st.markdown(
    "<span style='opacity:0.62; font-size:0.86rem;'>"
    "Overall anomaly-detection F1 and AUROC, mean over seeds "
    "(water 10, power 5, traffic 5). The method transfers with no "
    "domain-specific tuning.</span>",
    unsafe_allow_html=True,
)

dnames = [DLABEL[k] for k in ["water", "power", "traffic"]]
f1_m = [stat(k, "F1")[0] for k in ["water", "power", "traffic"]]
f1_s = [stat(k, "F1")[1] for k in ["water", "power", "traffic"]]
au_m = [stat(k, "AUROC")[0] for k in ["water", "power", "traffic"]]
au_s = [stat(k, "AUROC")[1] for k in ["water", "power", "traffic"]]

c1, c2 = st.columns([3, 2], gap="large")
with c1:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dnames, y=f1_m, name="F1", marker_color=BLUE, marker_line_width=0,
        error_y=dict(type="data", array=f1_s, color="rgba(31,41,55,0.4)",
                     thickness=1.2, width=4),
        text=[f"{v:.3f}" for v in f1_m], textposition="outside",
        textfont=dict(color=TEXT_COLOR, size=11.5),
        hovertemplate="%{x} · F1 %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=dnames, y=au_m, name="AUROC", marker_color=CYAN, marker_line_width=0,
        error_y=dict(type="data", array=au_s, color="rgba(31,41,55,0.4)",
                     thickness=1.2, width=4),
        text=[f"{v:.3f}" for v in au_m], textposition="outside",
        textfont=dict(color=TEXT_COLOR, size=11.5),
        hovertemplate="%{x} · AUROC %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(**plotly_layout(
        yaxis_title="Score", height=380, yaxis=dict(range=[0, 1.12]),
        barmode="group", bargap=0.42, bargroupgap=0.08,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.14),
    ))
    st.plotly_chart(fig, width="stretch")
with c2:
    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    rows = []
    for k in ["water", "power", "traffic"]:
        f1, f1s = stat(k, "F1")
        au, aus = stat(k, "AUROC")
        rt, rts = stat(k, "router")
        rows.append({"Domain": DLABEL[k], "F1": f"{f1:.3f}",
                     "AUROC": f"{au:.3f}", "Router": f"{rt:.3f}"})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.markdown(
        "<span style='font-size:0.83rem; opacity:0.72;'>"
        "F1 lands in a tight <b>0.845–0.870</b> band across three "
        "unrelated physical systems — strong evidence the temporal-MoE "
        "detector is domain-agnostic.</span>",
        unsafe_allow_html=True,
    )

st.divider()

# ── Section 2 — The replay-ceiling insight (the star) ────────────────
st.markdown("##### Why replay detection depends on the domain, not the model")
st.markdown(
    "<span style='opacity:0.62; font-size:0.86rem;'>"
    "A replay attack re-broadcasts a stale sensor reading. Whether that "
    "is detectable depends entirely on how fast the underlying signal "
    "moves — a property of the <i>domain</i>, not the network. Ordered "
    "by signal speed, replay F1 climbs from near-zero to near-one.</span>",
    unsafe_allow_html=True,
)

order = ["water", "power", "traffic"]
rep_m = [stat(k, "replay")[0] for k in order]
rep_s = [stat(k, "replay")[1] for k in order]
why = {"water": "pressure ~0.01 m / window",
       "power": "voltage drifts slowly",
       "traffic": "speed swings at rush hour"}

c1, c2 = st.columns([3, 2], gap="large")
with c1:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[DLABEL[k] for k in order], y=rep_m,
        marker_color=[DCOLOR[k] for k in order], marker_line_width=0,
        width=0.56,
        error_y=dict(type="data", array=rep_s, color="rgba(31,41,55,0.35)",
                     thickness=1.2, width=5),
        text=[f"{v:.3f}" for v in rep_m], textposition="outside",
        textfont=dict(color=TEXT_COLOR, size=13, weight=600),
        customdata=[why[k] for k in order],
        hovertemplate="%{x} · replay F1 %{y:.3f}<br>%{customdata}<extra></extra>",
    ))
    fig.update_layout(**plotly_layout(
        yaxis_title="Replay-attack F1", height=390,
        yaxis=dict(range=[0, 1.05]),
        xaxis=dict(title="signal speed  →  faster"),
    ))
    # Annotate the trend direction.
    fig.add_annotation(
        x=0, y=rep_m[0] + 0.12, text="undetectable", showarrow=False,
        font=dict(color=TEXT_MUTED, size=10.5))
    fig.add_annotation(
        x=2, y=rep_m[2] - 0.10, text="easily caught", showarrow=False,
        font=dict(color="#ffffff", size=10.5))
    st.plotly_chart(fig, width="stretch")
with c2:
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="border-left:3px solid {PURPLE}; padding:2px 0 2px 14px;">
        <div style="font-weight:660; font-size:0.95rem; margin-bottom:8px;">
        The replay ceiling is physical, not a bug.</div>
        <div style="font-size:0.87rem; opacity:0.8; line-height:1.6;">
        In water and power the monitored quantity barely changes between
        consecutive readings, so a replayed value sits <i>inside the
        sensor-noise band</i> — information-theoretically invisible.
        Traffic speed, by contrast, drops sharply during rush hour, so a
        stale reading stands out immediately.<br><br>
        The same model reports <b>replay F1 0.02 → 0.27 → 0.87</b> across
        the three domains purely because of signal dynamics — turning a
        one-domain negative result into a general principle.
        </div></div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ── Section 3 — Per-attack difficulty profile (heatmap) ──────────────
st.markdown("##### The difficulty profile inverts between domains")
st.markdown(
    "<span style='opacity:0.62; font-size:0.86rem;'>"
    "Per-attack F1 for every domain. Water and power find stealthy and "
    "noise easy but replay impossible; traffic is the mirror image — "
    "replay is easy while stealthy blends into natural congestion. Same "
    "detector, opposite hard cases.</span>",
    unsafe_allow_html=True,
)

attacks = ["random", "targeted", "stealthy", "noise", "replay"]
alabels = [ATTACK_LABELS[a] for a in attacks]
z, textz = [], []
for k in order:
    rowv = [stat(k, a)[0] for a in attacks]
    z.append(rowv)
    textz.append([f"{v:.2f}" for v in rowv])

# Sequential single-hue scale (light surface → BLUE), per the color formula.
BLUE_SCALE = [[0.0, "#eef4ff"], [0.5, "#8fb6f5"], [1.0, BLUE]]
fig = go.Figure(data=go.Heatmap(
    z=z, x=alabels, y=[DLABEL[k] for k in order],
    text=textz, texttemplate="%{text}",
    textfont=dict(size=13, color=TEXT_COLOR),
    colorscale=BLUE_SCALE, zmin=0, zmax=1,
    xgap=3, ygap=3, showscale=True,
    colorbar=dict(title=dict(text="F1", side="right"), thickness=12,
                  len=0.85, x=1.01),
    hovertemplate="%{y} · %{x}<br>F1 %{z:.3f}<extra></extra>",
))
fig.update_layout(**plotly_layout(
    height=300,
    yaxis=dict(autorange="reversed"),
    margin=dict(l=70, r=60, t=20, b=40),
))
st.plotly_chart(fig, width="stretch")

st.caption(
    "Random and targeted attacks are trivial everywhere (F1 ≈ 1.0). The "
    "interesting variation is in the middle — and it flips: the hardest "
    "class in one domain is among the easiest in another."
)

st.divider()

# ── Method note ──────────────────────────────────────────────────────
with st.expander("How the datasets were built (same pipeline, three physics)"):
    st.markdown(
        """
        | | Water (Modena) | Power (IEEE-118) | Traffic (200-sensor) |
        |---|---|---|---|
        | Node | junction | bus | loop detector |
        | Edge | pipe | line / transformer | road segment |
        | Node signal | pressure (m) | voltage (pu) | speed (km/h) |
        | Edge signal | flow | active power | volume |
        | Dynamics | EPANET hydraulics | pandapower power-flow | rush-hour + spatial diffusion |
        | Conservation | mass | Kirchhoff (KCL) | flow at junctions |

        Every dataset is wrapped in the **identical** `Snapshot` format and
        run through the **same** five-attack corruption pipeline
        (`corrupt_all_snapshots`). The temporal-MoE model and training
        script are unchanged across all three — the only new code is a
        per-domain data adapter (`scripts/generate_power_grid.py`,
        `scripts/generate_traffic.py`).
        """
    )
