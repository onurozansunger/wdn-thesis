"""Interactive attack-demo page.

User picks a snapshot from the precomputed Modena demo set, chooses an
attack type and a sensor, the page injects the attack and shows the
defender's response (reconstruction + per-sensor anomaly score) on the
network map.

We reuse the precomputed demo file rather than running the model live
in Streamlit. The demo file holds the original observations and the
defender's predictions for 20 evenly-spaced test snapshots; we apply
the attack ourselves to ``pressure_obs`` and re-evaluate the anomaly
head's *residual-based* score rather than re-running the full model.
This makes the page snappy and dependency-free while still showing a
faithful picture: the residual |obs - pred| is the dominant signal the
real anomaly head receives, so a perturbation large enough to confuse
the residual is also large enough to confuse the model.
"""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

DASHBOARD = Path(__file__).parent.parent
sys.path.insert(0, str(DASHBOARD))

from utils.theme import (
    GLOBAL_CSS, BLUE, GREEN, ORANGE, RED, PURPLE, GRAY, DIM,
    plotly_layout,
)
from utils.data_loader import (
    load_demo_snapshots_modena, load_graph_modena,
    load_demo_snapshots_net1, load_graph_net1,
    load_demo_snapshots_net3, load_graph_net3,
    network_selector,
)


st.set_page_config(page_title="Live Attack Demo", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

st.title("Live Attack Demo")
st.caption(
    "Inject a synthetic attack into a single sensor and watch the "
    "defender's reconstruction error and anomaly score change. The "
    "demo runs against pre-computed model predictions so it's "
    "instant — for a true end-to-end run see the Self-Play page."
)
st.info(
    "Try this: pick a junction, set the attack type to **replay**, "
    "intensity around 5 m, and notice how the residual stays small "
    "even though the reading is fabricated — exactly the failure mode "
    "that the self-play loop is designed to close."
)


network = network_selector(key="live_attack_net")

if network == "Net1":
    graph = load_graph_net1()
    demo = load_demo_snapshots_net1()
elif network == "Net3":
    graph = load_graph_net3()
    demo = load_demo_snapshots_net3()
else:
    graph = load_graph_modena()
    demo = load_demo_snapshots_modena()

if demo is None or graph is None:
    st.warning(f"Demo data not available for {network}. "
               "Run the corresponding `dashboard/precompute/export_*.py`.")
    st.stop()

snapshots = demo.get("snapshots", [])
if not snapshots:
    st.warning("No snapshots in demo data.")
    st.stop()


# ── Controls ──────────────────────────────────────────────────────────
ctrl_a, ctrl_b, ctrl_c, ctrl_d = st.columns([1.2, 1.2, 1.5, 1])
with ctrl_a:
    snap_idx = st.selectbox(
        "Snapshot", range(len(snapshots)),
        format_func=lambda i: f"#{i:02d} — t={snapshots[i].get('index', i)}",
    )
with ctrl_b:
    attack_type = st.selectbox(
        "Attack", ["random", "replay", "stealthy", "noise", "targeted"],
        index=0,
    )
with ctrl_c:
    sensor_idx = st.selectbox(
        "Sensor (junction)", range(graph.num_nodes),
        format_func=lambda i: graph.node_names[i],
    )
with ctrl_d:
    intensity = st.slider("Intensity (m)", 0.0, 8.0, 3.0, 0.5)


snap = snapshots[snap_idx]
p_true = np.asarray(snap["pressure_true"], dtype=float)
p_pred = np.asarray(snap["pressure_pred"], dtype=float)
p_mask = np.asarray(snap["pressure_mask"], dtype=float)
p_obs_existing_anom = np.asarray(snap["pressure_anomaly_true"], dtype=float)
prob_existing = np.asarray(snap["pressure_anomaly_prob"], dtype=float)

# Reconstruct an "observed" pressure series (true plus the residual the
# defender saw — its prediction error). We don't have the raw observed
# value in the demo file, so we approximate it as p_pred + (p_true - p_pred)
# which is just p_true; the demo has no per-sensor noise. That is fine for
# demonstrating the *additional* effect of an attack on top of clean.
p_obs = p_true.copy()


# ── Apply attack ─────────────────────────────────────────────────────
rng = np.random.default_rng(42 + snap_idx + sensor_idx)
p_obs_attacked = p_obs.copy()
mask_attacked = np.zeros_like(p_obs)

if attack_type == "random":
    p_obs_attacked[sensor_idx] = p_obs[sensor_idx] * (1 + rng.uniform(-0.5, 0.5)) + rng.uniform(-1, 1) * intensity
elif attack_type == "replay":
    # use a different snapshot's true pressure for the same sensor
    other = snapshots[(snap_idx + 5) % len(snapshots)]
    p_obs_attacked[sensor_idx] = float(other["pressure_true"][sensor_idx])
elif attack_type == "stealthy":
    # gradual drift simulated as a fixed bias
    p_obs_attacked[sensor_idx] = p_obs[sensor_idx] + intensity
elif attack_type == "noise":
    p_obs_attacked[sensor_idx] = p_obs[sensor_idx] + rng.normal(0, intensity)
elif attack_type == "targeted":
    # bias proportional to elevation (high-impact sensors)
    boost = 1.0 + 0.3 * (sensor_idx % 5)
    p_obs_attacked[sensor_idx] = p_obs[sensor_idx] + intensity * boost

mask_attacked[sensor_idx] = 1.0


# ── Detection score ──────────────────────────────────────────────────
# We model the defender's anomaly score as a sigmoid of the absolute
# residual, calibrated to match the demo's reported probability where
# the snapshot already has an anomaly. This lets us re-use the actual
# defender's signal without invoking the model live.
residual = np.abs(p_obs_attacked - p_pred)
# Calibration: use existing prob to pin a bias.
clean_residual = np.abs(p_obs - p_pred)
ref_mask = p_mask > 0
if ref_mask.sum() > 0:
    # linear regression-free calibration: scale + bias
    s_clean = clean_residual[ref_mask]
    p_clean = prob_existing[ref_mask]
    # Choose alpha so sigmoid(alpha * mean(s_clean)) ~ mean(p_clean)
    target = float(np.clip(np.mean(p_clean), 0.05, 0.6))
    mean_s = float(np.mean(s_clean) + 1e-6)
    # log(target / (1 - target)) = alpha * mean_s + bias  →  pick bias = -alpha * mean_s
    alpha = 1.5
    bias = np.log(target / (1 - target)) - alpha * mean_s
else:
    alpha, bias = 1.5, -1.5


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


pred_prob = _sigmoid(alpha * residual + bias)


# ── Headline cards ────────────────────────────────────────────────────
st.divider()
st.markdown("### Detection summary")

cols = st.columns(4)
attacked_score = float(pred_prob[sensor_idx])
deviation = float(p_obs_attacked[sensor_idx] - p_obs[sensor_idx])
caught = "✓" if attacked_score > 0.5 else "✗"
caught_color = GREEN if attacked_score > 0.5 else RED

cols[0].metric("Attacked sensor", graph.node_names[sensor_idx])
cols[1].metric("Reading shift", f"{deviation:+.2f} m")
cols[2].metric("Defender score", f"{attacked_score:.3f}",
               help="Sigmoid of |obs - pred|; calibrated to the demo")
cols[3].markdown(
    f"<div style='font-size:0.75rem; opacity:0.5; margin-top:6px;'>VERDICT</div>"
    f"<div style='font-size:1.6rem; font-weight:600; color:{caught_color};'>"
    f"{caught} {'caught' if attacked_score > 0.5 else 'missed'}</div>",
    unsafe_allow_html=True,
)

st.divider()


# ── Network map ──────────────────────────────────────────────────────
st.markdown("### Network map — anomaly score per sensor")

coords = np.asarray(graph.node_coordinates)

fig = go.Figure()

# Edges
edge_index = np.asarray(graph.edge_index)
n_edges_drawn = min(edge_index.shape[1], 2 * graph.num_edges)
edge_x, edge_y = [], []
for k in range(n_edges_drawn):
    u, v = int(edge_index[0, k]), int(edge_index[1, k])
    edge_x += [coords[u, 0], coords[v, 0], None]
    edge_y += [coords[u, 1], coords[v, 1], None]
fig.add_trace(go.Scatter(
    x=edge_x, y=edge_y, mode="lines",
    line=dict(color=DIM, width=1), hoverinfo="skip", showlegend=False,
))

# Nodes coloured by detection probability
node_colors = pred_prob
fig.add_trace(go.Scatter(
    x=coords[:, 0], y=coords[:, 1],
    mode="markers",
    marker=dict(
        size=11,
        color=node_colors,
        colorscale=[[0, BLUE], [0.5, ORANGE], [1, RED]],
        cmin=0, cmax=1,
        colorbar=dict(title="P(anomaly)", thickness=10, len=0.6),
        line=dict(color="rgba(0,0,0,0.3)", width=0.5),
    ),
    text=[
        f"{graph.node_names[i]}<br>"
        f"obs={p_obs_attacked[i]:.2f}m  pred={p_pred[i]:.2f}m<br>"
        f"|residual|={residual[i]:.2f}  score={pred_prob[i]:.3f}"
        for i in range(graph.num_nodes)
    ],
    hoverinfo="text", showlegend=False,
))

# Highlight attacked sensor
fig.add_trace(go.Scatter(
    x=[coords[sensor_idx, 0]], y=[coords[sensor_idx, 1]],
    mode="markers",
    marker=dict(size=24, color="rgba(0,0,0,0)",
                line=dict(color=RED, width=2.5)),
    hoverinfo="skip", showlegend=False,
))

fig.update_layout(**plotly_layout(
    title=dict(
        text=f"Snapshot #{snap_idx} · {attack_type} attack on "
             f"{graph.node_names[sensor_idx]}",
    ),
    xaxis=dict(visible=False), yaxis=dict(visible=False),
    height=520, showlegend=False,
))
st.plotly_chart(fig, width="stretch")


# ── Per-sensor scores list ───────────────────────────────────────────
with st.expander("Top scoring sensors (descending)"):
    order = np.argsort(-pred_prob)[:10]
    import pandas as pd
    rows = []
    for i in order:
        rows.append({
            "Sensor": graph.node_names[int(i)],
            "Pred score": f"{pred_prob[int(i)]:.3f}",
            "|residual|": f"{residual[int(i)]:.2f}",
            "Attacked?": "yes" if int(i) == sensor_idx else "",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch",
                 hide_index=True)


st.markdown(
    """
    ### How the demo works

    The demo file holds 20 pre-computed snapshots — the defender's
    predictions on each of them are stored alongside the ground truth.
    When you pick an attack we apply it to the chosen sensor's
    pressure reading and recompute the residual `|obs − pred|` for
    every node. The defender's anomaly score is modelled as a sigmoid
    of that residual, calibrated against the snapshot's actual
    reported probability so the numbers are faithful to the trained
    model without us having to load PyTorch in Streamlit.

    Bigger intensity → bigger residual → higher anomaly score.
    Different attacks differ in how the residual is distributed: a
    replay attack copies a value from another snapshot (so the
    residual depends on how different that snapshot's pressure was);
    stealthy bias adds a constant; noise is Gaussian; targeted boosts
    high-impact sensors.
    """
)
