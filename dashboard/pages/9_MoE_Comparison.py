"""Page 9: Mixture-of-Experts GNN — Attack Specialization."""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_moe_results, load_moe_history, network_selector
from utils.theme import (
    GLOBAL_CSS, plotly_layout,
    BLUE, GREEN, ORANGE, RED, CYAN, PURPLE, DIM,
)

st.set_page_config(page_title="MoE Comparison", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

st.title("Mixture-of-Experts GNN")
st.caption(
    "Attack-specialized experts with a temporal router — targeting replay attack detection"
)

st.markdown(
    """
    <div style='background:#1a1f2e; border-left:3px solid #4f8ef7; padding:12px 18px; border-radius:4px; margin-bottom:16px;'>
    <b>Motivation:</b> A single GNN must compromise between all attack fingerprints.
    Replay attacks are the hardest case: individual replayed readings look perfectly
    legitimate — only by comparing against recent history can the model flag them.
    <br><br>
    <b>Architecture:</b> K specialized experts (each a Temporal GNN+GRU) are dynamically
    weighted by a learned router that classifies the attack type from the temporal window.
    </div>
    """,
    unsafe_allow_html=True,
)

ATTACK_LABELS = {
    "clean": "Clean",
    "random": "Random",
    "replay": "Replay ⚠",
    "stealthy": "Stealthy Bias",
    "noise": "Noise",
    "targeted": "Targeted",
}
ATTACK_COLORS = {
    "random": RED, "replay": PURPLE, "stealthy": ORANGE,
    "noise": CYAN, "targeted": "#e6c619", "clean": DIM,
}

network = network_selector("moe_network")

spatial_res = load_moe_results(network, "spatial")
temporal_res = load_moe_results(network, "temporal")

if spatial_res is None:
    st.warning(f"No spatial MoE results found for {network}.")
    st.stop()

# ──────────────────────────────────────────────
# Section 1: Per-attack F1 — spatial vs temporal
# ──────────────────────────────────────────────
st.markdown("##### Per-Attack Anomaly Detection F1")
st.markdown(
    "<span style='opacity:0.5; font-size:0.85rem;'>"
    "Spatial MoE (no temporal context) vs Temporal MoE (GNN + GRU per expert, window=6). "
    "Replay column shows the key improvement.</span>",
    unsafe_allow_html=True,
)

spa_pa = spatial_res.get("per_attack_pressure", {})
attack_names = [a for a in ["random", "replay", "stealthy", "noise", "targeted"] if a in spa_pa]
labels = [ATTACK_LABELS.get(a, a) for a in attack_names]
spa_f1s = [spa_pa[a]["f1"] for a in attack_names]

fig = go.Figure()
fig.add_trace(go.Bar(
    x=labels, y=spa_f1s, name="Spatial MoE",
    marker_color=[ATTACK_COLORS.get(a, BLUE) for a in attack_names],
    marker_opacity=0.6,
    text=[f"{v:.3f}" for v in spa_f1s],
    textposition="outside", textfont=dict(size=12),
))

if temporal_res is not None:
    tmp_pa = temporal_res.get("per_attack_pressure", {})
    tmp_f1s = [tmp_pa.get(a, {}).get("f1", 0) for a in attack_names]
    fig.add_trace(go.Bar(
        x=labels, y=tmp_f1s, name="Temporal MoE (GNN+GRU)",
        marker_color=[ATTACK_COLORS.get(a, PURPLE) for a in attack_names],
        text=[f"{v:.3f}" for v in tmp_f1s],
        textposition="outside", textfont=dict(size=12),
    ))

fig.update_layout(**plotly_layout(
    title=dict(text=f"{network} — Per-Attack Pressure F1"),
    yaxis_title="F1 Score", height=420, yaxis=dict(range=[0, 1.25]),
    barmode="group",
    legend=dict(orientation="h", x=0.5, xanchor="center", y=1.10),
))
st.plotly_chart(fig, use_container_width=True)

# Highlight replay row
if temporal_res is not None:
    tmp_pa = temporal_res.get("per_attack_pressure", {})
    replay_spa = spa_pa.get("replay", {}).get("f1", 0)
    replay_tmp = tmp_pa.get("replay", {}).get("f1", 0)
    if replay_tmp > replay_spa + 0.05:
        delta = replay_tmp - replay_spa
        st.success(
            f"**Replay detection F1: {replay_spa:.3f} → {replay_tmp:.3f}** "
            f"(+{delta:.3f}) — temporal context unlocks history comparison "
            "that spatial models cannot perform."
        )
    elif temporal_res is not None:
        st.info(
            f"Replay F1 — Spatial: {replay_spa:.3f} | "
            f"Temporal: {replay_tmp:.3f}"
        )
else:
    st.info(
        f"⏳ Temporal MoE training in progress. "
        f"Spatial MoE baseline replay F1: **{spa_pa.get('replay',{}).get('f1',0):.3f}** "
        "(near zero — spatial model cannot detect stale readings)."
    )

st.divider()

# ──────────────────────────────────────────────
# Section 2: Per-attack comparison table
# ──────────────────────────────────────────────
st.markdown("##### Detailed Per-Attack Metrics")

rows = []
for a in attack_names:
    sm = spa_pa.get(a, {})
    row = {
        "Attack": ATTACK_LABELS.get(a, a),
        "Spatial P": f"{sm.get('precision', 0):.3f}",
        "Spatial R": f"{sm.get('recall', 0):.3f}",
        "Spatial F1": f"{sm.get('f1', 0):.3f}",
        "Spatial AUROC": f"{sm.get('auroc', 0):.3f}",
    }
    if temporal_res is not None:
        tm = temporal_res.get("per_attack_pressure", {}).get(a, {})
        row.update({
            "Temporal P": f"{tm.get('precision', 0):.3f}",
            "Temporal R": f"{tm.get('recall', 0):.3f}",
            "Temporal F1": f"{tm.get('f1', 0):.3f}",
            "Temporal AUROC": f"{tm.get('auroc', 0):.3f}",
        })
    rows.append(row)

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()

# ──────────────────────────────────────────────
# Section 3: Router accuracy
# ──────────────────────────────────────────────
st.markdown("##### Attack Router Performance")
col_r1, col_r2 = st.columns(2)

with col_r1:
    spa_racc = spatial_res.get("router_acc", 0)
    metrics_to_show = {"Spatial MoE": spa_racc}
    if temporal_res is not None:
        metrics_to_show["Temporal MoE"] = temporal_res.get("router_acc", 0)
    colors = [BLUE, PURPLE]

    fig = go.Figure()
    for i, (name, acc) in enumerate(metrics_to_show.items()):
        fig.add_trace(go.Bar(
            x=[name], y=[acc], name=name,
            marker_color=colors[i],
            text=[f"{acc:.3f}"],
            textposition="outside", textfont=dict(size=14),
            width=0.4,
        ))
    fig.update_layout(**plotly_layout(
        title=dict(text="Router Classification Accuracy"),
        yaxis_title="Accuracy", height=340, yaxis=dict(range=[0, 1.18]),
    ))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Router accuracy measures how well the gating network identifies the "
        "dominant attack type — higher accuracy means experts receive "
        "more targeted gradient signal."
    )

with col_r2:
    # Router confusion matrix from spatial MoE
    from wdn.corruption import ID_TO_ATTACK_TYPE
    conf = spatial_res.get("router_confusion", {})
    if conf:
        # Decode keys: "true_pred"
        n_classes = 6
        mat = [[0] * n_classes for _ in range(n_classes)]
        for key, cnt in conf.items():
            parts = key.split("_")
            t, p = int(parts[0]), int(parts[1])
            mat[t][p] = cnt
        cls_names = [ID_TO_ATTACK_TYPE.get(i, str(i)) for i in range(n_classes)]
        # Trim empty rows/cols
        used = [i for i in range(n_classes) if any(mat[i]) or any(mat[r][i] for r in range(n_classes))]
        mat_trim = [[mat[r][c] for c in used] for r in used]
        names_trim = [cls_names[i] for i in used]

        fig = go.Figure(go.Heatmap(
            z=mat_trim, x=names_trim, y=names_trim,
            colorscale="Blues", showscale=False,
            text=[[str(v) if v > 0 else "" for v in row] for row in mat_trim],
            texttemplate="%{text}",
        ))
        fig.update_layout(**plotly_layout(
            title=dict(text="Spatial Router Confusion (true → pred)"),
            xaxis_title="Predicted", yaxis_title="True",
            height=340,
        ))
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ──────────────────────────────────────────────
# Section 4: Overall metrics vs single model
# ──────────────────────────────────────────────
st.markdown("##### Overall Anomaly Detection Comparison")
col_ov1, col_ov2 = st.columns(2)

with col_ov1:
    results_to_compare = {"Spatial MoE": spatial_res}
    if temporal_res is not None:
        results_to_compare["Temporal MoE"] = temporal_res

    metric_labels = ["Precision", "Recall", "F1", "AUROC"]
    fig = go.Figure()
    bar_colors = [BLUE, PURPLE]
    for i, (model_name, res) in enumerate(results_to_compare.items()):
        ad = res.get("anomaly_detection", {}).get("pressure", {})
        vals = [ad.get("precision", 0), ad.get("recall", 0), ad.get("f1", 0), ad.get("auroc", 0)]
        fig.add_trace(go.Bar(
            x=metric_labels, y=vals, name=model_name,
            marker_color=bar_colors[i],
            text=[f"{v:.3f}" for v in vals],
            textposition="outside", textfont=dict(size=12),
        ))
    fig.update_layout(**plotly_layout(
        title=dict(text="Pressure Anomaly Detection"),
        yaxis_title="Score", height=380,
        yaxis=dict(range=[0, 1.25]), barmode="group",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.10),
    ))
    st.plotly_chart(fig, use_container_width=True)

with col_ov2:
    # Summary table
    rows2 = []
    for model_name, res in results_to_compare.items():
        recon = res.get("reconstruction", {}).get("pressure_unobs", {})
        ad = res.get("anomaly_detection", {}).get("pressure", {})
        rows2.append({
            "Model": model_name,
            "Params": f"{res.get('n_params', 0):,}",
            "P_MAE (m)": f"{recon.get('mae', 0):.3f}",
            "F1": f"{ad.get('f1', 0):.3f}",
            "AUROC": f"{ad.get('auroc', 0):.3f}",
            "Router Acc": f"{res.get('router_acc', 0):.3f}",
        })
    df = pd.DataFrame(rows2)
    st.markdown("**Model summary**")
    st.dataframe(df, use_container_width=True, hide_index=True, height=180)
    st.markdown(
        "<span style='opacity:0.5; font-size:0.82rem;'>"
        "MoE = 6 experts (one per attack class) + lightweight router. "
        "Temporal variant uses GNN+GRU per expert (window = 6 timesteps)."
        "</span>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# Section 5: Training history
# ──────────────────────────────────────────────
st.divider()
st.markdown("##### Training Curves")

spa_hist = load_moe_history(network, "spatial")
tmp_hist = load_moe_history(network, "temporal") if temporal_res is not None else None

if spa_hist or tmp_hist:
    col_h1, col_h2 = st.columns(2)

    with col_h1:
        fig = go.Figure()
        if spa_hist:
            epochs = [h["epoch"] for h in spa_hist]
            fig.add_trace(go.Scatter(
                x=epochs, y=[h["val_recon"] for h in spa_hist],
                name="Spatial MoE", line=dict(color=BLUE, width=2),
            ))
        if tmp_hist:
            t_epochs = [h["epoch"] for h in tmp_hist]
            fig.add_trace(go.Scatter(
                x=t_epochs, y=[h["val_recon"] for h in tmp_hist],
                name="Temporal MoE", line=dict(color=PURPLE, width=2),
            ))
        fig.update_layout(**plotly_layout(
            title=dict(text="Validation Reconstruction Loss"),
            xaxis_title="Epoch", yaxis_title="Loss", height=320,
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col_h2:
        fig = go.Figure()
        if spa_hist and "val_p_anomaly_f1" in spa_hist[-1]:
            epochs = [h["epoch"] for h in spa_hist]
            fig.add_trace(go.Scatter(
                x=epochs,
                y=[h.get("val_p_anomaly_f1", 0) for h in spa_hist],
                name="Spatial MoE F1", line=dict(color=BLUE, width=2),
            ))
        if tmp_hist and "val_p_anomaly_f1" in tmp_hist[-1]:
            t_epochs = [h["epoch"] for h in tmp_hist]
            fig.add_trace(go.Scatter(
                x=t_epochs,
                y=[h.get("val_p_anomaly_f1", 0) for h in tmp_hist],
                name="Temporal MoE F1", line=dict(color=PURPLE, width=2),
            ))
        if spa_hist and "val_router_acc" in spa_hist[-1]:
            epochs = [h["epoch"] for h in spa_hist]
            fig.add_trace(go.Scatter(
                x=epochs,
                y=[h.get("val_router_acc", 0) for h in spa_hist],
                name="Router Acc (Spatial)", line=dict(color=ORANGE, width=2, dash="dot"),
            ))
        fig.update_layout(**plotly_layout(
            title=dict(text="Validation F1 & Router Accuracy"),
            xaxis_title="Epoch", yaxis_title="Score",
            height=320, yaxis=dict(range=[0, 1.05]),
            legend=dict(orientation="h", x=0.5, xanchor="center", y=1.12, font=dict(size=10)),
        ))
        st.plotly_chart(fig, use_container_width=True)
