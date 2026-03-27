"""Page 4: Model Comparison."""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_architecture_comparison, load_baseline_comparison, load_test_results
from utils.theme import GLOBAL_CSS, plotly_layout, BLUE, GREEN, ORANGE, RED, CYAN, PURPLE, DIM

st.set_page_config(page_title="Model Comparison", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

st.title("Model Comparison")
st.caption("Benchmarking 5 GNN architectures against traditional analytical methods")

arch_data = load_architecture_comparison()
baseline_data = load_baseline_comparison()
test_results = load_test_results()

# ──────────────────────────────────────────────
# Section 1: GNN vs Baselines — the main story
# ──────────────────────────────────────────────
st.markdown("##### GNN vs Analytical Baselines")

baselines = baseline_data["baselines"]
gnn_mae = test_results["reconstruction"]["pressure_unobs"]["mae"]

methods = ["GNN\n(GraphSAGE)", "WLS", "Pseudo-inverse", "Mean\nImputation"]
mae_values = [
    gnn_mae,
    baselines["WLS"]["pressure_unobserved"]["mae"],
    baselines["Pseudo-inverse"]["pressure_unobserved"]["mae"],
    baselines["Mean imputation"]["pressure_unobserved"]["mae"],
]

col_chart, col_factors = st.columns([3, 1])

with col_chart:
    bar_colors = [GREEN, RED, RED, RED]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=methods, y=mae_values,
        marker=dict(
            color=bar_colors,
            line=dict(width=0),
        ),
        text=[f"<b>{v:.1f}</b>" for v in mae_values],
        textposition="outside",
        textfont=dict(size=14),
        hovertemplate="%{x}: %{y:.2f} m<extra></extra>",
        width=0.55,
    ))
    fig.update_layout(**plotly_layout(
        title=dict(text="Pressure MAE on Missing Sensors (lower = better)"),
        yaxis_title="MAE (m)",
        yaxis=dict(range=[0, max(mae_values) * 1.18]),
        height=460,
        xaxis=dict(tickfont=dict(size=12)),
    ))
    # Annotation highlighting the gap
    fig.add_annotation(
        x="GNN\n(GraphSAGE)", y=gnn_mae + 2.5,
        text=f"<b>{mae_values[1]/gnn_mae:.0f}x</b> better than WLS",
        showarrow=True, arrowhead=0, arrowcolor=GREEN, ay=-30,
        font=dict(size=13, color=GREEN),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_factors:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### Improvement Factors")
    for method, mae in zip(["WLS", "Pseudo-inverse", "Mean Imputation"], mae_values[1:]):
        factor = mae / gnn_mae
        st.metric(f"vs {method}", f"{factor:.1f}x")

st.divider()

# ──────────────────────────────────────────────
# Section 2: Architecture comparison
# ──────────────────────────────────────────────
st.markdown("##### GNN Architecture Benchmark")
st.markdown(
    f"<span style='opacity:0.5; font-size:0.85rem;'>"
    f"All architectures trained on the same data (50% missing, no attacks) for 100 epochs with early stopping.</span>",
    unsafe_allow_html=True,
)

arch_names = list(arch_data.keys())
p_unobs_mae = [arch_data[a]["p_unobs_mae"] for a in arch_names]
n_params = [arch_data[a]["n_params"] for a in arch_names]
train_time = [arch_data[a]["train_time"] for a in arch_names]

best_idx = p_unobs_mae.index(min(p_unobs_mae))
arch_colors = [BLUE] * len(arch_names)
arch_colors[best_idx] = GREEN

col_arch, col_table = st.columns([3, 2])

with col_arch:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=arch_names, y=p_unobs_mae,
        marker=dict(color=arch_colors, line=dict(width=0)),
        text=[f"{v:.2f}" for v in p_unobs_mae],
        textposition="outside", textfont=dict(size=12),
        hovertemplate="%{x}<br>MAE: %{y:.3f} m<extra></extra>",
        width=0.5,
    ))
    fig.update_layout(**plotly_layout(
        title=dict(text="Pressure MAE on Unobserved Nodes"),
        yaxis_title="MAE (m)",
        yaxis=dict(range=[0, max(p_unobs_mae) * 1.25]),
        height=400,
    ))
    st.plotly_chart(fig, use_container_width=True)

with col_table:
    df = pd.DataFrame({
        "Architecture": arch_names,
        "MAE (unobs)": [f"{v:.3f} m" for v in p_unobs_mae],
        "MAE (all)": [f"{arch_data[a]['p_all_mae']:.3f} m" for a in arch_names],
        "Params": [f"{v:,}" for v in n_params],
        "Train (s)": [f"{v:.0f}" for v in train_time],
        "Best Epoch": [arch_data[a]["best_epoch"] for a in arch_names],
    })
    st.dataframe(df, use_container_width=True, hide_index=True, height=250)
    st.success(f"Best: **{arch_names[best_idx]}** — MAE = {p_unobs_mae[best_idx]:.3f} m with only {n_params[best_idx]:,} parameters")

st.divider()

# ──────────────────────────────────────────────
# Section 3: Missing data robustness
# ──────────────────────────────────────────────
st.markdown("##### Robustness to Missing Data")

comp = baseline_data["comparison"]
rates = ["30%", "50%"]

gnn_v = [comp[r]["ReconGNN (GAT)"]["P_MAE_unobs"] for r in rates]
wls_v = [comp[r]["WLS"]["P_MAE_unobs"] for r in rates]
pi_v = [comp[r]["Pseudo-inverse"]["P_MAE_unobs"] for r in rates]

fig = go.Figure()
fig.add_trace(go.Bar(
    name="GNN (GAT)", x=rates, y=gnn_v, marker_color=GREEN,
    text=[f"{v:.2f}" for v in gnn_v], textposition="outside", textfont=dict(size=12),
    width=0.2,
))
fig.add_trace(go.Bar(
    name="WLS", x=rates, y=wls_v, marker_color=RED,
    text=[f"{v:.1f}" for v in wls_v], textposition="outside", textfont=dict(size=12),
    width=0.2,
))
fig.add_trace(go.Bar(
    name="Pseudo-inverse", x=rates, y=pi_v, marker_color=ORANGE,
    text=[f"{v:.1f}" for v in pi_v], textposition="outside", textfont=dict(size=12),
    width=0.2,
))
fig.update_layout(**plotly_layout(
    title=dict(text="How methods degrade as more sensors go offline"),
    xaxis_title="Missing Data Rate", yaxis_title="MAE (m)",
    barmode="group", height=420,
    yaxis=dict(range=[0, max(pi_v) * 1.18]),
    legend=dict(orientation="h", x=0.5, xanchor="center", y=1.08),
))
st.plotly_chart(fig, use_container_width=True)

degradation = ((gnn_v[1] / gnn_v[0]) - 1) * 100
st.info(
    f"GNN degrades gracefully: {gnn_v[0]:.2f} to {gnn_v[1]:.2f} MAE "
    f"(+{degradation:.0f}%), while baselines remain at 18-22 MAE regardless of missing rate."
)
