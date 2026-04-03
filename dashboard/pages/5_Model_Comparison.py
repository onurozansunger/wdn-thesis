"""Page 5: Model Comparison."""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import (
    load_architecture_comparison, load_baseline_comparison,
    load_test_results_net1, load_test_results_modena,
)
from utils.theme import GLOBAL_CSS, plotly_layout, BLUE, GREEN, ORANGE, RED, CYAN, PURPLE, DIM

st.set_page_config(page_title="Model Comparison", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

st.title("Model Comparison")
st.caption("Benchmarking GNN performance across networks and against analytical baselines")

net1_results = load_test_results_net1()
modena_results = load_test_results_modena()

# ──────────────────────────────────────────────
# Section 1: Net1 vs Modena — scalability story
# ──────────────────────────────────────────────
st.markdown("##### Cross-Network Performance (same model, same hyperparameters)")

col_recon, col_anom = st.columns(2)

with col_recon:
    networks = ["Net1\n(11 nodes)", "Modena\n(272 nodes)"]
    net1_mae = net1_results["reconstruction"]["pressure_unobs"]["mae"]
    mod_mae = modena_results["reconstruction"]["pressure_unobs"]["mae"]
    net1_rmse = net1_results["reconstruction"]["pressure_unobs"]["rmse"]
    mod_rmse = modena_results["reconstruction"]["pressure_unobs"]["rmse"]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=networks, y=[net1_mae, mod_mae], name="MAE (m)", marker_color=BLUE,
                         text=[f"{net1_mae:.3f}", f"{mod_mae:.3f}"],
                         textposition="outside", textfont=dict(size=13)))
    fig.add_trace(go.Bar(x=networks, y=[net1_rmse, mod_rmse], name="RMSE (m)", marker_color=CYAN,
                         text=[f"{net1_rmse:.3f}", f"{mod_rmse:.3f}"],
                         textposition="outside", textfont=dict(size=13)))
    fig.update_layout(**plotly_layout(
        title=dict(text="Pressure Reconstruction (Unobserved)"),
        yaxis_title="Error (m)", height=400, barmode="group",
        legend=dict(x=0.7, y=0.95),
    ))
    st.plotly_chart(fig, use_container_width=True)

with col_anom:
    net1_a = net1_results["anomaly_detection"]["pressure"]
    mod_a = modena_results["anomaly_detection"]["pressure"]
    metrics_names = ["Precision", "Recall", "F1", "AUROC"]
    net1_vals = [net1_a["precision"], net1_a["recall"], net1_a["f1"], net1_a["auroc"]]
    mod_vals = [mod_a["precision"], mod_a["recall"], mod_a["f1"], mod_a["auroc"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=metrics_names, y=net1_vals, name="Net1", marker_color=BLUE,
                         text=[f"{v:.3f}" for v in net1_vals], textposition="outside", textfont=dict(size=13)))
    fig.add_trace(go.Bar(x=metrics_names, y=mod_vals, name="Modena", marker_color=GREEN,
                         text=[f"{v:.3f}" for v in mod_vals], textposition="outside", textfont=dict(size=13)))
    fig.update_layout(**plotly_layout(
        title=dict(text="Anomaly Detection (Pressure)"),
        yaxis_title="Score", height=400, yaxis=dict(range=[0, 1.15]),
        barmode="group", legend=dict(x=0.02, y=0.95),
    ))
    st.plotly_chart(fig, use_container_width=True)

improvement = net1_mae / mod_mae
st.info(
    f"Modena achieves **{mod_mae:.3f}m** MAE vs Net1's **{net1_mae:.3f}m** — "
    f"**{improvement:.1f}x better** on the larger network. "
    f"Denser graph topology provides more spatial context for the GNN."
)

st.divider()

# ──────────────────────────────────────────────
# Section 2: GNN vs Baselines (Net1)
# ──────────────────────────────────────────────
st.markdown("##### GNN vs Analytical Baselines (Net1)")

baseline_data = load_baseline_comparison()
baselines = baseline_data["baselines"]
gnn_mae = net1_results["reconstruction"]["pressure_unobs"]["mae"]

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
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"<b>{v:.1f}</b>" for v in mae_values],
        textposition="outside", textfont=dict(size=14),
        hovertemplate="%{x}: %{y:.2f} m<extra></extra>", width=0.55,
    ))
    fig.update_layout(**plotly_layout(
        title=dict(text="Pressure MAE on Missing Sensors (lower = better)"),
        yaxis_title="MAE (m)", yaxis=dict(range=[0, max(mae_values) * 1.18]),
        height=420, xaxis=dict(tickfont=dict(size=12)),
    ))
    fig.add_annotation(
        x="GNN\n(GraphSAGE)", y=gnn_mae + 2.5,
        text=f"<b>{mae_values[1]/gnn_mae:.0f}x</b> better than WLS",
        showarrow=True, arrowhead=0, arrowcolor=GREEN, ay=-30,
        font=dict(size=13, color=GREEN),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_factors:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### Improvement")
    for method, mae in zip(["WLS", "Pseudo-inverse", "Mean Imputation"], mae_values[1:]):
        st.metric(f"vs {method}", f"{mae / gnn_mae:.1f}x")

st.divider()

# ──────────────────────────────────────────────
# Section 3: Architecture comparison (Net1)
# ──────────────────────────────────────────────
st.markdown("##### GNN Architecture Benchmark (Net1)")
st.markdown("<span style='opacity:0.5; font-size:0.85rem;'>"
            "All architectures trained on the same data split for 100 epochs.</span>",
            unsafe_allow_html=True)

arch_data = load_architecture_comparison()
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
        width=0.5,
    ))
    fig.update_layout(**plotly_layout(
        title=dict(text="Pressure MAE on Unobserved Nodes"),
        yaxis_title="MAE (m)", yaxis=dict(range=[0, max(p_unobs_mae) * 1.25]),
        height=380,
    ))
    st.plotly_chart(fig, use_container_width=True)

with col_table:
    df = pd.DataFrame({
        "Architecture": arch_names,
        "MAE (unobs)": [f"{v:.3f} m" for v in p_unobs_mae],
        "Params": [f"{v:,}" for v in n_params],
        "Train (s)": [f"{v:.0f}" for v in train_time],
        "Best Epoch": [arch_data[a]["best_epoch"] for a in arch_names],
    })
    st.dataframe(df, use_container_width=True, hide_index=True, height=250)
    st.success(f"Best: **{arch_names[best_idx]}** — {p_unobs_mae[best_idx]:.3f} m")
