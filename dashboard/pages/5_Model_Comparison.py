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
    load_attack_analysis_net1, load_attack_analysis_modena,
    load_temporal_results, NETWORKS,
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
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.08),
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
        yaxis_title="Score", height=400, yaxis=dict(range=[0, 1.22]),
        barmode="group",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.08),
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
# Section 2: Spatial vs Spatio-Temporal (GNN vs GNN+GRU)
# ──────────────────────────────────────────────
temporal_net1 = load_temporal_results("Net1")
temporal_modena = load_temporal_results("Modena")

has_temporal = temporal_net1 or temporal_modena

if has_temporal:
    st.markdown("##### Spatial vs Spatio-Temporal Model")
    st.markdown("<span style='opacity:0.5; font-size:0.85rem;'>"
                "MultiTaskGNN (spatial only) vs TemporalMultiTaskGNN (GNN + GRU, window=6 timesteps)</span>",
                unsafe_allow_html=True)

    # Build comparison data for each available network
    temporal_pairs = []
    if temporal_net1:
        temporal_pairs.append(("Net1", net1_results, temporal_net1))
    if temporal_modena and modena_results:
        temporal_pairs.append(("Modena", modena_results, temporal_modena))

    for net_name, spatial_res, temporal_res in temporal_pairs:
        if len(temporal_pairs) > 1:
            st.markdown(f"**{net_name}** ({NETWORKS[net_name]['label']})")

        col_t_recon, col_t_anom = st.columns(2)

        spatial_recon = spatial_res["reconstruction"]["pressure_unobs"]
        temporal_recon = temporal_res["reconstruction"]["pressure_unobs"]
        spatial_anom = spatial_res["anomaly_detection"]["pressure"]
        temporal_anom = temporal_res["anomaly_detection"]["pressure"]

        with col_t_recon:
            models = ["MultiTaskGNN\n(Spatial)", "TemporalMultiTaskGNN\n(GNN + GRU)"]
            s_mae, t_mae = spatial_recon["mae"], temporal_recon["mae"]
            s_rmse, t_rmse = spatial_recon["rmse"], temporal_recon["rmse"]

            fig = go.Figure()
            fig.add_trace(go.Bar(x=models, y=[s_mae, t_mae], name="MAE (m)", marker_color=BLUE,
                                 text=[f"{s_mae:.3f}", f"{t_mae:.3f}"],
                                 textposition="outside", textfont=dict(size=13)))
            fig.add_trace(go.Bar(x=models, y=[s_rmse, t_rmse], name="RMSE (m)", marker_color=CYAN,
                                 text=[f"{s_rmse:.3f}", f"{t_rmse:.3f}"],
                                 textposition="outside", textfont=dict(size=13)))
            fig.update_layout(**plotly_layout(
                title=dict(text=f"{net_name} — Reconstruction (Unobserved)"),
                yaxis_title="Error (m)", height=400, barmode="group",
                legend=dict(orientation="h", x=0.5, xanchor="center", y=1.08),
            ))
            st.plotly_chart(fig, use_container_width=True)

        with col_t_anom:
            metrics_names = ["Precision", "Recall", "F1", "AUROC"]
            s_vals = [spatial_anom["precision"], spatial_anom["recall"], spatial_anom["f1"], spatial_anom["auroc"]]
            t_vals = [temporal_anom["precision"], temporal_anom["recall"], temporal_anom["f1"], temporal_anom["auroc"]]

            fig = go.Figure()
            fig.add_trace(go.Bar(x=metrics_names, y=s_vals, name="Spatial", marker_color=BLUE,
                                 text=[f"{v:.3f}" for v in s_vals], textposition="outside",
                                 textfont=dict(size=13)))
            fig.add_trace(go.Bar(x=metrics_names, y=t_vals, name="Temporal", marker_color=PURPLE,
                                 text=[f"{v:.3f}" for v in t_vals], textposition="outside",
                                 textfont=dict(size=13)))
            fig.update_layout(**plotly_layout(
                title=dict(text=f"{net_name} — Anomaly Detection (Pressure)"),
                yaxis_title="Score", height=400, yaxis=dict(range=[0, 1.22]),
                barmode="group",
                legend=dict(orientation="h", x=0.5, xanchor="center", y=1.08),
            ))
            st.plotly_chart(fig, use_container_width=True)

        recon_improvement = (s_mae - t_mae) / s_mae * 100
        st.info(
            f"**{net_name}**: Temporal context improves reconstruction by "
            f"**{recon_improvement:.0f}%** (MAE: {s_mae:.3f}m → {t_mae:.3f}m). "
            f"Anomaly AUROC: {temporal_anom['auroc']:.3f} (temporal) vs {spatial_anom['auroc']:.3f} (spatial)."
        )

    # Summary table with all models
    table_rows = []
    if temporal_net1:
        s_r = net1_results["reconstruction"]["pressure_unobs"]
        t_r = temporal_net1["reconstruction"]["pressure_unobs"]
        s_a = net1_results["anomaly_detection"]["pressure"]
        t_a = temporal_net1["anomaly_detection"]["pressure"]
        table_rows.extend([
            {"Network": "Net1", "Model": "MultiTaskGNN (Spatial)", "Parameters": f"{net1_results['n_params']:,}",
             "P_MAE (m)": f"{s_r['mae']:.3f}", "F1": f"{s_a['f1']:.3f}", "AUROC": f"{s_a['auroc']:.3f}"},
            {"Network": "Net1", "Model": "TemporalMultiTaskGNN", "Parameters": f"{temporal_net1['n_params']:,}",
             "P_MAE (m)": f"{t_r['mae']:.3f}", "F1": f"{t_a['f1']:.3f}", "AUROC": f"{t_a['auroc']:.3f}"},
        ])
    if temporal_modena and modena_results:
        s_r = modena_results["reconstruction"]["pressure_unobs"]
        t_r = temporal_modena["reconstruction"]["pressure_unobs"]
        s_a = modena_results["anomaly_detection"]["pressure"]
        t_a = temporal_modena["anomaly_detection"]["pressure"]
        table_rows.extend([
            {"Network": "Modena", "Model": "MultiTaskGNN (Spatial)", "Parameters": f"{modena_results['n_params']:,}",
             "P_MAE (m)": f"{s_r['mae']:.3f}", "F1": f"{s_a['f1']:.3f}", "AUROC": f"{s_a['auroc']:.3f}"},
            {"Network": "Modena", "Model": "TemporalMultiTaskGNN", "Parameters": f"{temporal_modena['n_params']:,}",
             "P_MAE (m)": f"{t_r['mae']:.3f}", "F1": f"{t_a['f1']:.3f}", "AUROC": f"{t_a['auroc']:.3f}"},
        ])

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    st.divider()

# ──────────────────────────────────────────────
# Section 3: Attack Detection Comparison (Net1 vs Modena)
# ──────────────────────────────────────────────
st.markdown("##### Attack Detection: Net1 vs Modena")

net1_attack = load_attack_analysis_net1()
modena_attack = load_attack_analysis_modena()

if net1_attack and modena_attack:
    ATTACK_LABELS = {
        "random": "Random",
        "replay": "Replay",
        "stealthy": "Stealthy Bias",
        "noise": "Noise Injection",
        "targeted": "Targeted",
    }
    ATTACK_COLORS_MAP = {
        "random": RED, "replay": PURPLE, "stealthy": ORANGE,
        "noise": CYAN, "targeted": "#e6c619",
    }
    attack_types = net1_attack["attack_types"]

    # F1 comparison at 15% fraction
    col_f1_cmp, col_table = st.columns([3, 2])

    with col_f1_cmp:
        labels = [ATTACK_LABELS[a] for a in attack_types]
        net1_f1s, mod_f1s = [], []
        for atype in attack_types:
            n1_frac = net1_attack["results"][atype]["fraction_data"]
            md_frac = modena_attack["results"][atype]["fraction_data"]
            n1_rep = next((d for d in n1_frac if d["fraction"] == 0.15), n1_frac[2])
            md_rep = next((d for d in md_frac if d["fraction"] == 0.15), md_frac[2])
            net1_f1s.append(n1_rep["f1"])
            mod_f1s.append(md_rep["f1"])

        fig = go.Figure()
        fig.add_trace(go.Bar(x=labels, y=net1_f1s, name="Net1", marker_color=BLUE,
                             text=[f"{v:.3f}" for v in net1_f1s], textposition="outside",
                             textfont=dict(size=12)))
        fig.add_trace(go.Bar(x=labels, y=mod_f1s, name="Modena", marker_color=GREEN,
                             text=[f"{v:.3f}" for v in mod_f1s], textposition="outside",
                             textfont=dict(size=12)))
        fig.update_layout(**plotly_layout(
            title=dict(text="Detection F1 Score at 15% Attack Fraction"),
            yaxis_title="F1 Score", height=420, yaxis=dict(range=[0, 1.22]),
            barmode="group",
            legend=dict(orientation="h", x=0.5, xanchor="center", y=1.08),
            margin=dict(t=60),
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        rows = []
        for i, atype in enumerate(attack_types):
            rows.append({
                "Attack Type": ATTACK_LABELS[atype],
                "Net1 F1": f"{net1_f1s[i]:.3f}",
                "Modena F1": f"{mod_f1s[i]:.3f}",
                "Winner": "Net1" if net1_f1s[i] > mod_f1s[i] else "Modena" if mod_f1s[i] > net1_f1s[i] else "Tie",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=230)

        net1_avg = sum(net1_f1s) / len(net1_f1s)
        mod_avg = sum(mod_f1s) / len(mod_f1s)
        st.markdown(
            f"**Average F1** — Net1: `{net1_avg:.3f}` | Modena: `{mod_avg:.3f}`"
        )

    # F1 curves across all fractions — side by side
    col_net1_curve, col_mod_curve = st.columns(2)

    with col_net1_curve:
        fig = go.Figure()
        for atype in attack_types:
            r = net1_attack["results"][atype]
            fracs = [d["fraction"] * 100 for d in r["fraction_data"]]
            f1s = [d["f1"] for d in r["fraction_data"]]
            fig.add_trace(go.Scatter(
                x=fracs, y=f1s, mode="lines+markers", name=ATTACK_LABELS[atype],
                line=dict(color=ATTACK_COLORS_MAP[atype], width=2.5), marker=dict(size=6),
            ))
        fig.update_layout(**plotly_layout(
            title=dict(text="Net1 — F1 vs Attack Fraction"),
            xaxis_title="Attack Fraction (%)", yaxis_title="F1 Score",
            height=400, yaxis=dict(range=[0, 1.05]),
            legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.22,
                        font=dict(size=10)),
            margin=dict(b=80),
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col_mod_curve:
        fig = go.Figure()
        for atype in attack_types:
            r = modena_attack["results"][atype]
            fracs = [d["fraction"] * 100 for d in r["fraction_data"]]
            f1s = [d["f1"] for d in r["fraction_data"]]
            fig.add_trace(go.Scatter(
                x=fracs, y=f1s, mode="lines+markers", name=ATTACK_LABELS[atype],
                line=dict(color=ATTACK_COLORS_MAP[atype], width=2.5), marker=dict(size=6),
            ))
        fig.update_layout(**plotly_layout(
            title=dict(text="Modena — F1 vs Attack Fraction"),
            xaxis_title="Attack Fraction (%)", yaxis_title="F1 Score",
            height=400, yaxis=dict(range=[0, 1.05]),
            legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.22,
                        font=dict(size=10)),
            margin=dict(b=80),
        ))
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ──────────────────────────────────────────────
# Section 4: GNN vs Baselines (Net1)
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
# Section 5: Architecture comparison (Net1)
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
