"""Page 6: Training History."""

import streamlit as st
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_history, load_test_results
from utils.theme import GLOBAL_CSS, plotly_layout, BLUE, ORANGE, GREEN, PURPLE, RED, CYAN, DIM

st.set_page_config(page_title="Training History", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

st.title("Training History")
st.caption("MultiTaskGNN (GraphSAGE) trained on attack data — joint reconstruction and anomaly detection")

history = load_history()
test_results = load_test_results()

epochs = [h["epoch"] for h in history]
train_recon = [h["train_recon"] for h in history]
val_recon = [h["val_recon"] for h in history]
train_anomaly = [h["train_anomaly"] for h in history]
val_p_mae = [h["val_p_mae_unobs"] for h in history]
val_f1 = [h.get("val_p_anomaly_f1", 0) for h in history]
val_auroc = [h.get("val_p_anomaly_auroc", 0) for h in history]

# ── Metrics ──
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Epochs", len(history))
c2.metric("Best Val Loss", f"{min(val_recon):.4f}")
c3.metric("Best F1", f"{max(val_f1):.3f}")
c4.metric("Best AUROC", f"{max(val_auroc):.3f}")

st.divider()
log_scale = st.checkbox("Logarithmic y-axis")

# ── Loss curves ──
col1, col2 = st.columns(2)

with col1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=train_recon, name="Train",
        line=dict(color=BLUE, width=2),
        fill="tozeroy", fillcolor="rgba(77,166,255,0.05)",
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=val_recon, name="Validation",
        line=dict(color=ORANGE, width=2),
    ))
    fig.update_layout(**plotly_layout(
        title=dict(text="Reconstruction Loss (MSE)"),
        xaxis_title="Epoch", yaxis_title="Loss",
        height=380,
        yaxis_type="log" if log_scale else "linear",
        legend=dict(x=0.7, y=0.95),
    ))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=train_anomaly, name="Anomaly BCE",
        line=dict(color=PURPLE, width=2),
        fill="tozeroy", fillcolor="rgba(167,139,250,0.05)",
    ))
    fig.update_layout(**plotly_layout(
        title=dict(text="Anomaly Detection Loss (BCE)"),
        xaxis_title="Epoch", yaxis_title="Loss",
        height=380,
        yaxis_type="log" if log_scale else "linear",
        legend=dict(x=0.7, y=0.95),
    ))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Validation metrics ──
col1, col2 = st.columns(2)

with col1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=val_p_mae, name="Pressure MAE (unobs)",
        line=dict(color=ORANGE, width=2.5),
        fill="tozeroy", fillcolor="rgba(251,146,60,0.05)",
    ))
    # Mark best epoch
    best_ep = epochs[val_p_mae.index(min(val_p_mae))]
    best_val = min(val_p_mae)
    fig.add_trace(go.Scatter(
        x=[best_ep], y=[best_val], mode="markers",
        marker=dict(size=10, color=GREEN, symbol="diamond",
                    line=dict(width=1.5, color="rgba(255,255,255,0.4)")),
        name=f"Best ({best_val:.3f} @ epoch {best_ep})",
    ))
    fig.update_layout(**plotly_layout(
        title=dict(text="Validation Pressure MAE (Unobserved)"),
        xaxis_title="Epoch", yaxis_title="MAE (m)",
        height=380,
        legend=dict(x=0.45, y=0.95),
    ))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=val_f1, name="F1 Score",
        line=dict(color=GREEN, width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=val_auroc, name="AUROC",
        line=dict(color=CYAN, width=2.5),
    ))
    fig.update_layout(**plotly_layout(
        title=dict(text="Anomaly Detection Metrics (Validation)"),
        xaxis_title="Epoch", yaxis_title="Score",
        height=380,
        yaxis=dict(range=[0, 1.05]),
        legend=dict(x=0.7, y=0.25),
    ))
    st.plotly_chart(fig, use_container_width=True)

# ── Final results table ──
st.divider()
st.markdown("##### Final Test Set Results")

recon = test_results["reconstruction"]
anom = test_results.get("anomaly_detection", {})

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
**Reconstruction**

| Metric | All Nodes | Unobserved Only |
|--------|-----------|-----------------|
| Pressure MAE | {recon['pressure_all']['mae']:.3f} m | {recon['pressure_unobs']['mae']:.3f} m |
| Pressure RMSE | {recon['pressure_all']['rmse']:.3f} m | {recon['pressure_unobs']['rmse']:.3f} m |
| Flow MAE | {recon['flow_all']['mae']:.6f} | {recon['flow_unobs']['mae']:.6f} |
| Flow RMSE | {recon['flow_all']['rmse']:.6f} | {recon['flow_unobs']['rmse']:.6f} |
""")

with col2:
    if anom:
        p = anom.get("pressure", {})
        f = anom.get("flow", {})
        st.markdown(f"""
**Anomaly Detection**

| Metric | Pressure | Flow |
|--------|----------|------|
| Precision | {p.get('precision', 0):.1%} | {f.get('precision', 0):.1%} |
| Recall | {p.get('recall', 0):.1%} | {f.get('recall', 0):.1%} |
| F1 | {p.get('f1', 0):.3f} | {f.get('f1', 0):.3f} |
| AUROC | {p.get('auroc', 0):.3f} | {f.get('auroc', 0):.3f} |
""")
