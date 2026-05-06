"""WDN Thesis Dashboard — Landing page."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from utils.theme import GLOBAL_CSS
from utils.data_loader import (
    load_test_results_net1, load_test_results_net3, load_test_results_modena,
    load_moe_results, load_selfplay_summary,
)

st.set_page_config(
    page_title="WDN Adversarial Defence",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

st.title("Adversarial Self-Play GNNs for Water-Network Cyber-Defence")
st.caption(
    "An attacker GNN and a defender GNN co-evolve in a Stackelberg "
    "game; the resulting defender is more accurate on hand-crafted "
    "attacks and generalises to attack types it has never seen."
)

st.divider()

# ── Headline numbers ─────────────────────────────────────────────────
net1 = load_test_results_net1()
net3 = load_test_results_net3()
mod = load_test_results_modena()
net1_moe_t = load_moe_results("Net1", "temporal")
net3_moe_t = load_moe_results("Net3", "temporal")
mod_moe_t = load_moe_results("Modena", "temporal")
sp = load_selfplay_summary()


def _replay_f1(res):
    if not res:
        return None
    return res.get("per_attack_pressure", {}).get("replay", {}).get("f1")


def _atkmoe_metric(key, default=None):
    if not sp.get("atkmoe"):
        return default
    return sp["atkmoe"]["sp_moe"].get(key, default)


# Top row: per-network snapshot
col_n1, col_n3, col_md = st.columns(3)

with col_n1:
    st.markdown("**Net1** &nbsp;·&nbsp; 11 nodes · 13 pipes")
    a, b, c = st.columns(3)
    a.metric("P MAE", f"{net1['reconstruction']['pressure_unobs']['mae']:.2f} m")
    b.metric("Anomaly F1", f"{net1['anomaly_detection']['pressure']['f1']:.3f}")
    rep = _replay_f1(net1_moe_t)
    c.metric("Replay F1", f"{rep:.3f}" if rep is not None else "—")

with col_n3:
    st.markdown("**Net3** &nbsp;·&nbsp; 97 nodes · 117 pipes")
    a, b, c = st.columns(3)
    if net3:
        a.metric("P MAE", f"{net3['reconstruction']['pressure_unobs']['mae']:.2f} m")
        b.metric("Anomaly F1", f"{net3['anomaly_detection']['pressure']['f1']:.3f}")
        rep = _replay_f1(net3_moe_t) if net3_moe_t else _replay_f1(net3)
        c.metric("Replay F1", f"{rep:.3f}" if rep is not None else "—")
    else:
        a.metric("P MAE", "—")
        b.metric("Anomaly F1", "—")
        c.metric("Replay F1", "—")

with col_md:
    st.markdown("**Modena** &nbsp;·&nbsp; 272 nodes · 317 pipes")
    a, b, c = st.columns(3)
    mae = _atkmoe_metric("p_mae", mod["reconstruction"]["pressure_unobs"]["mae"])
    f1 = _atkmoe_metric("f1", mod["anomaly_detection"]["pressure"]["f1"])
    a.metric("P MAE", f"{mae:.3f} m")
    b.metric("Anomaly F1", f"{f1:.3f}")
    rep = _replay_f1(mod_moe_t)
    c.metric("Replay F1", f"{rep:.3f}" if rep is not None else "—")

st.divider()

# ── Self-play tagline + key numbers ──────────────────────────────────
st.markdown("### Self-play vs supervised baseline (Modena)")

if sp.get("atkmoe"):
    pre = sp["atkmoe"]["pretrained"]
    sp_moe = sp["atkmoe"]["sp_moe"]
    cols = st.columns(4)
    cols[0].metric(
        "Anomaly F1", f"{sp_moe['f1']:.3f}",
        f"{sp_moe['f1'] - pre['f1']:+.3f} vs pretrained",
    )
    cols[1].metric(
        "P MAE (m)", f"{sp_moe['p_mae']:.3f}",
        f"{sp_moe['p_mae'] - pre['p_mae']:+.3f}", delta_color="inverse",
    )
    cols[2].metric(
        "Targeted F1", f"{sp_moe['per_attack']['targeted']['f1']:.3f}",
        f"{sp_moe['per_attack']['targeted']['f1'] - pre['per_attack']['targeted']['f1']:+.3f}",
    )
    if sp.get("heldout"):
        h = sp["heldout"]["sinusoidal"]
        cols[3].metric(
            "Novel attack F1", f"{h['sp_moe']['f1']:.3f}",
            f"{h['sp_moe']['f1'] - h['pretrained']['f1']:+.3f} on unseen attack",
        )
    else:
        cols[3].metric("Novel attack F1", "—")

st.divider()

# ── What the dashboard shows ─────────────────────────────────────────
col_l, col_r = st.columns([3, 2], gap="large")

with col_l:
    st.markdown(
        """
        **Approach.** The pipe network is a graph; a Graph Neural
        Network jointly reconstructs missing pressure / flow values and
        flags sensors that have been compromised by a cyber-attack.
        Built up in five stages, each addressing a specific weakness
        of the previous one:

        1. **Spatial GNN** — GraphSAGE backbone with four prediction
           heads, mass-conservation penalty.
        2. **Temporal GNN+GRU** — 6-step sliding window adds the
           memory needed to spot replay attacks.
        3. **Mixture-of-Experts** — six attack-specialised experts plus
           a learned router; load-balancing entropy prevents collapse.
        4. **Pattern-detection features** — autocorrelation,
           adjacent-difference std, noise ratio target the missing
           observation noise of replayed values.
        5. **Adversarial self-play** — an Attacker GNN learns to
           generate sparse, physics-aware perturbations; the defender
           catches them. The Attacker-MoE auto-discovers two attack
           families without class supervision and the resulting
           defender generalises to held-out novel attacks.
        """
    )

with col_r:
    st.markdown(
        """
        | Page | Content |
        |------|---------|
        | Network Overview | Topology and per-node properties |
        | Reconstruction | True vs predicted pressure / flow |
        | Attack Analysis | Per-attack-type detection curves |
        | Anomaly Detection | Interactive thresholding demo |
        | Model Comparison | Spatial → Temporal → MoE benchmarks |
        | Explainability | Feature and node-importance maps |
        | **Self-Play** | Co-evolution, novel-attack generalisation, vocabulary |
        | **GNN Ablation** | 4 backbones × 3 networks — defending the GraphSAGE choice |
        | **Methodology** | Pipeline diagram + architecture progression + self-play loop |
        | **Live Attack** | Inject an attack, watch the defender flag it on the map |
        """
    )
