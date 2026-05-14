"""WDN Adversarial Self-Play Dashboard — Landing page."""

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

# Extra CSS for the hero band on the landing page only.
st.markdown(
    """
    <style>
    .hero-band {
        background: linear-gradient(135deg,
            rgba(96,165,250,0.10) 0%,
            rgba(167,139,250,0.08) 50%,
            rgba(74,222,128,0.08) 100%);
        border: 1px solid rgba(128,128,128,0.18);
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 6px;
    }
    .hero-eyebrow {
        font-size: 0.72rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        opacity: 0.55;
        margin-bottom: 6px;
    }
    .hero-title {
        font-size: 1.55rem;
        font-weight: 600;
        line-height: 1.25;
        margin-bottom: 8px;
    }
    .hero-sub {
        font-size: 0.95rem;
        opacity: 0.78;
        line-height: 1.5;
    }
    .pill {
        display: inline-block;
        background: rgba(74,222,128,0.12);
        color: #4ade80;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .pill.purple { background: rgba(167,139,250,0.14); color: #a78bfa; }
    .pill.blue   { background: rgba(96,165,250,0.14); color: #60a5fa; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-band">
      <div class="hero-eyebrow">MSc thesis · Sapienza Università di Roma</div>
      <div class="hero-title">Adversarial Self-Play GNNs for<br>Water-Network Cyber-Defence</div>
      <div class="hero-sub">
        An attacker GNN and a defender GNN co-evolve in a Stackelberg
        game. The defender ends up <b>more accurate on hand-crafted
        attacks</b> and <b>generalises to attack types it has never seen</b>.
      </div>
      <div style="margin-top:14px;">
        <span class="pill">3 benchmark networks</span>
        <span class="pill blue">10 trained models</span>
        <span class="pill purple">9 evaluation scripts</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

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


# Top row: per-network snapshot
st.markdown("##### Per-network snapshot")
col_n1, col_n3, col_md = st.columns(3)

with col_n1:
    st.markdown("**Net1** &nbsp;·&nbsp; 11 junctions · 13 pipes")
    a, b, c = st.columns(3)
    a.metric("P MAE", f"{net1['reconstruction']['pressure_unobs']['mae']:.2f} m")
    b.metric("Anomaly F1", f"{net1['anomaly_detection']['pressure']['f1']:.3f}")
    rep = _replay_f1(net1_moe_t)
    c.metric("Replay F1", f"{rep:.3f}" if rep is not None else "—")

with col_n3:
    st.markdown("**Net3** &nbsp;·&nbsp; 97 junctions · 117 pipes")
    a, b, c = st.columns(3)
    if net3:
        a.metric("P MAE", f"{net3['reconstruction']['pressure_unobs']['mae']:.2f} m")
        b.metric("Anomaly F1", f"{net3['anomaly_detection']['pressure']['f1']:.3f}")
        rep = _replay_f1(net3_moe_t) if net3_moe_t else _replay_f1(net3)
        c.metric("Replay F1", f"{rep:.3f}" if rep is not None else "—")
    else:
        a.metric("P MAE", "—"); b.metric("Anomaly F1", "—"); c.metric("Replay F1", "—")

with col_md:
    st.markdown("**Modena** &nbsp;·&nbsp; 272 junctions · 317 pipes")
    a, b, c = st.columns(3)
    if sp.get("atkmoe"):
        sp_moe = sp["atkmoe"]["sp_moe"]
        a.metric("P MAE", f"{sp_moe['p_mae']:.3f} m")
        b.metric("Anomaly F1", f"{sp_moe['f1']:.3f}")
        c.metric("Replay F1", f"{sp_moe['per_attack']['replay']['f1']:.3f}")
    else:
        a.metric("P MAE", f"{mod['reconstruction']['pressure_unobs']['mae']:.2f} m")
        b.metric("Anomaly F1", f"{mod['anomaly_detection']['pressure']['f1']:.3f}")
        c.metric("Replay F1", "—")

st.divider()

# ── Self-play tagline + key numbers ──────────────────────────────────
st.markdown("##### Self-play vs supervised baseline (Modena)")

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
            "Novel-attack F1", f"{h['sp_moe']['f1']:.3f}",
            f"{h['sp_moe']['f1'] - h['pretrained']['f1']:+.3f} on unseen attack",
        )
    else:
        cols[3].metric("Novel-attack F1", "—")

st.divider()

# ── What the dashboard shows ─────────────────────────────────────────
col_l, col_r = st.columns([3, 2], gap="large")

with col_l:
    st.markdown(
        """
        ##### Approach
        The pipe network is a graph; a Graph Neural Network jointly
        reconstructs missing pressure / flow values and flags sensors
        that have been compromised by a cyber-attack. Built up in five
        stages, each addressing a specific weakness of the previous one:

        1. **Spatial GNN** — GraphSAGE backbone with four prediction
           heads, mass-conservation penalty.
        2. **Temporal GNN + GRU** — 6-step sliding window adds the
           memory needed to spot replay attacks.
        3. **Mixture-of-Experts** — six attack-specialised experts plus
           a learned router; load-balancing entropy prevents collapse.
        4. **Pattern-detection features** — autocorrelation,
           adjacent-difference std, noise ratio target the missing
           observation noise of replayed values.
        5. **Adversarial self-play** — an attacker GNN learns to
           generate sparse, physics-aware perturbations; the defender
           catches them. The attacker MoE auto-discovers two attack
           families without class supervision and the resulting
           defender generalises to held-out novel attacks.
        """
    )

with col_r:
    st.markdown(
        """
        ##### Dashboard guide

        | # | Page | What it answers |
        |---|------|-----------------|
        | 1 | Network Overview | Topology + per-node properties |
        | 2 | Reconstruction | True vs predicted pressure / flow |
        | 3 | Attack Analysis | Per-attack-type detection curves |
        | 4 | Model Comparison | Spatial → Temporal → MoE benchmarks |
        | 5 | Explainability | Feature and node-importance maps |
        | 6 | **Self-Play** ⭐ | Co-evolution, classical baselines, robustness, cross-network, novel-attack generalisation, vocabulary |
        | 7 | GNN Ablation | 4 backbones × 3 networks |
        | 8 | Methodology | Pipeline diagram + architecture progression |
        | 9 | Live Attack | Inject an attack, watch the defender flag it |
        """
    )
