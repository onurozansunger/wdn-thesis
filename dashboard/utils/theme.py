"""Shared theme constants and CSS for the dashboard.

The categorical palette below is validated for the locked light surface
(#fcfcfb, see .streamlit/config.toml) with the dataviz palette validator:
all hues sit in the L 0.43–0.77 band, clear the chroma floor, keep worst
adjacent-pair CVD separation ΔE ≈ 35, and every mark carries a direct
label so the single sub-3:1 contrast case (yellow) is relieved.
"""

# ── Categorical palette (validated, fixed order) ─────────────────────
# Assign in this order; never cycle a 9th hue.
BLUE = "#2563eb"
ORANGE = "#ea580c"
GREEN = "#059669"
PURPLE = "#7c3aed"
RED = "#dc2626"
CYAN = "#0891b2"
YELLOW = "#ca8a04"
GRAY = "#6b7280"
DIM = "#9aa1ac"

# Ordered categorical sequence (identity, in fixed order).
CATEGORICAL = [BLUE, ORANGE, GREEN, PURPLE, RED, CYAN, YELLOW]

# Semantic colors
PRIMARY = BLUE
SUCCESS = GREEN
WARNING = ORANGE
DANGER = RED
ACCENT = CYAN

# Canonical per-attack colors — keep identical across every page so a
# colour always means the same attack.
ATTACK_COLORS = {
    "random": BLUE,
    "replay": PURPLE,
    "stealthy": ORANGE,
    "noise": CYAN,
    "targeted": GREEN,
}
ATTACK_LABELS = {
    "random": "Random", "replay": "Replay", "stealthy": "Stealthy",
    "noise": "Noise", "targeted": "Targeted",
}

# Per-domain colors (water / power / traffic) — used by the cross-domain page.
DOMAIN_COLORS = {"water": BLUE, "power": ORANGE, "traffic": GREEN}

# Node type colors
NODE_JUNCTION = BLUE
NODE_RESERVOIR = GREEN
NODE_TANK = ORANGE

# ── Surfaces & ink tokens (light theme) ──────────────────────────────
SURFACE = "#fcfcfb"
CHART_BG = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(31,41,55,0.07)"        # recessive grid
ZERO_COLOR = "rgba(31,41,55,0.18)"        # zero line, a touch stronger
AXIS_COLOR = "rgba(31,41,55,0.55)"        # tick labels
TEXT_COLOR = "#1f2937"                     # primary ink
TEXT_DIM = "rgba(31,41,55,0.62)"           # secondary ink
TEXT_MUTED = "rgba(31,41,55,0.45)"         # muted ink


def plotly_layout(**overrides):
    """Return a base Plotly layout dict for consistent, polished styling.

    Recessive grid/axes, rounded bar ends, generous margins, ink-token
    text (never the series colour), and a dark hover card.
    """
    base = dict(
        plot_bgcolor=CHART_BG,
        paper_bgcolor=CHART_BG,
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
                  color=TEXT_COLOR, size=12),
        title=dict(text="", font=dict(size=14.5, color=TEXT_COLOR, weight=600),
                   x=0.5, xanchor="center", y=0.97),
        margin=dict(l=48, r=22, t=52, b=42),
        barcornerradius=4,
        bargap=0.28,
        colorway=CATEGORICAL,
        xaxis=dict(
            gridcolor=GRID_COLOR, zerolinecolor=ZERO_COLOR,
            linecolor=GRID_COLOR, tickfont=dict(color=AXIS_COLOR, size=10.5),
            title_font=dict(color=TEXT_DIM, size=11.5),
            automargin=True,
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR, zerolinecolor=ZERO_COLOR,
            linecolor=GRID_COLOR, tickfont=dict(color=AXIS_COLOR, size=10.5),
            title_font=dict(color=TEXT_DIM, size=11.5),
            automargin=True,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", font=dict(size=10.5, color=TEXT_DIM),
            borderwidth=0,
        ),
        hoverlabel=dict(
            bgcolor="rgba(31,41,55,0.94)", font_size=11.5,
            font_color="#f8fafc", bordercolor="rgba(255,255,255,0.12)",
            font_family="Inter, sans-serif",
        ),
    )
    # Deep-merge the nested style dicts so a caller passing e.g.
    # yaxis=dict(range=[0,1]) keeps the base grid/tick styling instead
    # of replacing it, and title=dict(text=...) keeps the base font.
    for key in ("title", "xaxis", "yaxis", "legend", "hoverlabel"):
        if key in overrides and isinstance(overrides[key], dict) \
                and isinstance(base.get(key), dict):
            merged = {**base[key], **overrides.pop(key)}
            base[key] = merged
    base.update(overrides)
    return base


GLOBAL_CSS = """
<style>
    /* ── Type & rhythm ──────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-feature-settings: "cv02","cv03","cv04","cv11";
    }
    .block-container {
        padding-top: 2.2rem !important;
        max-width: 1200px;
    }

    /* Page titles */
    h1 {
        letter-spacing: -0.022em !important;
        font-weight: 680 !important;
        font-size: 2.0rem !important;
    }
    h2, h3 {
        letter-spacing: -0.012em !important;
        font-weight: 640 !important;
    }
    /* Section headers (##### ) */
    h5 {
        font-size: 0.80rem !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase;
        color: rgba(31,41,55,0.55) !important;
        font-weight: 650 !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.15rem !important;
    }

    /* Captions quieter */
    div[data-testid="stCaptionContainer"], .stCaption {
        opacity: 0.72;
        font-size: 0.86rem !important;
        line-height: 1.5;
    }

    /* ── Metric cards — clean, subtle, hover lift ───────────────── */
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg,
            rgba(37,99,235,0.035), rgba(37,99,235,0.012));
        border: 1px solid rgba(31,41,55,0.09);
        border-radius: 12px;
        padding: 13px 17px;
        transition: transform 0.18s ease, border-color 0.18s ease,
                    box-shadow 0.18s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(37,99,235,0.32);
        box-shadow: 0 10px 24px -14px rgba(37,99,235,0.5);
    }
    div[data-testid="stMetric"] label {
        font-size: 0.70rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        opacity: 0.58;
        font-weight: 600;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.7rem;
        font-weight: 680;
        letter-spacing: -0.02em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        font-size: 0.76rem;
        opacity: 0.85;
    }

    /* Controls */
    .stSelectbox label, .stSlider label, .stCheckbox label, .stRadio label {
        font-size: 0.80rem;
        opacity: 0.66;
        font-weight: 500;
    }

    /* Alerts — softer, classier */
    div[data-testid="stAlert"] {
        border-radius: 10px;
        border: 1px solid rgba(31,41,55,0.10);
        font-size: 0.87rem;
    }

    /* Dataframes */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* Dividers — airy */
    hr {
        border-color: rgba(31,41,55,0.08) !important;
        margin: 1.5rem 0 !important;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        border-radius: 10px;
        border: 1px solid rgba(31,41,55,0.09);
    }

    /* Tabs */
    button[data-baseweb="tab"] { font-size: 0.86rem; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(31,41,55,0.07);
    }
    section[data-testid="stSidebar"] a { font-size: 0.92rem !important; }

    /* Plotly charts sit in a soft card */
    div[data-testid="stPlotlyChart"] {
        border-radius: 12px;
    }

    /* Hide chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    html { scroll-behavior: smooth; }
</style>
"""
