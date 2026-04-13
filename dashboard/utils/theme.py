"""Shared theme constants and CSS for the dashboard."""

# Color palette — minimal, high contrast
BLUE = "#4da6ff"
GREEN = "#4ade80"
ORANGE = "#fb923c"
RED = "#f87171"
PURPLE = "#a78bfa"
CYAN = "#22d3ee"
YELLOW = "#fbbf24"
GRAY = "#9ca3af"
DIM = "#6b7280"

# Semantic colors
PRIMARY = BLUE
SUCCESS = GREEN
WARNING = ORANGE
DANGER = RED
ACCENT = CYAN

# Node type colors
NODE_JUNCTION = BLUE
NODE_RESERVOIR = GREEN
NODE_TANK = ORANGE

# Chart template
CHART_BG = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(128,128,128,0.1)"
AXIS_COLOR = "rgba(128,128,128,0.6)"
TEXT_COLOR = "rgba(80,80,80,1)"
TEXT_DIM = "rgba(110,110,110,0.8)"


def plotly_layout(**overrides):
    """Return a base Plotly layout dict for consistent styling."""
    base = dict(
        plot_bgcolor=CHART_BG,
        paper_bgcolor=CHART_BG,
        font=dict(family="Inter, -apple-system, sans-serif", color=TEXT_COLOR, size=12),
        title=dict(font=dict(size=14, color=TEXT_COLOR, weight="normal"), x=0.5, xanchor="center"),
        margin=dict(l=45, r=20, t=50, b=40),
        xaxis=dict(
            gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
            tickfont=dict(color=AXIS_COLOR, size=10),
            title_font=dict(color=TEXT_DIM, size=11),
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
            tickfont=dict(color=AXIS_COLOR, size=10),
            title_font=dict(color=TEXT_DIM, size=11),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=AXIS_COLOR),
            borderwidth=0,
        ),
        hoverlabel=dict(
            bgcolor="rgba(40,40,50,0.92)", font_size=11,
            font_color="white", bordercolor="rgba(200,200,200,0.15)",
        ),
    )
    base.update(overrides)
    return base


GLOBAL_CSS = """
<style>
    /* Metric cards — clean, subtle */
    div[data-testid="stMetric"] {
        background: rgba(128,128,128,0.04);
        border: 1px solid rgba(128,128,128,0.1);
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label {
        font-size: 0.75rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        opacity: 0.5;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 600;
    }

    /* Controls */
    .stSelectbox label, .stSlider label, .stCheckbox label, .stRadio label {
        font-size: 0.82rem;
        opacity: 0.6;
    }

    /* Info boxes */
    div[data-testid="stAlert"] {
        border-radius: 6px;
        border: 1px solid rgba(128,128,128,0.1);
        font-size: 0.88rem;
    }

    /* Dataframes */
    .stDataFrame {
        border-radius: 6px;
        overflow: hidden;
    }

    /* Dividers */
    hr {
        border-color: rgba(128,128,128,0.1) !important;
        margin: 1.2rem 0 !important;
    }

    /* Section headers */
    h5 {
        font-size: 0.95rem !important;
        letter-spacing: 0.02em;
        opacity: 0.8;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-size: 0.85rem;
    }

    /* Hide hamburger and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Reduce top padding */
    .block-container {
        padding-top: 2rem !important;
    }
</style>
"""
