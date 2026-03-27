"""Shared theme constants and CSS for the dashboard."""

# Color palette — dark-mode first, high contrast
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
GRID_COLOR = "rgba(255,255,255,0.06)"
AXIS_COLOR = "rgba(255,255,255,0.4)"
TEXT_COLOR = "rgba(255,255,255,0.85)"
TEXT_DIM = "rgba(255,255,255,0.5)"


def plotly_layout(**overrides):
    """Return a base Plotly layout dict for consistent styling."""
    base = dict(
        plot_bgcolor=CHART_BG,
        paper_bgcolor=CHART_BG,
        font=dict(family="Inter, -apple-system, sans-serif", color=TEXT_COLOR, size=13),
        title=dict(font=dict(size=15, color=TEXT_COLOR), x=0.5, xanchor="center"),
        margin=dict(l=50, r=30, t=60, b=50),
        xaxis=dict(
            gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
            tickfont=dict(color=AXIS_COLOR, size=11),
            title_font=dict(color=TEXT_DIM, size=12),
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
            tickfont=dict(color=AXIS_COLOR, size=11),
            title_font=dict(color=TEXT_DIM, size=12),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=AXIS_COLOR),
            borderwidth=0,
        ),
        hoverlabel=dict(
            bgcolor="rgba(30,30,40,0.95)", font_size=12,
            font_color="white", bordercolor="rgba(255,255,255,0.1)",
        ),
    )
    base.update(overrides)
    return base


GLOBAL_CSS = """
<style>
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 16px 20px;
    }
    div[data-testid="stMetric"] label {
        font-size: 0.8rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        opacity: 0.6;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
    }

    /* Selectbox / slider labels */
    .stSelectbox label, .stSlider label, .stCheckbox label {
        font-size: 0.85rem;
        opacity: 0.7;
    }

    /* Info boxes */
    div[data-testid="stAlert"] {
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.08);
    }

    /* Dataframes */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Dividers */
    hr {
        border-color: rgba(255,255,255,0.06) !important;
        margin: 1.5rem 0 !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-size: 0.9rem;
    }

    /* Hide hamburger menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""
