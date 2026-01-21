"""
Module: styles.py
Description: Custom CSS styles for MARBL dashboard matching marbl.energy brand.
Author: MARBL Dashboard Team
Date: 2026-01-20

This module provides CSS styling to match the marbl.energy website aesthetic:
- Teal/Navy color palette
- Inter/Montserrat typography
- Card shadows and hover effects
- Smooth transitions (150-250ms)
"""

import streamlit as st


# --- Color Constants ---
MARBL_COLORS = {
    "primary": "#00B4BC",        # Teal - CTAs, accents
    "primary_dark": "#009AA2",   # Darker teal - hover states
    "secondary": "#31CEDC",      # Soft cyan - gradients
    "navy": "#0B253F",           # Dark navy - text
    "grey_light": "#F5F7FA",     # Light grey - backgrounds
    "grey_medium": "#6D7C8A",    # Medium grey - secondary text
    "white": "#FFFFFF",          # White - cards
    "success": "#2ECC71",        # Green - success states
    "warning": "#F39C12",        # Orange - warnings
    "error": "#E74C3C",          # Red - errors
}


# --- CSS Styles ---
def get_custom_css() -> str:
    """
    Return custom CSS for the MARBL dashboard.

    Returns
    -------
    str
        CSS string to inject via st.markdown.
    """
    return f"""
    <style>
    /* === Google Fonts Import === */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Montserrat:wght@600;700;800&display=swap');

    /* === Base Typography === */
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        color: {MARBL_COLORS['navy']};
    }}

    /* Headings use Montserrat */
    h1, h2, h3, .stTitle, [data-testid="stHeader"] {{
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 700 !important;
        color: {MARBL_COLORS['navy']} !important;
        letter-spacing: -0.02em;
    }}

    h1 {{
        font-size: 2.25rem !important;
    }}

    h2 {{
        font-size: 1.75rem !important;
    }}

    h3 {{
        font-size: 1.25rem !important;
    }}

    /* === Main Container === */
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }}

    /* === Cards and Containers === */
    [data-testid="stMetric"],
    [data-testid="stExpander"],
    .stDataFrame {{
        background-color: {MARBL_COLORS['white']};
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(11, 37, 63, 0.08);
        transition: box-shadow 0.2s ease, transform 0.2s ease;
    }}

    [data-testid="stMetric"]:hover,
    [data-testid="stExpander"]:hover {{
        box-shadow: 0 4px 16px rgba(11, 37, 63, 0.12);
        transform: translateY(-2px);
    }}

    /* Metric styling */
    [data-testid="stMetric"] {{
        border-left: 4px solid {MARBL_COLORS['primary']};
    }}

    [data-testid="stMetricLabel"] {{
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        color: {MARBL_COLORS['grey_medium']} !important;
        font-size: 0.875rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    [data-testid="stMetricValue"] {{
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 700 !important;
        color: {MARBL_COLORS['navy']} !important;
        font-size: 1.75rem !important;
    }}

    /* === Buttons === */
    .stButton > button {{
        background-color: {MARBL_COLORS['primary']} !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        padding: 0.625rem 1.25rem !important;
        transition: all 0.2s ease !important;
    }}

    .stButton > button:hover {{
        background-color: {MARBL_COLORS['primary_dark']} !important;
        box-shadow: 0 4px 12px rgba(0, 180, 188, 0.3) !important;
        transform: translateY(-1px) !important;
    }}

    .stButton > button:active {{
        transform: translateY(0) !important;
    }}

    /* Secondary buttons (outline style) */
    .stButton > button[kind="secondary"] {{
        background-color: transparent !important;
        color: {MARBL_COLORS['primary']} !important;
        border: 2px solid {MARBL_COLORS['primary']} !important;
    }}

    .stButton > button[kind="secondary"]:hover {{
        background-color: {MARBL_COLORS['primary']} !important;
        color: white !important;
    }}

    /* === Sidebar === */
    [data-testid="stSidebar"] {{
        background-color: {MARBL_COLORS['white']};
        border-right: 1px solid rgba(11, 37, 63, 0.1);
    }}

    [data-testid="stSidebar"] [data-testid="stMarkdown"] {{
        color: {MARBL_COLORS['navy']};
    }}

    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stDateInput label {{
        font-weight: 600;
        color: {MARBL_COLORS['navy']};
    }}

    /* === Select boxes and inputs === */
    .stSelectbox > div > div,
    .stDateInput > div > div {{
        border-radius: 8px !important;
        border-color: rgba(11, 37, 63, 0.2) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }}

    .stSelectbox > div > div:focus-within,
    .stDateInput > div > div:focus-within {{
        border-color: {MARBL_COLORS['primary']} !important;
        box-shadow: 0 0 0 3px rgba(0, 180, 188, 0.15) !important;
    }}

    /* === Expanders === */
    [data-testid="stExpander"] {{
        border: 1px solid rgba(11, 37, 63, 0.1);
    }}

    [data-testid="stExpander"] summary {{
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: {MARBL_COLORS['navy']};
        transition: color 0.2s ease;
    }}

    [data-testid="stExpander"] summary:hover {{
        color: {MARBL_COLORS['primary']};
    }}

    /* === Dividers === */
    hr {{
        border: none;
        height: 1px;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(11, 37, 63, 0.15),
            transparent
        );
        margin: 2rem 0;
    }}

    /* === Success/Info/Warning/Error boxes - subtle styling === */
    .stSuccess, .stInfo, .stWarning, .stError {{
        padding: 0.5rem 0.75rem !important;
        font-size: 0.875rem !important;
        border-radius: 0 6px 6px 0 !important;
    }}

    .stSuccess {{
        background-color: rgba(46, 204, 113, 0.08) !important;
        border-left: 3px solid {MARBL_COLORS['success']} !important;
    }}

    .stInfo {{
        background-color: rgba(0, 180, 188, 0.08) !important;
        border-left: 3px solid {MARBL_COLORS['primary']} !important;
    }}

    .stWarning {{
        background-color: rgba(243, 156, 18, 0.08) !important;
        border-left: 3px solid {MARBL_COLORS['warning']} !important;
    }}

    .stError {{
        background-color: rgba(231, 76, 60, 0.08) !important;
        border-left: 3px solid {MARBL_COLORS['error']} !important;
    }}

    /* === DataFrames === */
    .stDataFrame {{
        border: 1px solid rgba(11, 37, 63, 0.1);
    }}

    .stDataFrame thead th {{
        background-color: {MARBL_COLORS['primary']} !important;
        color: white !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
    }}

    /* Alternating row colors using marbl palette */
    .stDataFrame tbody tr:nth-child(even) {{
        background-color: rgba(0, 180, 188, 0.05) !important;
    }}

    .stDataFrame tbody tr:hover {{
        background-color: rgba(0, 180, 188, 0.1) !important;
    }}

    /* === Tabs === */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: transparent;
    }}

    .stTabs [data-baseweb="tab"] {{
        background-color: {MARBL_COLORS['white']};
        border-radius: 8px 8px 0 0;
        border: 1px solid rgba(11, 37, 63, 0.1);
        border-bottom: none;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        background-color: {MARBL_COLORS['grey_light']};
    }}

    .stTabs [aria-selected="true"] {{
        background-color: {MARBL_COLORS['primary']} !important;
        color: white !important;
    }}

    /* === Charts container === */
    [data-testid="stPlotlyChart"] {{
        background-color: {MARBL_COLORS['white']};
        border-radius: 12px;
        padding: 0.75rem;
        box-shadow: 0 2px 8px rgba(11, 37, 63, 0.08);
        overflow: hidden;
    }}

    /* Fix chart overflow - ensure charts don't extend beyond container */
    [data-testid="stPlotlyChart"] > div {{
        max-width: 100% !important;
        overflow: hidden !important;
    }}

    [data-testid="stPlotlyChart"] .js-plotly-plot {{
        max-width: 100% !important;
    }}

    /* === Footer === */
    footer {{
        font-family: 'Inter', sans-serif;
        color: {MARBL_COLORS['grey_medium']};
    }}

    /* === Animations === */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    .main .block-container > div {{
        animation: fadeIn 0.3s ease-out;
    }}

    /* === Spinner === */
    .stSpinner > div {{
        border-top-color: {MARBL_COLORS['primary']} !important;
    }}

    /* === Progress bar === */
    .stProgress > div > div {{
        background-color: {MARBL_COLORS['primary']} !important;
    }}

    /* === Toggle === */
    [data-testid="stToggle"] span[data-checked="true"] {{
        background-color: {MARBL_COLORS['primary']} !important;
    }}

    /* === Captions === */
    .stCaption {{
        color: {MARBL_COLORS['grey_medium']} !important;
        font-size: 0.875rem !important;
    }}
    </style>
    """


def apply_custom_styles():
    """
    Apply custom CSS styles to the Streamlit app.

    Call this function at the beginning of each page to apply consistent styling.

    Example
    -------
    >>> from utils.styles import apply_custom_styles
    >>> apply_custom_styles()
    """
    st.markdown(get_custom_css(), unsafe_allow_html=True)


def styled_metric_card(label: str, value: str, delta: str = None, delta_color: str = "normal"):
    """
    Create a styled metric card with marbl branding.

    Parameters
    ----------
    label : str
        The metric label.
    value : str
        The metric value.
    delta : str, optional
        The delta/change value.
    delta_color : str
        Color scheme for delta: "normal", "inverse", or "off".
    """
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


def styled_header(title: str, subtitle: str = None):
    """
    Create a styled page header.

    Parameters
    ----------
    title : str
        Main page title.
    subtitle : str, optional
        Subtitle or description.
    """
    st.title(title)
    if subtitle:
        st.markdown(f"<p style='color: {MARBL_COLORS['grey_medium']}; font-size: 1.1rem; margin-top: -1rem;'>{subtitle}</p>", unsafe_allow_html=True)


def styled_section_header(title: str, description: str = None):
    """
    Create a styled section header.

    Parameters
    ----------
    title : str
        Section title.
    description : str, optional
        Section description.
    """
    st.subheader(title)
    if description:
        st.markdown(f"<p style='color: {MARBL_COLORS['grey_medium']};'>{description}</p>", unsafe_allow_html=True)


def styled_info_card(title: str, subtitle: str, description: str, icon: str = None, color: str = None):
    """
    Create a styled info card/box for the landing page.

    Parameters
    ----------
    title : str
        Card title (e.g., zone name).
    subtitle : str
        Subtitle (e.g., market type).
    description : str
        Description text.
    icon : str, optional
        Emoji or icon to display.
    color : str, optional
        Accent color (defaults to primary teal).
    """
    accent = color or MARBL_COLORS['primary']
    icon_html = f"<span style='font-size: 1.5rem; margin-right: 0.5rem;'>{icon}</span>" if icon else ""

    html = f"""<div style="background: white; border-radius: 12px; padding: 1.25rem; box-shadow: 0 2px 8px rgba(11, 37, 63, 0.08); border-left: 4px solid {accent}; margin-bottom: 1rem;">
<h4 style="margin: 0 0 0.5rem 0; font-family: 'Montserrat', sans-serif; font-weight: 700; color: {MARBL_COLORS['navy']}; font-size: 1.1rem;">{icon_html}{title}</h4>
<p style="margin: 0 0 0.5rem 0; font-family: 'Inter', sans-serif; font-weight: 600; color: {accent}; font-size: 0.9rem;">{subtitle}</p>
<p style="margin: 0; font-family: 'Inter', sans-serif; color: {MARBL_COLORS['grey_medium']}; font-size: 0.85rem; line-height: 1.4;">{description}</p>
</div>"""
    st.markdown(html, unsafe_allow_html=True)


def styled_page_card(title: str, description: str, icon: str = None):
    """
    Create a styled page navigation card.

    Parameters
    ----------
    title : str
        Page title.
    description : str
        Page description.
    icon : str, optional
        Emoji or icon to display.
    """
    icon_html = f"<span style='font-size: 1.25rem; margin-right: 0.5rem;'>{icon}</span>" if icon else ""

    html = f"""<div style="background: white; border-radius: 12px; padding: 1.25rem; box-shadow: 0 2px 8px rgba(11, 37, 63, 0.08); height: 100%;">
<h4 style="margin: 0 0 0.75rem 0; font-family: 'Montserrat', sans-serif; font-weight: 700; color: {MARBL_COLORS['navy']}; font-size: 1rem;">{icon_html}{title}</h4>
<p style="margin: 0; font-family: 'Inter', sans-serif; color: {MARBL_COLORS['grey_medium']}; font-size: 0.85rem; line-height: 1.5;">{description}</p>
</div>"""
    st.markdown(html, unsafe_allow_html=True)


def styled_footer():
    """Create a consistent styled footer for all pages."""
    st.markdown("---")
    html = f"""<div style="padding: 1rem 0; font-family: 'Inter', sans-serif; font-size: 0.8rem; color: {MARBL_COLORS['grey_medium']}; line-height: 1.6;">
<p style="margin: 0 0 0.5rem 0;"><strong>Data Sources:</strong> Electricity prices from ENTSO-E Transparency Platform | Historical weather from ERA5 Reanalysis (Copernicus Climate Data Store) | Weather forecasts from WeatherAPI.com</p>
<p style="margin: 0;"><strong>Disclaimer:</strong> This dashboard is for educational and research purposes only. Forecasts are model predictions and should not be used for trading decisions.</p>
</div>"""
    st.markdown(html, unsafe_allow_html=True)
