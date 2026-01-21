"""
Module: app.py
Description: Main entry point for the MARBL Dashboard.
Author: MARBL Dashboard Team
Date: 2026-01-16

Run with: python -m streamlit run app.py
"""

# --- Imports ---
from pathlib import Path

import streamlit as st

from utils.data_loader import (
    check_data_availability,
    ZONE_NAMES,
    ZONE_DESCRIPTIONS,
    ZONE_SHORT
)


# --- Page Configuration ---
st.set_page_config(
    page_title="MARBL Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- Sidebar with Logo ---
def render_sidebar():
    """Render the sidebar with MARBL branding."""
    # Display logo
    logo_path = Path(__file__).parent / "assets" / "marbl_logo.png"
    if logo_path.exists():
        st.sidebar.image(str(logo_path), use_container_width=True)
    else:
        st.sidebar.markdown("### MARBL")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Navigation**")
    st.sidebar.markdown("Use the pages above to explore the dashboard.")


# --- Main Page Content ---
def main():
    """Render the main landing page of the dashboard."""

    render_sidebar()

    # Header
    st.title("MARBL Energy Dashboard")
    st.markdown(
        """
        **Electricity Price Trends and Market Volatility in Europe**

        This dashboard provides insights into European electricity markets,
        including historical price analysis, pattern detection, and day-ahead
        price forecasting for three distinct bidding zones.
        """
    )

    st.divider()

    # --- Bidding Zones Overview ---
    st.subheader("Bidding Zones")

    col1, col2, col3 = st.columns(3)

    zones_data = [
        ("DK1", col1, "#2563eb"),
        ("ES", col2, "#dc2626"),
        ("NO2", col3, "#059669")
    ]

    availability = check_data_availability()

    for zone, col, color in zones_data:
        with col:
            status = "Available" if availability.get(zone, False) else "Not Found"
            status_icon = "white_check_mark" if availability.get(zone, False) else "x"

            st.markdown(f"**{ZONE_NAMES.get(zone, zone)}**")
            st.markdown(f"*{ZONE_SHORT.get(zone, '')} Market*")
            st.caption(ZONE_DESCRIPTIONS.get(zone, ""))
            st.markdown(f"Data: :{status_icon}: {status}")

    st.divider()

    # --- Dashboard Pages ---
    st.subheader("Dashboard Pages")

    pages_col1, pages_col2, pages_col3 = st.columns(3)

    with pages_col1:
        st.markdown("**Market Overview**")
        st.markdown(
            "Explore historical price and weather data. "
            "View price trends, daily profiles, and correlations "
            "between weather conditions and electricity prices."
        )

    with pages_col2:
        st.markdown("**Cluster Analysis**")
        st.markdown(
            "View identified price patterns from hierarchical clustering. "
            "Understand recurring daily price shapes and their seasonal distribution."
        )

    with pages_col3:
        st.markdown("**Live Forecast**")
        st.markdown(
            "Day-ahead price predictions based on weather forecasts "
            "and the XGBoost Mix-of-Experts model."
        )

    st.divider()

    # --- About Section ---
    st.subheader("About This Project")

    st.markdown(
        """
        This dashboard was developed as part of the **Data Science Lab 2025/26**
        at **WU Vienna** in collaboration with **Marbl FlexCo**.

        The project investigates electricity price dynamics across three European
        bidding zones with distinct generation mixes: Denmark (wind-dominated),
        Spain (solar-dominated), and Norway (hydro-dominated).

        **Key Components:**
        - Historical price analysis and weather correlation
        - Hierarchical clustering for pattern detection
        - XGBoost Mix-of-Experts forecasting model
        """
    )

    # Team info in expander
    with st.expander("Project Team"):
        st.markdown(
            """
            **Team Members:**
            - Maximilian Dieringer
            - Harald Körbel
            - Maxim Gomez Valverde
            - Daniel Klaric

            **Supervisor:** Univ.Prof. Dr. Kavita Surana

            **Client:** Marbl FlexCo
            """
        )

    st.divider()

    # --- Data Attribution Footer ---
    st.markdown("---")
    st.caption(
        """
        **Data Sources:**
        Electricity prices from ENTSO-E Transparency Platform |
        Historical weather from ERA5 Reanalysis (Copernicus Climate Data Store) |
        Weather forecasts from WeatherAPI.com
        """
    )
    st.caption(
        """
        **Disclaimer:** This dashboard is for educational and research purposes only.
        Forecasts are model predictions and should not be used for trading decisions.
        """
    )


# --- Entry Point ---
if __name__ == "__main__":
    main()
