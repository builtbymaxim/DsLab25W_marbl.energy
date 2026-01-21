"""
Module: 1_Market_Overview.py
Description: Historical price and weather data exploration page.
Author: MARBL Dashboard Team
Date: 2026-01-16
"""

# --- Imports ---
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import io

import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import (
    load_masterset,
    load_masterset_filtered,
    load_all_mastersets,
    get_daily_prices,
    get_daily_profile,
    get_data_summary,
    get_date_range,
    VALID_ZONES,
    ZONE_NAMES,
    ZONE_DESCRIPTIONS,
    ZONE_SHORT
)
from utils.visualizations import (
    create_price_timeseries,
    create_multi_zone_prices,
    create_daily_profile_chart,
    create_weather_chart,
    create_price_weather_scatter,
    ZONE_COLORS
)
from utils.styles import apply_custom_styles, styled_footer


# --- Page Configuration ---
st.set_page_config(
    page_title="Market Overview - MARBL",
    page_icon="chart_with_upwards_trend",
    layout="wide"
)


# --- Sidebar ---
def render_sidebar():
    """Render the sidebar with MARBL branding."""
    logo_path = Path(__file__).parent.parent / "assets" / "marbl_logo.png"
    if logo_path.exists():
        st.sidebar.image(str(logo_path), use_container_width=True)
    st.sidebar.markdown("---")


# --- Helper Functions ---
def format_number(value: float, decimals: int = 2) -> str:
    """Format a number with thousand separators."""
    return f"{value:,.{decimals}f}"


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to CSV bytes for download."""
    return df.to_csv().encode("utf-8")


# --- Main Page ---
def main():
    """Render the market overview page."""

    # Apply custom CSS styles
    apply_custom_styles()

    render_sidebar()

    st.title("Market Overview")
    st.markdown("Explore historical electricity prices and weather data across European bidding zones.")

    st.divider()

    # --- Sidebar Controls ---
    st.sidebar.header("Filters")

    # View mode selection
    view_mode = st.sidebar.radio(
        label="View Mode",
        options=["Single Zone", "Multi-Zone Comparison"],
        index=0,
        help="Compare prices across all zones or focus on a single zone"
    )

    st.sidebar.markdown("---")

    if view_mode == "Single Zone":
        # Zone selection
        selected_zone = st.sidebar.selectbox(
            label="Bidding Zone",
            options=VALID_ZONES,
            format_func=lambda x: ZONE_NAMES.get(x, x),
            index=0
        )

        # Display zone info
        st.sidebar.caption(f"{ZONE_SHORT.get(selected_zone, '')} Market")
        st.sidebar.caption(ZONE_DESCRIPTIONS.get(selected_zone, ""))

        st.sidebar.markdown("---")

        # Get available date range for selected zone
        try:
            min_date, max_date = get_date_range(selected_zone)
            min_date = min_date.date()
            max_date = max_date.date()
        except Exception as e:
            st.error(f"Error loading data for {selected_zone}: {str(e)}")
            st.stop()

        # Date range selection
        st.sidebar.markdown("**Date Range**")

        # Default to last 3 months
        default_start = max(min_date, max_date - timedelta(days=90))

        start_date = st.sidebar.date_input(
            label="Start Date",
            value=default_start,
            min_value=min_date,
            max_value=max_date
        )

        end_date = st.sidebar.date_input(
            label="End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )

        # Validate date range
        if start_date > end_date:
            st.sidebar.error("Start date must be before end date.")
            st.stop()

        # --- Load Data ---
        with st.spinner("Loading data..."):
            try:
                df = load_masterset_filtered(
                    zone=selected_zone,
                    start_date=str(start_date),
                    end_date=str(end_date)
                )
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                st.stop()

        # Check if data is available
        if df.empty:
            st.warning("No data available for the selected date range.")
            st.stop()

        # --- Export Button ---
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Export Data**")

        csv_data = convert_df_to_csv(df)
        st.sidebar.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"{selected_zone}_data_{start_date}_{end_date}.csv",
            mime="text/csv",
            use_container_width=True
        )

        # --- Summary Statistics ---
        st.subheader("Summary Statistics")

        summary = get_data_summary(df)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Average Price",
                value=f"{summary['price_mean']:.2f} EUR/MWh"
            )

        with col2:
            st.metric(
                label="Price Range (EUR/MWh)",
                value=f"{summary['price_min']:.2f} - {summary['price_max']:.2f}"
            )

        with col3:
            st.metric(
                label="Std Deviation",
                value=f"{summary['price_std']:.2f} EUR/MWh"
            )

        with col4:
            st.metric(
                label="Data Points",
                value=format_number(summary['total_records'], 0)
            )

        st.divider()

        # --- Price Time Series ---
        st.subheader("Price History")

        # Aggregation option
        agg_option = st.radio(
            label="Aggregation",
            options=["Hourly", "Daily Average"],
            horizontal=True
        )

        if agg_option == "Daily Average":
            daily_df = get_daily_prices(df)
            # Reshape for plotting
            plot_df = pd.DataFrame({
                "price_eur_mwh": daily_df["price_mean"]
            })
            plot_df.index = daily_df.index
        else:
            plot_df = df

        fig_price = create_price_timeseries(
            df=plot_df,
            zone=selected_zone,
            title=f"Electricity Prices - {ZONE_NAMES.get(selected_zone, selected_zone)}"
        )

        st.plotly_chart(fig_price, use_container_width=True)

        st.divider()

        # --- Daily Profile ---
        st.subheader("Average Daily Profile")

        st.markdown(
            """
            The average price at each hour of the day, calculated across all days in the selected period.
            The shaded area shows one standard deviation.
            """
        )

        profile_df = get_daily_profile(df)

        fig_profile = create_daily_profile_chart(
            profile_df=profile_df,
            zone=selected_zone,
            show_std=True
        )

        st.plotly_chart(fig_profile, use_container_width=True)

        st.divider()

        # --- Weather Data ---
        st.subheader("Weather Conditions")

        # Weather variable tabs
        weather_tab1, weather_tab2, weather_tab3, weather_tab4 = st.tabs([
            "Temperature",
            "Wind Speed",
            "Solar Radiation",
            "Precipitation"
        ])

        with weather_tab1:
            fig_temp = create_weather_chart(
                df=df,
                variable="temperature_2m",
                title="Temperature (2m)",
                color="#e74c3c"
            )
            st.plotly_chart(fig_temp, use_container_width=True)

        with weather_tab2:
            fig_wind = create_weather_chart(
                df=df,
                variable="wind_speed_10m",
                title="Wind Speed (10m)",
                color="#3498db"
            )
            st.plotly_chart(fig_wind, use_container_width=True)

        with weather_tab3:
            fig_solar = create_weather_chart(
                df=df,
                variable="solar_radiation_W",
                title="Solar Radiation",
                color="#f39c12"
            )
            st.plotly_chart(fig_solar, use_container_width=True)

        with weather_tab4:
            fig_precip = create_weather_chart(
                df=df,
                variable="precipitation_mm",
                title="Precipitation",
                color="#2ecc71"
            )
            st.plotly_chart(fig_precip, use_container_width=True)

        st.divider()

        # --- Price vs Weather Correlations ---
        st.subheader("Price-Weather Correlations")

        st.markdown(
            """
            Scatter plots showing the relationship between electricity prices and weather variables.
            The red dashed line indicates the linear trend.
            """
        )

        # Sample data for performance (scatter with full data can be slow)
        if len(df) > 5000:
            df_sample = df.sample(n=5000, random_state=42)
            st.caption("Showing a random sample of 5,000 data points for performance.")
        else:
            df_sample = df

        corr_col1, corr_col2 = st.columns(2)

        with corr_col1:
            fig_scatter_temp = create_price_weather_scatter(
                df=df_sample,
                weather_var="temperature_2m",
                zone=selected_zone
            )
            st.plotly_chart(fig_scatter_temp, use_container_width=True)

        with corr_col2:
            fig_scatter_wind = create_price_weather_scatter(
                df=df_sample,
                weather_var="wind_speed_10m",
                zone=selected_zone
            )
            st.plotly_chart(fig_scatter_wind, use_container_width=True)

        corr_col3, corr_col4 = st.columns(2)

        with corr_col3:
            fig_scatter_solar = create_price_weather_scatter(
                df=df_sample,
                weather_var="solar_radiation_W",
                zone=selected_zone
            )
            st.plotly_chart(fig_scatter_solar, use_container_width=True)

        with corr_col4:
            fig_scatter_precip = create_price_weather_scatter(
                df=df_sample,
                weather_var="precipitation_mm",
                zone=selected_zone
            )
            st.plotly_chart(fig_scatter_precip, use_container_width=True)

        st.divider()

        # --- Raw Data Preview ---
        with st.expander("View Raw Data"):
            st.dataframe(
                df.head(100),
                use_container_width=True
            )
            st.caption("Showing first 100 rows.")

    else:
        # Multi-Zone Comparison Mode
        st.sidebar.markdown("**Multi-Zone Comparison**")
        st.sidebar.caption("Compare prices across all three bidding zones.")

        # Get common date range
        try:
            date_ranges = {}
            for zone in VALID_ZONES:
                min_d, max_d = get_date_range(zone)
                date_ranges[zone] = (min_d.date(), max_d.date())

            # Find overlapping date range
            common_min = max(dr[0] for dr in date_ranges.values())
            common_max = min(dr[1] for dr in date_ranges.values())
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.stop()

        st.sidebar.markdown("---")
        st.sidebar.markdown("**Date Range**")

        # Default to last 30 days for comparison
        default_start = max(common_min, common_max - timedelta(days=30))

        start_date = st.sidebar.date_input(
            label="Start Date",
            value=default_start,
            min_value=common_min,
            max_value=common_max
        )

        end_date = st.sidebar.date_input(
            label="End Date",
            value=common_max,
            min_value=common_min,
            max_value=common_max
        )

        if start_date > end_date:
            st.sidebar.error("Start date must be before end date.")
            st.stop()

        # Load data for all zones
        with st.spinner("Loading data for all zones..."):
            zone_data = {}
            for zone in VALID_ZONES:
                try:
                    df = load_masterset_filtered(
                        zone=zone,
                        start_date=str(start_date),
                        end_date=str(end_date)
                    )
                    if not df.empty:
                        zone_data[zone] = df
                except Exception:
                    continue

        if not zone_data:
            st.warning("No data available for the selected date range.")
            st.stop()

        # --- Zone Overview Cards ---
        st.subheader("Zone Comparison")

        zone_cols = st.columns(3)

        for i, zone in enumerate(VALID_ZONES):
            with zone_cols[i]:
                if zone in zone_data:
                    df = zone_data[zone]
                    avg_price = df["price_eur_mwh"].mean()
                    std_price = df["price_eur_mwh"].std()

                    st.markdown(f"**{ZONE_NAMES.get(zone, zone)}**")
                    st.caption(f"{ZONE_SHORT.get(zone, '')} Market")
                    st.metric(
                        label="Avg Price",
                        value=f"{avg_price:.2f} EUR/MWh",
                        delta=f"Std: {std_price:.1f}"
                    )
                else:
                    st.markdown(f"**{ZONE_NAMES.get(zone, zone)}**")
                    st.caption("Data not available")

        st.divider()

        # --- Multi-Zone Price Chart ---
        st.subheader("Price Comparison")

        # Aggregation option
        agg_option = st.radio(
            label="Aggregation",
            options=["Hourly", "Daily Average"],
            horizontal=True,
            key="multi_agg"
        )

        if agg_option == "Daily Average":
            plot_data = {}
            for zone, df in zone_data.items():
                daily = get_daily_prices(df)
                plot_df = pd.DataFrame({"price_eur_mwh": daily["price_mean"]})
                plot_df.index = daily.index
                plot_data[zone] = plot_df
        else:
            plot_data = zone_data

        fig_multi = create_multi_zone_prices(
            data_dict=plot_data,
            title="Price Comparison Across Zones"
        )

        st.plotly_chart(fig_multi, use_container_width=True)

        st.divider()

        # --- Daily Profile Comparison ---
        st.subheader("Daily Profile Comparison")

        st.markdown(
            """
            Average hourly price profiles for each zone. Different market characteristics
            are visible in the shape of these profiles.
            """
        )

        profile_cols = st.columns(3)

        for i, zone in enumerate(VALID_ZONES):
            with profile_cols[i]:
                if zone in zone_data:
                    profile_df = get_daily_profile(zone_data[zone])
                    fig_profile = create_daily_profile_chart(
                        profile_df=profile_df,
                        zone=zone,
                        show_std=False
                    )
                    fig_profile.update_layout(height=300)
                    st.plotly_chart(fig_profile, use_container_width=True)
                else:
                    st.info(f"No data for {zone}")

        st.divider()

        # --- Statistics Table ---
        st.subheader("Summary Statistics")

        stats_data = []
        for zone in VALID_ZONES:
            if zone in zone_data:
                df = zone_data[zone]
                stats_data.append({
                    "Zone": ZONE_NAMES.get(zone, zone),
                    "Type": ZONE_SHORT.get(zone, ""),
                    "Avg Price": f"{df['price_eur_mwh'].mean():.2f}",
                    "Std Dev": f"{df['price_eur_mwh'].std():.2f}",
                    "Min": f"{df['price_eur_mwh'].min():.2f}",
                    "Max": f"{df['price_eur_mwh'].max():.2f}",
                    "Records": f"{len(df):,}"
                })

        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # --- Footer ---
    styled_footer()


# --- Entry Point ---
if __name__ == "__main__":
    main()
