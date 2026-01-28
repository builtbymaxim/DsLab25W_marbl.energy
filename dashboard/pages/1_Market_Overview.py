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
    load_hourly_prices,
    load_all_mastersets,
    get_daily_prices,
    get_daily_profile,
    get_data_summary,
    get_date_range,
    get_price_date_range,
    get_daily_mean_history,
    VALID_ZONES,
    ZONE_NAMES,
    ZONE_DESCRIPTIONS,
    ZONE_SHORT
)
from utils.visualizations import (
    create_price_timeseries,
    create_multi_zone_prices,
    create_daily_profile_chart,
    create_daily_mean_history_chart,
    create_multi_zone_daily_mean_chart,
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

        # Get available date range for selected zone (using full price range)
        try:
            min_date, max_date = get_price_date_range(selected_zone)
            min_date = min_date.date() if hasattr(min_date, 'date') else min_date
            max_date = max_date.date() if hasattr(max_date, 'date') else max_date
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

        # Note about weather data limitation
        try:
            _, masterset_max = get_date_range(selected_zone)
            masterset_max_date = masterset_max.date()
            if end_date > masterset_max_date:
                st.sidebar.info(
                    f"Note: Weather data available through {masterset_max_date}. "
                    "Price-weather charts limited to this range."
                )
        except Exception:
            pass

        # --- Load Data ---
        with st.spinner("Loading data..."):
            # Load price data (full range, not limited by weather)
            try:
                price_df = load_hourly_prices(
                    zone=selected_zone,
                    start_date=str(start_date),
                    end_date=str(end_date)
                )
            except Exception as e:
                st.error(f"Error loading price data: {str(e)}")
                st.stop()

            # Load masterset (price + weather, limited range)
            try:
                df = load_masterset_filtered(
                    zone=selected_zone,
                    start_date=str(start_date),
                    end_date=str(end_date)
                )
            except Exception:
                df = pd.DataFrame()  # Empty if masterset fails

        # Check if price data is available
        if price_df.empty:
            st.warning("No price data available for the selected date range.")
            styled_footer()
            st.stop()

        # Flag for weather data availability
        has_weather_data = not df.empty

        # --- Export Button ---
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Export Data**")

        csv_data = convert_df_to_csv(price_df)
        st.sidebar.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"{selected_zone}_prices_{start_date}_{end_date}.csv",
            mime="text/csv",
            use_container_width=True
        )

        # --- Summary Statistics ---
        st.subheader("Summary Statistics")

        # Calculate summary from price data
        price_mean = price_df["price_eur_mwh"].mean()
        price_std = price_df["price_eur_mwh"].std()
        price_min = price_df["price_eur_mwh"].min()
        price_max = price_df["price_eur_mwh"].max()
        total_records = len(price_df)
        negative_count = (price_df["price_eur_mwh"] < 0).sum()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Average Price",
                value=f"{price_mean:.2f} EUR/MWh"
            )

        with col2:
            # Highlight negative minimum in red
            min_display = f"{price_min:.2f}"
            max_display = f"{price_max:.2f}"
            if price_min < 0:
                st.metric(
                    label="Price Range (EUR/MWh)",
                    value=f"{min_display} - {max_display}"
                )
                st.caption(f":red[{int(negative_count)} negative price hours]")
            else:
                st.metric(
                    label="Price Range (EUR/MWh)",
                    value=f"{min_display} - {max_display}"
                )

        with col3:
            st.metric(
                label="Std Deviation",
                value=f"{price_std:.2f} EUR/MWh"
            )

        with col4:
            st.metric(
                label="Data Points",
                value=format_number(total_records, 0)
            )

        st.divider()

        # --- Historical Daily Mean Prices ---
        st.subheader("Historical Daily Mean Prices")

        st.markdown(
            """
            Long-term view of daily average electricity prices across all available years.
            Use the range selector buttons or slider to zoom in on specific periods.
            """
        )

        # Load full preprocessed data for daily mean chart
        try:
            daily_mean_df = get_daily_mean_history(selected_zone)

            fig_daily_mean = create_daily_mean_history_chart(
                df=daily_mean_df,
                zone=selected_zone,
                title=f"Daily Mean Price History - {ZONE_NAMES.get(selected_zone, selected_zone)}"
            )

            st.plotly_chart(fig_daily_mean, use_container_width=True)

        except FileNotFoundError:
            st.info("Historical daily mean data not available for this zone.")
        except Exception as e:
            st.warning(f"Could not load historical data: {str(e)}")

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
            daily_df = get_daily_prices(price_df)
            # Reshape for plotting
            plot_df = pd.DataFrame({
                "price_eur_mwh": daily_df["price_mean"]
            })
            plot_df.index = daily_df.index
        else:
            plot_df = price_df

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

        profile_df = get_daily_profile(price_df)

        fig_profile = create_daily_profile_chart(
            profile_df=profile_df,
            zone=selected_zone,
            show_std=True
        )

        st.plotly_chart(fig_profile, use_container_width=True)

        st.divider()

        # --- Weather Data ---
        st.subheader("Weather Conditions")

        if has_weather_data:
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
        else:
            st.info(
                "Weather data (ERA5) is available through late November 2025. "
                "Select an earlier date range to view weather conditions."
            )

        st.divider()

        # --- Price vs Weather Correlations ---
        st.subheader("Price-Weather Correlations")

        if has_weather_data:
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
        else:
            st.info(
                "Weather data (ERA5) is available through late November 2025. "
                "Select an earlier date range to view price-weather correlations."
            )

        st.divider()

        # --- Raw Data Preview ---
        with st.expander("View Raw Data"):
            # Show price data with negative values highlighted in red
            display_df = price_df.head(100).copy()
            display_df = display_df.reset_index()
            display_df.columns = ["Timestamp", "Price (EUR/MWh)"]

            def highlight_negative(val):
                if isinstance(val, (int, float)) and val < 0:
                    return "color: red; font-weight: bold"
                return ""

            styled_df = display_df.style.applymap(
                highlight_negative,
                subset=["Price (EUR/MWh)"]
            )
            st.dataframe(styled_df, use_container_width=True)
            st.caption("Showing first 100 rows. :red[Negative prices highlighted in red.]")

    else:
        # Multi-Zone Comparison Mode
        st.sidebar.markdown("**Multi-Zone Comparison**")
        st.sidebar.caption("Compare prices across all three bidding zones.")

        # Get common date range (using full price range)
        try:
            date_ranges = {}
            for zone in VALID_ZONES:
                min_d, max_d = get_price_date_range(zone)
                min_d = min_d.date() if hasattr(min_d, 'date') else min_d
                max_d = max_d.date() if hasattr(max_d, 'date') else max_d
                date_ranges[zone] = (min_d, max_d)

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

        # Load price data for all zones (full range, not limited by weather)
        with st.spinner("Loading data for all zones..."):
            zone_price_data = {}
            for zone in VALID_ZONES:
                try:
                    price_df = load_hourly_prices(
                        zone=zone,
                        start_date=str(start_date),
                        end_date=str(end_date)
                    )
                    if not price_df.empty:
                        zone_price_data[zone] = price_df
                except Exception:
                    continue

        if not zone_price_data:
            st.warning("No data available for the selected date range.")
            st.stop()

        # --- Zone Overview Cards ---
        st.subheader("Zone Comparison")

        zone_cols = st.columns(3)

        for i, zone in enumerate(VALID_ZONES):
            with zone_cols[i]:
                if zone in zone_price_data:
                    zdf = zone_price_data[zone]
                    avg_price = zdf["price_eur_mwh"].mean()
                    std_price = zdf["price_eur_mwh"].std()
                    neg_count = (zdf["price_eur_mwh"] < 0).sum()

                    st.markdown(f"**{ZONE_NAMES.get(zone, zone)}**")
                    st.caption(f"{ZONE_SHORT.get(zone, '')} Market")
                    st.metric(
                        label="Avg Price",
                        value=f"{avg_price:.2f} EUR/MWh",
                        delta=f"Std: {std_price:.1f}"
                    )
                    if neg_count > 0:
                        st.caption(f":red[{int(neg_count)} negative hours]")
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
            for zone, zdf in zone_price_data.items():
                daily = get_daily_prices(zdf)
                plot_df = pd.DataFrame({"price_eur_mwh": daily["price_mean"]})
                plot_df.index = daily.index
                plot_data[zone] = plot_df
        else:
            plot_data = zone_price_data

        fig_multi = create_multi_zone_prices(
            data_dict=plot_data,
            title="Price Comparison Across Zones"
        )

        st.plotly_chart(fig_multi, use_container_width=True)

        st.divider()

        # --- Historical Daily Mean Comparison ---
        st.subheader("Historical Daily Mean Prices")

        st.markdown(
            """
            Long-term view of daily average prices across all zones (~10 years of data).
            Use the range selector to zoom in on specific periods.
            """
        )

        # Load full preprocessed data for all zones
        try:
            daily_mean_data = {}
            for zone in VALID_ZONES:
                try:
                    daily_mean_data[zone] = get_daily_mean_history(zone)
                except FileNotFoundError:
                    continue

            if daily_mean_data:
                fig_daily_mean_multi = create_multi_zone_daily_mean_chart(
                    data_dict=daily_mean_data,
                    title="Daily Mean Price Comparison (Historical)"
                )
                st.plotly_chart(fig_daily_mean_multi, use_container_width=True)
            else:
                st.info("Historical daily mean data not available.")

        except Exception as e:
            st.warning(f"Could not load historical data: {str(e)}")

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
                if zone in zone_price_data:
                    profile_df = get_daily_profile(zone_price_data[zone])
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
            if zone in zone_price_data:
                zdf = zone_price_data[zone]
                min_price = zdf['price_eur_mwh'].min()
                neg_count = (zdf['price_eur_mwh'] < 0).sum()
                stats_data.append({
                    "Zone": ZONE_NAMES.get(zone, zone),
                    "Type": ZONE_SHORT.get(zone, ""),
                    "Avg Price": zdf['price_eur_mwh'].mean(),
                    "Std Dev": zdf['price_eur_mwh'].std(),
                    "Min": min_price,
                    "Max": zdf['price_eur_mwh'].max(),
                    "Negative Hours": int(neg_count),
                    "Records": len(zdf)
                })

        if stats_data:
            stats_df = pd.DataFrame(stats_data)

            # Style negative values in red
            def highlight_negative(val):
                if isinstance(val, (int, float)) and val < 0:
                    return "color: red; font-weight: bold"
                return ""

            styled_stats = stats_df.style.applymap(
                highlight_negative,
                subset=["Min"]
            ).format({
                "Avg Price": "{:.2f}",
                "Std Dev": "{:.2f}",
                "Min": "{:.2f}",
                "Max": "{:.2f}",
                "Records": "{:,}"
            })

            st.dataframe(styled_stats, use_container_width=True, hide_index=True)

    # --- Footer ---
    styled_footer()


# --- Entry Point ---
if __name__ == "__main__":
    main()
