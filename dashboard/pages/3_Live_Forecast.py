"""
Module: 3_Live_Forecast.py
Description: Day-ahead price forecast page with ML model predictions.
Author: MARBL Dashboard Team
Date: 2026-01-16

This page uses:
- Real weather forecasts from WeatherAPI
- Trained XGBoost Mix-of-Experts models for price prediction
"""

# --- Imports ---
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time

import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import (
    load_masterset,
    load_predictions,
    load_all_predictions,
    VALID_ZONES,
    ZONE_NAMES,
    ZONE_DESCRIPTIONS,
    ZONE_SHORT
)
from utils.visualizations import (
    create_forecast_chart,
    create_cluster_probability_bar,
    ZONE_COLORS,
    CLUSTER_COLORS
)
from utils.styles import apply_custom_styles, styled_footer
from utils.model_inference import (
    generate_real_forecast,
    check_models_available,
    ZONE_CLUSTERS
)
import plotly.graph_objects as go


# --- Page Configuration ---
st.set_page_config(
    page_title="Live Forecast - MARBL",
    page_icon="chart_with_upwards_trend",
    layout="wide"
)


# --- Data Paths ---
DATA_DIR = Path(__file__).parent.parent.parent / "data"
LIVE_DIR = DATA_DIR / "live"
MODELS_DIR = Path(__file__).parent.parent / "models"
# Fresh price data from ENTSO-E (updated more frequently than mastersets)
FRESH_PRICES_DIR = Path(__file__).parent.parent.parent / "notebooks" / "01_ingestion" / "data" / "clean_pd-pkg"


# --- Sidebar ---
def render_sidebar():
    """Render the sidebar with MARBL branding."""
    logo_path = Path(__file__).parent.parent / "assets" / "marbl_logo.png"
    if logo_path.exists():
        st.sidebar.image(str(logo_path), use_container_width=True)
    st.sidebar.markdown("---")


# --- Mock Functions (Replace with actual model inference) ---
def generate_mock_forecast(zone: str, forecast_date: datetime) -> pd.DataFrame:
    """
    Generate mock price forecast for demonstration.

    In production, this would call the trained XGBoost models.

    Parameters
    ----------
    zone : str
        Bidding zone identifier.
    forecast_date : datetime
        Date for which to generate forecast.

    Returns
    -------
    pd.DataFrame
        DataFrame with hourly price forecasts.
    """
    np.random.seed(int(forecast_date.timestamp()) % 10000 + hash(zone) % 100)

    hours = pd.date_range(
        start=forecast_date,
        periods=24,
        freq="h"
    )

    # Generate realistic price profile
    hour_of_day = np.arange(24)

    # Base pattern depends on zone
    if zone == "DK1":
        # Wind-dominated: higher variability, lower midday prices when windy
        base = 60 + 15 * np.sin((hour_of_day - 6) * np.pi / 12)
        noise = np.random.randn(24) * 15
    elif zone == "ES":
        # Solar-dominated: clear midday dip
        base = 70 + 20 * np.sin((hour_of_day - 6) * np.pi / 12)
        solar_dip = -25 * np.exp(-((hour_of_day - 13) ** 2) / 8)
        base = base + solar_dip
        noise = np.random.randn(24) * 10
    else:  # NO2
        # Hydro-dominated: more stable
        base = 55 + 10 * np.sin((hour_of_day - 6) * np.pi / 12)
        noise = np.random.randn(24) * 8

    # Day of week effect
    if forecast_date.weekday() >= 5:  # Weekend
        base = base * 0.85

    prices = np.maximum(base + noise, 0)

    df = pd.DataFrame({
        "timestamp": hours,
        "price_forecast": prices
    })
    df.set_index("timestamp", inplace=True)

    return df


def generate_mock_cluster_probabilities(zone: str, forecast_date: datetime) -> dict:
    """
    Generate mock cluster probabilities for demonstration.

    In production, this would come from the cluster classification model.

    Parameters
    ----------
    zone : str
        Bidding zone identifier.
    forecast_date : datetime
        Date for prediction.

    Returns
    -------
    dict
        Dictionary mapping cluster IDs to probabilities.
    """
    np.random.seed(int(forecast_date.timestamp()) % 10000 + hash(zone) % 100)

    n_clusters = 5

    # Generate random probabilities with some structure
    raw_probs = np.random.dirichlet(np.ones(n_clusters) * 2)

    # Make one cluster dominant
    dominant = np.argmax(raw_probs)
    raw_probs[dominant] += 0.2
    raw_probs = raw_probs / raw_probs.sum()

    return {i + 1: prob for i, prob in enumerate(raw_probs)}


def generate_mock_weather_forecast(zone: str, forecast_date: datetime) -> pd.DataFrame:
    """
    Generate mock weather forecast for demonstration (fallback).

    Used when real WeatherAPI data is not available.

    Parameters
    ----------
    zone : str
        Bidding zone identifier.
    forecast_date : datetime
        Date for forecast.

    Returns
    -------
    pd.DataFrame
        DataFrame with hourly weather forecasts.
    """
    np.random.seed(int(forecast_date.timestamp()) % 10000 + hash(zone) % 100)

    hours = pd.date_range(
        start=forecast_date,
        periods=24,
        freq="h"
    )

    hour_of_day = np.arange(24)

    # Temperature: daily cycle
    temp_base = {
        "DK1": 5,  # Denmark: colder
        "ES": 15,  # Spain: warmer
        "NO2": 2   # Norway: coldest
    }

    temperature = temp_base.get(zone, 10) + 5 * np.sin((hour_of_day - 6) * np.pi / 12)
    temperature += np.random.randn(24) * 2

    # Wind speed
    wind_base = {"DK1": 8, "ES": 4, "NO2": 5}
    wind = wind_base.get(zone, 5) + np.random.randn(24) * 2
    wind = np.maximum(wind, 0)

    # Solar radiation: daylight hours only
    solar = np.zeros(24)
    for h in range(24):
        if 6 <= h <= 18:
            solar[h] = 400 * np.sin((h - 6) * np.pi / 12) * (1 + np.random.randn() * 0.2)
    solar = np.maximum(solar, 0)

    # Precipitation
    precip = np.random.exponential(0.5, 24)
    precip[precip < 0.1] = 0

    df = pd.DataFrame({
        "timestamp": hours,
        "temperature_2m": temperature,
        "wind_speed_10m": wind,
        "solar_radiation_W": solar,
        "precipitation_mm": precip
    })
    df.set_index("timestamp", inplace=True)

    return df


def load_live_weather_forecast(zone: str) -> tuple[pd.DataFrame, bool]:
    """
    Load real weather forecast from WeatherAPI data.

    Parameters
    ----------
    zone : str
        Bidding zone identifier.

    Returns
    -------
    tuple[pd.DataFrame, bool]
        DataFrame with hourly weather forecasts and boolean indicating if real data.
    """
    forecast_path = LIVE_DIR / f"{zone}_forecast.csv"

    if not forecast_path.exists():
        return pd.DataFrame(), False

    try:
        df = pd.read_csv(forecast_path, index_col=0, parse_dates=True)
        df.index.name = "timestamp"

        # Ensure required columns exist
        required_cols = ["temperature_2m", "wind_speed_10m", "solar_radiation_W", "precipitation_mm"]
        if not all(col in df.columns for col in required_cols):
            return pd.DataFrame(), False

        return df, True

    except Exception:
        return pd.DataFrame(), False


def load_fresh_prices(zone: str, days: int = 2) -> tuple[pd.DataFrame, datetime | None]:
    """
    Load recent actual prices from fresh ENTSO-E data (clean_pd-pkg).

    Converts wide format (date x hours) to long format for plotting.

    Parameters
    ----------
    zone : str
        Bidding zone identifier.
    days : int
        Number of recent days to load.

    Returns
    -------
    tuple[pd.DataFrame, datetime | None]
        DataFrame with recent actual prices and the last data timestamp.
    """
    price_path = FRESH_PRICES_DIR / f"{zone}_preprocessed.csv"

    if not price_path.exists():
        # Fallback to masterset if fresh data not available
        try:
            df = load_masterset(zone)
            recent = df.tail(days * 24)
            if not recent.empty:
                last_timestamp = recent.index.max()
                return recent, last_timestamp
            return pd.DataFrame(), None
        except Exception:
            return pd.DataFrame(), None

    try:
        # Load wide format data (date, h00, h01, ..., h23)
        df_wide = pd.read_csv(price_path, index_col=0, parse_dates=True)

        # Get the last N days
        df_recent = df_wide.tail(days)

        if df_recent.empty:
            return pd.DataFrame(), None

        # Convert wide to long format
        records = []
        for date_idx, row in df_recent.iterrows():
            for hour in range(24):
                col_name = f"h{hour:02d}"
                if col_name in row.index:
                    timestamp = pd.Timestamp(date_idx) + pd.Timedelta(hours=hour)
                    price = row[col_name]
                    if pd.notna(price):
                        records.append({
                            "timestamp": timestamp,
                            "price_eur_mwh": price
                        })

        if not records:
            return pd.DataFrame(), None

        df_long = pd.DataFrame(records)
        df_long.set_index("timestamp", inplace=True)
        df_long.sort_index(inplace=True)

        last_timestamp = df_long.index.max()
        return df_long, last_timestamp

    except Exception:
        return pd.DataFrame(), None


def load_recent_actual_prices(zone: str, days: int = 2) -> pd.DataFrame:
    """
    Load recent actual prices for comparison (legacy wrapper).

    Parameters
    ----------
    zone : str
        Bidding zone identifier.
    days : int
        Number of recent days to load.

    Returns
    -------
    pd.DataFrame
        DataFrame with recent actual prices.
    """
    df, _ = load_fresh_prices(zone, days)
    return df


# --- Visualization Functions ---
def create_24h_forecast_chart(
    forecast: pd.DataFrame,
    actual: pd.DataFrame,
    zone: str
) -> go.Figure:
    """
    Create chart showing 24-hour forecast with recent actuals.

    Parameters
    ----------
    forecast : pd.DataFrame
        Forecasted prices.
    actual : pd.DataFrame
        Actual recent prices.
    zone : str
        Zone identifier.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    fig = go.Figure()

    # Actual prices (if available)
    if not actual.empty:
        fig.add_trace(
            go.Scatter(
                x=actual.index,
                y=actual["price_eur_mwh"],
                mode="lines",
                name="Actual",
                line=dict(color=ZONE_COLORS.get(zone, "#1f77b4"), width=2),
                hovertemplate="Time: %{x}<br>Actual: %{y:.2f} EUR/MWh<extra></extra>"
            )
        )

    # Forecast prices
    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast["price_forecast"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#d62728", width=2, dash="dash"),
            marker=dict(size=6),
            hovertemplate="Time: %{x}<br>Forecast: %{y:.2f} EUR/MWh<extra></extra>"
        )
    )

    # Add vertical line at forecast start
    if not actual.empty and not forecast.empty:
        # Use add_shape instead of add_vline to avoid Plotly/pandas compatibility issues
        forecast_start = forecast.index[0]
        fig.add_shape(
            type="line",
            x0=forecast_start,
            x1=forecast_start,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="gray", dash="dot", width=1)
        )
        fig.add_annotation(
            x=forecast_start,
            y=1.05,
            yref="paper",
            text="Forecast Start",
            showarrow=False,
            font=dict(size=10, color="gray")
        )

    fig.update_layout(
        title=f"Day-Ahead Price Forecast - {ZONE_NAMES.get(zone, zone)}",
        xaxis_title="Time",
        yaxis_title="Price (EUR/MWh)",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        height=450,
        margin=dict(r=20)  # Add right margin to prevent cutoff
    )

    return fig


def create_weather_forecast_chart(weather: pd.DataFrame) -> go.Figure:
    """
    Create multi-panel chart showing weather forecast.

    Parameters
    ----------
    weather : pd.DataFrame
        Weather forecast data.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Temperature", "Wind Speed", "Solar Radiation", "Precipitation"),
        vertical_spacing=0.18,
        horizontal_spacing=0.12
    )

    # Temperature
    fig.add_trace(
        go.Scatter(
            x=weather.index,
            y=weather["temperature_2m"],
            mode="lines",
            name="Temperature",
            line=dict(color="#e74c3c"),
            showlegend=False
        ),
        row=1, col=1
    )

    # Wind
    fig.add_trace(
        go.Scatter(
            x=weather.index,
            y=weather["wind_speed_10m"],
            mode="lines",
            name="Wind",
            line=dict(color="#3498db"),
            showlegend=False
        ),
        row=1, col=2
    )

    # Solar
    fig.add_trace(
        go.Scatter(
            x=weather.index,
            y=weather["solar_radiation_W"],
            mode="lines",
            fill="tozeroy",
            name="Solar",
            line=dict(color="#f39c12"),
            showlegend=False
        ),
        row=2, col=1
    )

    # Precipitation
    fig.add_trace(
        go.Bar(
            x=weather.index,
            y=weather["precipitation_mm"],
            name="Precipitation",
            marker_color="#2ecc71",
            showlegend=False
        ),
        row=2, col=2
    )

    # Update axes labels
    fig.update_yaxes(title_text="Celsius", row=1, col=1)
    fig.update_yaxes(title_text="m/s", row=1, col=2)
    fig.update_yaxes(title_text="W/m2", row=2, col=1)
    fig.update_yaxes(title_text="mm", row=2, col=2)

    fig.update_layout(
        title="Weather Forecast (Model Input)",
        template="plotly_white",
        height=550,
        showlegend=False,
        margin=dict(t=80)  # More space for subplot titles
    )

    return fig


# --- Main Page ---
def main():
    """Render the live forecast page."""

    # Apply custom CSS styles
    apply_custom_styles()

    render_sidebar()

    st.title("Day-Ahead Price Forecast")
    st.markdown(
        """
        Predicted electricity prices for tomorrow based on weather forecasts
        and the XGBoost Mix-of-Experts model.
        """
    )

    # Info box about data status (will be updated after loading weather)
    info_placeholder = st.empty()

    st.divider()

    # --- Sidebar Controls ---
    st.sidebar.header("Settings")

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

    # Forecast date (default to tomorrow)
    forecast_date = st.sidebar.date_input(
        label="Forecast Date",
        value=datetime.now().date() + timedelta(days=1)
    )

    forecast_datetime = datetime.combine(forecast_date, datetime.min.time())

    st.sidebar.markdown("---")

    # Auto-refresh toggle
    auto_refresh = st.sidebar.toggle(
        label="Auto-Refresh",
        value=False,
        help="Automatically refresh forecasts every 5 minutes"
    )

    refresh_interval = st.sidebar.selectbox(
        label="Refresh Interval",
        options=[1, 5, 10, 30],
        index=1,
        format_func=lambda x: f"{x} minute{'s' if x > 1 else ''}",
        disabled=not auto_refresh
    )

    # Manual refresh button
    if st.sidebar.button("Refresh Now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    # Auto-refresh logic
    if auto_refresh:
        st.sidebar.success(f"Auto-refresh: {refresh_interval} min")
        time.sleep(0.1)  # Small delay to ensure UI renders
        # Use st.empty() placeholder for countdown (optional)
        # Trigger rerun after interval
        time.sleep(refresh_interval * 60)
        st.rerun()

    # Last updated timestamp
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

    # --- Generate Forecasts ---
    with st.spinner("Generating forecast..."):
        # Load fresh price data (last 4 days for lagged features + visualization)
        recent_actual, last_price_timestamp = load_fresh_prices(selected_zone, days=4)

        # Try to load real weather forecast, fall back to mock if unavailable
        weather_forecast, is_real_weather = load_live_weather_forecast(selected_zone)
        if not is_real_weather or weather_forecast.empty:
            weather_forecast = generate_mock_weather_forecast(selected_zone, forecast_datetime)
            is_real_weather = False

        # Generate ML-based forecast using trained models
        price_forecast, cluster_probs, is_real_prediction = generate_real_forecast(
            zone=selected_zone,
            weather_forecast=weather_forecast,
            recent_prices=recent_actual,
            forecast_date=pd.Timestamp(forecast_datetime)
        )

        # Filter recent_actual for visualization (last 48h before forecast)
        if not recent_actual.empty:
            cutoff_time = forecast_datetime - timedelta(hours=48)
            # Handle timezone-aware vs naive comparison
            if recent_actual.index.tz is not None:
                # Index is timezone-aware, make cutoff_time aware too
                cutoff_time = pd.Timestamp(cutoff_time).tz_localize(recent_actual.index.tz)
            recent_actual = recent_actual[recent_actual.index >= cutoff_time]

    # Calculate data freshness
    is_price_fresh = False
    price_age_hours = None
    if last_price_timestamp is not None:
        price_age = datetime.now() - last_price_timestamp.to_pydatetime().replace(tzinfo=None)
        price_age_hours = price_age.total_seconds() / 3600
        is_price_fresh = price_age_hours < 48  # Consider fresh if less than 48h old

    # Update info message based on data status
    status_parts = []
    if is_real_weather:
        status_parts.append("Weather: live (WeatherAPI)")
    else:
        status_parts.append("Weather: mock data")

    if is_price_fresh and last_price_timestamp:
        status_parts.append(f"Prices: real (until {last_price_timestamp.strftime('%Y-%m-%d %H:%M')})")
    elif last_price_timestamp:
        status_parts.append(f"Prices: stale ({price_age_hours:.0f}h old)")
    else:
        status_parts.append("Prices: unavailable")

    if is_real_prediction:
        status_parts.append("Price Forecast: ML model")
    else:
        status_parts.append("Price Forecast: simulated")

    if is_real_weather and is_price_fresh:
        info_placeholder.success(" | ".join(status_parts))
    elif is_real_weather or is_price_fresh:
        info_placeholder.warning(" | ".join(status_parts))
    else:
        info_placeholder.info(" | ".join(status_parts))

    # --- Summary Metrics ---
    st.subheader("Forecast Summary")

    col1, col2, col3, col4 = st.columns(4)

    avg_price = price_forecast["price_forecast"].mean()
    min_price = price_forecast["price_forecast"].min()
    max_price = price_forecast["price_forecast"].max()
    predicted_cluster = max(cluster_probs, key=cluster_probs.get)

    with col1:
        st.metric(
            label="Average Price",
            value=f"{avg_price:.2f} EUR/MWh"
        )

    with col2:
        st.metric(
            label="Price Range (EUR/MWh)",
            value=f"{min_price:.1f} - {max_price:.1f}"
        )

    with col3:
        st.metric(
            label="Predicted Cluster",
            value=f"Cluster {predicted_cluster}"
        )

    with col4:
        st.metric(
            label="Forecast Date",
            value=forecast_date.strftime("%Y-%m-%d")
        )

    st.divider()

    # --- Price Forecast Chart ---
    st.subheader("Hourly Price Forecast")

    fig_forecast = create_24h_forecast_chart(
        forecast=price_forecast,
        actual=recent_actual,
        zone=selected_zone
    )

    st.plotly_chart(fig_forecast, use_container_width=True)

    # Hourly breakdown table
    with st.expander("View Hourly Forecast Data"):
        display_df = price_forecast.copy()
        display_df["hour"] = display_df.index.hour
        display_df = display_df.rename(columns={"price_forecast": "Price (EUR/MWh)"})
        display_df["Price (EUR/MWh)"] = display_df["Price (EUR/MWh)"].round(2)
        st.dataframe(display_df[["hour", "Price (EUR/MWh)"]], use_container_width=True)

    st.divider()

    # --- Cluster Prediction ---
    st.subheader("Cluster Prediction")

    col_cluster1, col_cluster2 = st.columns([3, 2])

    with col_cluster1:
        st.markdown("**Predicted Cluster Probabilities**")
        st.markdown(
            f"""
            The model predicts that tomorrow's price pattern will most likely
            follow **Cluster {predicted_cluster}** with {cluster_probs[predicted_cluster]:.1%} probability.
            """
        )

        fig_probs = create_cluster_probability_bar(
            probabilities=cluster_probs,
            title="Cluster Probabilities"
        )
        st.plotly_chart(fig_probs, use_container_width=True)

    with col_cluster2:
        st.markdown("**What This Means**")

        # Zone-specific cluster descriptions based on analysis
        cluster_descriptions_dk = {
            1: "Low volatility - stable prices with moderate renewable generation",
            2: "Morning peak - high demand during business start hours",
            3: "High wind - low prices due to strong wind generation",
            4: "Evening peak - residential demand surge after work",
            5: "Moderate variability - typical weekday pattern",
            6: "Weekend/holiday - lower overall demand pattern"
        }
        cluster_descriptions_es = {
            1: "Solar dip - low midday prices from high solar generation",
            2: "Evening peak - demand surge as solar declines",
            3: "Flat profile - balanced supply and demand"
        }
        cluster_descriptions_no = {
            1: "Low hydro - higher prices due to limited water availability",
            2: "High hydro - low prices from abundant hydro generation",
            3: "Evening peak - residential heating demand",
            4: "Stable profile - balanced Nordic grid",
            5: "Export mode - prices affected by interconnector flows"
        }

        zone_descriptions = {
            "DK1": cluster_descriptions_dk,
            "ES": cluster_descriptions_es,
            "NO2": cluster_descriptions_no
        }

        descriptions = zone_descriptions.get(selected_zone, {})

        for cluster_id, prob in sorted(cluster_probs.items(), key=lambda x: -x[1]):
            if prob > 0.1:
                desc = descriptions.get(cluster_id, "Price pattern cluster")
                st.markdown(f"- **Cluster {cluster_id}** ({prob:.1%}): {desc}")

    st.divider()

    # --- Weather Input ---
    st.subheader("Weather Forecast Input")

    if is_real_weather:
        st.markdown(
            f"""
            **Live weather forecasts** from WeatherAPI.com for {ZONE_NAMES.get(selected_zone, selected_zone)}.
            Data covers {weather_forecast.index.min().strftime('%Y-%m-%d %H:%M')} to {weather_forecast.index.max().strftime('%Y-%m-%d %H:%M')}.
            """
        )
    else:
        st.markdown(
            """
            Weather forecasts used as input features for the price prediction model.
            *(Mock data shown - run live forecast notebook for real data)*
            """
        )

    fig_weather = create_weather_forecast_chart(weather_forecast)
    st.plotly_chart(fig_weather, use_container_width=True)

    # Weather summary
    col_w1, col_w2, col_w3, col_w4 = st.columns(4)

    with col_w1:
        st.metric(
            label="Avg Temperature",
            value=f"{weather_forecast['temperature_2m'].mean():.1f} C"
        )

    with col_w2:
        st.metric(
            label="Avg Wind Speed",
            value=f"{weather_forecast['wind_speed_10m'].mean():.1f} m/s"
        )

    with col_w3:
        st.metric(
            label="Total Solar",
            value=f"{weather_forecast['solar_radiation_W'].sum():.0f} Wh/m2"
        )

    with col_w4:
        st.metric(
            label="Total Precipitation",
            value=f"{weather_forecast['precipitation_mm'].sum():.1f} mm"
        )

    st.divider()

    # --- Model Information ---
    with st.expander("Model Information"):
        st.markdown(
            """
            **Forecasting Methodology**

            The day-ahead price forecast uses a Mix-of-Experts approach:

            1. **Cluster Classification**: An XGBoost classifier predicts which price pattern
               (cluster) is most likely for the forecast day based on weather forecasts,
               lagged prices, and calendar features.

            2. **Cluster-Specific Regression**: Separate XGBoost regression models are trained
               for each cluster to predict hourly prices given the specific pattern type.

            3. **Weighted Combination**: Final hourly predictions are calculated as a
               probability-weighted average of all cluster model predictions.

            **Input Features**:
            - Weather: temperature, wind speed, solar radiation, precipitation
            - Lagged prices: previous 1, 2, and 3 day prices
            - Calendar: day of week, month, holiday indicators

            **Data Sources**:
            - Historical prices: ENTSO-E Transparency Platform
            - Historical weather: ERA5 Reanalysis (Copernicus)
            - Weather forecasts: WeatherAPI.com
            """
        )

    st.divider()

    # --- Model Performance (Test Set) ---
    st.subheader("Model Performance (Test Set)")

    st.markdown(
        """
        Performance metrics from the test set (June-November 2025) comparing three forecasting approaches.
        """
    )

    try:
        predictions = load_all_predictions(selected_zone)

        if predictions:
            # Calculate metrics for each approach
            approach_names = {
                "cluster": "Cluster (Mix-of-Experts)",
                "naive_one": "Naive One (Baseline)",
                "naive_two": "Naive Two (XGBoost)"
            }

            metrics_data = []
            for approach, df in predictions.items():
                mae = (df["price_real"] - df["price_predicted"]).abs().mean()
                rmse = np.sqrt(((df["price_real"] - df["price_predicted"]) ** 2).mean())
                mape = ((df["price_real"] - df["price_predicted"]).abs() / df["price_real"].abs().clip(lower=1)).mean() * 100

                metrics_data.append({
                    "Approach": approach_names.get(approach, approach),
                    "MAE (EUR/MWh)": f"{mae:.2f}",
                    "RMSE (EUR/MWh)": f"{rmse:.2f}",
                    "MAPE (%)": f"{mape:.1f}%",
                    "Test Samples": len(df)
                })

            # Display metrics table
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            # Actual vs Predicted chart for best model
            with st.expander("View Actual vs Predicted (Cluster Approach)"):
                if "cluster" in predictions:
                    df_cluster = predictions["cluster"].tail(168)  # Last week

                    fig_perf = go.Figure()
                    fig_perf.add_trace(
                        go.Scatter(
                            x=df_cluster.index,
                            y=df_cluster["price_real"],
                            mode="lines",
                            name="Actual",
                            line=dict(color=ZONE_COLORS.get(selected_zone, "#1f77b4"), width=2)
                        )
                    )
                    fig_perf.add_trace(
                        go.Scatter(
                            x=df_cluster.index,
                            y=df_cluster["price_predicted"],
                            mode="lines",
                            name="Predicted",
                            line=dict(color="#d62728", width=2, dash="dash")
                        )
                    )
                    fig_perf.update_layout(
                        title="Actual vs Predicted Prices (Last Week of Test Set)",
                        xaxis_title="Time",
                        yaxis_title="Price (EUR/MWh)",
                        template="plotly_white",
                        height=400,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02)
                    )
                    st.plotly_chart(fig_perf, use_container_width=True)

        else:
            st.info("Model predictions not available. Run the prediction notebooks to generate test results.")

    except Exception as e:
        st.info(f"Could not load model predictions: {e}")

    # --- Footer ---
    styled_footer()


# --- Entry Point ---
if __name__ == "__main__":
    main()
