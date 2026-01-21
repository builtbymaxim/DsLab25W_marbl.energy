"""
Module: visualizations.py
Description: Reusable Plotly chart functions for the MARBL dashboard.
Author: MARBL Dashboard Team
Date: 2026-01-16
"""

# --- Imports ---
from typing import Optional, List

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# --- Color Constants ---
# MARBL brand primary color
MARBL_PRIMARY = "#1a1a1a"
MARBL_SECONDARY = "#4a4a4a"
MARBL_ACCENT = "#666666"

# Consistent color scheme for zones
ZONE_COLORS = {
    "DK1": "#2563eb",  # Blue - Wind
    "ES": "#dc2626",   # Red - Solar
    "NO2": "#059669"   # Green - Hydro
}

# Color palette for clusters
CLUSTER_COLORS = [
    "#3b82f6",  # Blue
    "#ef4444",  # Red
    "#10b981",  # Green
    "#f59e0b",  # Amber
    "#8b5cf6",  # Purple
    "#ec4899",  # Pink
    "#06b6d4",  # Cyan
    "#84cc16"   # Lime
]


# --- Price Charts ---
def create_price_timeseries(
    df: pd.DataFrame,
    zone: str,
    title: Optional[str] = None
) -> go.Figure:
    """
    Create a time series chart of electricity prices.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index and 'price_eur_mwh' column.
    zone : str
        Zone identifier for color coding.
    title : str, optional
        Chart title. Defaults to 'Electricity Prices - {zone}'.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    if title is None:
        title = f"Electricity Prices - {zone}"

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["price_eur_mwh"],
            mode="lines",
            name=zone,
            line=dict(color=ZONE_COLORS.get(zone, "#1f77b4"), width=1),
            hovertemplate="Date: %{x}<br>Price: %{y:.2f} EUR/MWh<extra></extra>"
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price (EUR/MWh)",
        template="plotly_white",
        hovermode="x unified",
        height=400
    )

    return fig


def create_multi_zone_prices(
    data_dict: dict,
    title: str = "Price Comparison Across Zones"
) -> go.Figure:
    """
    Create a time series chart comparing prices across multiple zones.

    Parameters
    ----------
    data_dict : dict
        Dictionary mapping zone codes to DataFrames.
    title : str
        Chart title.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    fig = go.Figure()

    for zone, df in data_dict.items():
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["price_eur_mwh"],
                mode="lines",
                name=zone,
                line=dict(color=ZONE_COLORS.get(zone, "#666666"), width=1),
                hovertemplate=f"{zone}<br>Date: %{{x}}<br>Price: %{{y:.2f}} EUR/MWh<extra></extra>"
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price (EUR/MWh)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450
    )

    return fig


def create_daily_profile_chart(
    profile_df: pd.DataFrame,
    zone: str,
    show_std: bool = True
) -> go.Figure:
    """
    Create a chart showing the average daily price profile.

    Parameters
    ----------
    profile_df : pd.DataFrame
        DataFrame with hour index and 'price_mean', 'price_std' columns.
    zone : str
        Zone identifier for labeling.
    show_std : bool
        Whether to show standard deviation bands.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    fig = go.Figure()

    # Add standard deviation band if requested
    if show_std and "price_std" in profile_df.columns:
        upper = profile_df["price_mean"] + profile_df["price_std"]
        lower = profile_df["price_mean"] - profile_df["price_std"]

        fig.add_trace(
            go.Scatter(
                x=list(profile_df.index) + list(profile_df.index[::-1]),
                y=list(upper) + list(lower[::-1]),
                fill="toself",
                fillcolor="rgba(31, 119, 180, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Std Dev",
                showlegend=True
            )
        )

    # Add mean line
    fig.add_trace(
        go.Scatter(
            x=profile_df.index,
            y=profile_df["price_mean"],
            mode="lines+markers",
            name="Average Price",
            line=dict(color=ZONE_COLORS.get(zone, "#1f77b4"), width=2),
            marker=dict(size=6),
            hovertemplate="Hour: %{x}<br>Price: %{y:.2f} EUR/MWh<extra></extra>"
        )
    )

    fig.update_layout(
        title=f"Average Daily Price Profile - {zone}",
        xaxis_title="Hour of Day",
        yaxis_title="Price (EUR/MWh)",
        template="plotly_white",
        xaxis=dict(tickmode="linear", dtick=2),
        height=400
    )

    return fig


# --- Weather Charts ---
def create_weather_chart(
    df: pd.DataFrame,
    variable: str,
    title: Optional[str] = None,
    color: str = "#1f77b4"
) -> go.Figure:
    """
    Create a time series chart for a weather variable.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index and weather columns.
    variable : str
        Column name of the weather variable.
    title : str, optional
        Chart title.
    color : str
        Line color.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    # Variable labels for display
    var_labels = {
        "temperature_2m": "Temperature (C)",
        "wind_speed_10m": "Wind Speed (m/s)",
        "precipitation_mm": "Precipitation (mm)",
        "solar_radiation_W": "Solar Radiation (W/m2)"
    }

    ylabel = var_labels.get(variable, variable)

    if title is None:
        title = ylabel

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[variable],
            mode="lines",
            name=variable,
            line=dict(color=color, width=1),
            hovertemplate=f"Date: %{{x}}<br>{ylabel}: %{{y:.2f}}<extra></extra>"
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=ylabel,
        template="plotly_white",
        height=300
    )

    return fig


def create_price_weather_scatter(
    df: pd.DataFrame,
    weather_var: str,
    zone: str
) -> go.Figure:
    """
    Create a scatter plot of price vs weather variable.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price and weather columns.
    weather_var : str
        Weather variable column name.
    zone : str
        Zone identifier.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    # Variable labels for display
    var_labels = {
        "temperature_2m": "Temperature (C)",
        "wind_speed_10m": "Wind Speed (m/s)",
        "precipitation_mm": "Precipitation (mm)",
        "solar_radiation_W": "Solar Radiation (W/m2)"
    }

    xlabel = var_labels.get(weather_var, weather_var)

    fig = px.scatter(
        df,
        x=weather_var,
        y="price_eur_mwh",
        opacity=0.3,
        color_discrete_sequence=[ZONE_COLORS.get(zone, "#1f77b4")],
        title=f"Price vs {xlabel} - {zone}"
    )

    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title="Price (EUR/MWh)",
        template="plotly_white",
        height=400
    )

    # Add trendline
    if len(df) > 10:
        z = np.polyfit(df[weather_var].dropna(), df.loc[df[weather_var].notna(), "price_eur_mwh"], 1)
        p = np.poly1d(z)
        x_range = np.linspace(df[weather_var].min(), df[weather_var].max(), 100)

        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=p(x_range),
                mode="lines",
                name="Trend",
                line=dict(color="red", dash="dash", width=2)
            )
        )

    return fig


# --- Cluster Charts ---
def create_cluster_centroids_chart(
    centroids: pd.DataFrame,
    title: str = "Cluster Centroid Profiles"
) -> go.Figure:
    """
    Create a chart showing cluster centroid profiles (24-hour patterns).

    Parameters
    ----------
    centroids : pd.DataFrame
        DataFrame with cluster IDs as columns and hours (0-23) as index.
        Each column contains the average hourly price for that cluster.
    title : str
        Chart title.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    fig = go.Figure()

    for i, col in enumerate(centroids.columns):
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]

        fig.add_trace(
            go.Scatter(
                x=centroids.index,
                y=centroids[col],
                mode="lines+markers",
                name=f"Cluster {col}",
                line=dict(color=color, width=2),
                marker=dict(size=5),
                hovertemplate=f"Cluster {col}<br>Hour: %{{x}}<br>Price: %{{y:.2f}} EUR/MWh<extra></extra>"
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Hour of Day",
        yaxis_title="Price (EUR/MWh)",
        template="plotly_white",
        xaxis=dict(tickmode="linear", dtick=2),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450
    )

    return fig


def create_cluster_calendar_heatmap(
    df: pd.DataFrame,
    cluster_col: str = "cluster",
    year: Optional[int] = None
) -> go.Figure:
    """
    Create a calendar heatmap showing cluster assignments by date.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index and cluster column.
    cluster_col : str
        Name of the cluster column.
    year : int, optional
        Year to display. If None, shows all data.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    # Prepare daily data
    daily = df.copy()

    # Reset index to avoid ambiguity if index is named 'date'
    if daily.index.name == "date":
        daily = daily.reset_index()
    elif "date" not in daily.columns:
        daily["date"] = daily.index.date

    # Ensure date column exists and aggregate to daily
    if "date" in daily.columns:
        daily = daily.groupby("date")[cluster_col].first().reset_index()
    daily["date"] = pd.to_datetime(daily["date"])

    # Filter by year if specified
    if year is not None:
        daily = daily[daily["date"].dt.year == year]

    # Create calendar layout
    daily["week"] = daily["date"].dt.isocalendar().week
    daily["weekday"] = daily["date"].dt.weekday

    fig = go.Figure(
        data=go.Heatmap(
            x=daily["week"],
            y=daily["weekday"],
            z=daily[cluster_col],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Cluster")
        )
    )

    fig.update_layout(
        title=f"Cluster Calendar {year if year else ''}",
        xaxis_title="Week of Year",
        yaxis_title="Day of Week",
        yaxis=dict(
            tickmode="array",
            tickvals=[0, 1, 2, 3, 4, 5, 6],
            ticktext=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        ),
        template="plotly_white",
        height=300
    )

    return fig


# --- Forecast Charts ---
def create_forecast_chart(
    actual: pd.Series,
    forecast: pd.Series,
    zone: str,
    title: str = "Day-Ahead Price Forecast"
) -> go.Figure:
    """
    Create a chart comparing actual prices with forecasted prices.

    Parameters
    ----------
    actual : pd.Series
        Actual prices with datetime index.
    forecast : pd.Series
        Forecasted prices with datetime index.
    zone : str
        Zone identifier.
    title : str
        Chart title.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    fig = go.Figure()

    # Actual prices
    fig.add_trace(
        go.Scatter(
            x=actual.index,
            y=actual.values,
            mode="lines",
            name="Actual",
            line=dict(color=ZONE_COLORS.get(zone, "#1f77b4"), width=2),
            hovertemplate="Date: %{x}<br>Actual: %{y:.2f} EUR/MWh<extra></extra>"
        )
    )

    # Forecast prices
    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast.values,
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#d62728", width=2, dash="dash"),
            marker=dict(size=6),
            hovertemplate="Date: %{x}<br>Forecast: %{y:.2f} EUR/MWh<extra></extra>"
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Hour",
        yaxis_title="Price (EUR/MWh)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400
    )

    return fig


def create_cluster_probability_bar(
    probabilities: dict,
    title: str = "Cluster Probabilities"
) -> go.Figure:
    """
    Create a horizontal bar chart showing cluster probabilities.

    Parameters
    ----------
    probabilities : dict
        Dictionary mapping cluster IDs to probabilities (0-1).
    title : str
        Chart title.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    clusters = list(probabilities.keys())
    probs = list(probabilities.values())
    colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(len(clusters))]

    fig = go.Figure(
        go.Bar(
            x=probs,
            y=[f"Cluster {c}" for c in clusters],
            orientation="h",
            marker_color=colors,
            text=[f"{p:.1%}" for p in probs],
            textposition="auto",
            hovertemplate="Cluster %{y}<br>Probability: %{x:.2%}<extra></extra>"
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Probability",
        yaxis_title="",
        template="plotly_white",
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        height=250
    )

    return fig


# --- Metric Cards ---
def create_metric_indicator(
    value: float,
    title: str,
    unit: str = "",
    delta: Optional[float] = None,
    delta_suffix: str = ""
) -> go.Figure:
    """
    Create a gauge-style indicator for a single metric.

    Parameters
    ----------
    value : float
        The metric value to display.
    title : str
        Metric title.
    unit : str
        Unit suffix for the value.
    delta : float, optional
        Change from previous value.
    delta_suffix : str
        Suffix for delta display.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    fig = go.Figure(
        go.Indicator(
            mode="number+delta" if delta is not None else "number",
            value=value,
            title=dict(text=title),
            number=dict(suffix=unit),
            delta=dict(reference=value - delta, suffix=delta_suffix) if delta is not None else None
        )
    )

    fig.update_layout(
        height=150,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig
