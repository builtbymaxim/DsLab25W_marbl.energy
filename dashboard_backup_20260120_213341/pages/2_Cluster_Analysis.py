"""
Module: 2_Cluster_Analysis.py
Description: Cluster analysis and pattern visualization page.
Author: MARBL Dashboard Team
Date: 2026-01-16

This page displays real cluster assignments from the pattern detection analysis.
"""

# --- Imports ---
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import (
    load_masterset,
    load_cluster_assignments,
    get_masterset_with_clusters,
    get_cluster_distribution,
    VALID_ZONES,
    ZONE_NAMES,
    ZONE_DESCRIPTIONS,
    ZONE_SHORT,
    ZONE_CLUSTERS
)
from utils.visualizations import (
    create_cluster_centroids_chart,
    create_cluster_calendar_heatmap,
    CLUSTER_COLORS
)
import plotly.express as px
import plotly.graph_objects as go


# --- Page Configuration ---
st.set_page_config(
    page_title="Cluster Analysis - MARBL",
    page_icon="chart_with_upwards_trend",
    layout="wide"
)


# --- Data Paths ---
DATA_DIR = Path(__file__).parent.parent.parent / "data"
FIGURES_DIR = DATA_DIR / "figures"


# --- Sidebar ---
def render_sidebar():
    """Render the sidebar with MARBL branding."""
    logo_path = Path(__file__).parent.parent / "assets" / "marbl_logo.png"
    if logo_path.exists():
        st.sidebar.image(str(logo_path), use_container_width=True)
    st.sidebar.markdown("---")


# --- Helper Functions ---
def calculate_real_centroids(zone: str) -> pd.DataFrame:
    """
    Calculate cluster centroid profiles from real data.

    Parameters
    ----------
    zone : str
        Bidding zone identifier.

    Returns
    -------
    pd.DataFrame
        DataFrame with hours as index and cluster IDs as columns.
    """
    try:
        df = get_masterset_with_clusters(zone)

        # Add hour column
        df["hour"] = df.index.hour

        # Calculate mean price per hour per cluster
        centroids = df.groupby(["cluster", "hour"])["price_eur_mwh"].mean().unstack(level=0)

        # Reorder index to be 0-23
        centroids = centroids.reindex(range(24))

        return centroids

    except Exception as e:
        st.error(f"Error calculating centroids: {e}")
        return pd.DataFrame()


def load_real_cluster_assignments(zone: str) -> pd.DataFrame:
    """
    Load real cluster assignments from the analysis notebooks.

    Parameters
    ----------
    zone : str
        Bidding zone identifier.

    Returns
    -------
    pd.DataFrame
        DataFrame with date index and cluster column.
    """
    try:
        df = load_cluster_assignments(zone)

        # Set date as index
        if "date" in df.columns:
            df = df.set_index("date")

        return df

    except Exception as e:
        st.error(f"Error loading cluster assignments: {e}")
        return pd.DataFrame()


def create_cluster_distribution_chart(assignments: pd.DataFrame) -> go.Figure:
    """
    Create a bar chart showing cluster size distribution.

    Parameters
    ----------
    assignments : pd.DataFrame
        DataFrame with cluster column.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    counts = assignments["cluster"].value_counts().sort_index()

    fig = go.Figure(
        go.Bar(
            x=[f"Cluster {c}" for c in counts.index],
            y=counts.values,
            marker_color=[CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(len(counts))],
            text=counts.values,
            textposition="auto"
        )
    )

    fig.update_layout(
        title="Cluster Size Distribution",
        xaxis_title="Cluster",
        yaxis_title="Number of Days",
        template="plotly_white",
        height=350
    )

    return fig


def create_monthly_cluster_heatmap(assignments: pd.DataFrame) -> go.Figure:
    """
    Create a heatmap showing cluster distribution by month.

    Parameters
    ----------
    assignments : pd.DataFrame
        DataFrame with datetime index and cluster column.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    # Add month column
    df = assignments.copy()
    df["month"] = df.index.month
    df["year"] = df.index.year

    # Create pivot table: count clusters per month
    pivot = df.groupby(["year", "month", "cluster"]).size().unstack(fill_value=0)

    # Normalize by row to get proportions
    pivot_norm = pivot.div(pivot.sum(axis=1), axis=0)

    # Flatten for heatmap
    pivot_norm = pivot_norm.reset_index()
    pivot_norm["period"] = pivot_norm["year"].astype(str) + "-" + pivot_norm["month"].astype(str).str.zfill(2)

    # Prepare data for heatmap
    z_data = []
    x_labels = sorted(pivot_norm["period"].unique())
    y_labels = [f"Cluster {i}" for i in sorted(df["cluster"].unique())]

    for cluster in sorted(df["cluster"].unique()):
        row = []
        for period in x_labels:
            val = pivot_norm.loc[pivot_norm["period"] == period, cluster]
            row.append(val.values[0] if len(val) > 0 else 0)
        z_data.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_data,
            x=x_labels,
            y=y_labels,
            colorscale="Blues",
            colorbar=dict(title="Proportion")
        )
    )

    fig.update_layout(
        title="Cluster Distribution by Month",
        xaxis_title="Month",
        yaxis_title="Cluster",
        template="plotly_white",
        height=350,
        xaxis=dict(tickangle=45)
    )

    return fig


# --- Main Page ---
def main():
    """Render the cluster analysis page."""

    render_sidebar()

    st.title("Cluster Analysis")
    st.markdown(
        """
        Analysis of recurring daily price patterns identified through hierarchical clustering.
        Patterns are detected using Ward-linkage clustering on 24-hour price profiles.
        """
    )

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

    # Show number of clusters for this zone
    n_clusters = ZONE_CLUSTERS.get(selected_zone, 5)
    st.sidebar.markdown("---")
    st.sidebar.metric("Clusters Detected", n_clusters)

    # --- Load Real Data ---
    with st.spinner("Loading cluster data..."):
        assignments = load_real_cluster_assignments(selected_zone)
        centroids = calculate_real_centroids(selected_zone)

    # Check if data loaded successfully
    if assignments.empty or centroids.empty:
        st.error("Could not load cluster data. Please check that the analysis notebooks have been run.")
        st.stop()

    # Success indicator
    st.success(
        f"Loaded **real cluster data** for {ZONE_NAMES.get(selected_zone, selected_zone)}: "
        f"{len(assignments)} days assigned to {n_clusters} clusters."
    )

    # --- Cluster Centroids ---
    st.subheader("Cluster Centroid Profiles")

    st.markdown(
        """
        Each cluster represents a typical daily price pattern. The centroid profile
        shows the average hourly price for all days assigned to that cluster.
        """
    )

    fig_centroids = create_cluster_centroids_chart(
        centroids=centroids,
        title=f"Cluster Centroids - {ZONE_NAMES.get(selected_zone, selected_zone)}"
    )

    st.plotly_chart(fig_centroids, use_container_width=True)

    # Centroid interpretation
    with st.expander("Cluster Interpretation Guide"):
        st.markdown(
            """
            **How to interpret cluster profiles:**

            - **Morning Peak**: High prices in early morning hours (6-10) indicate high demand
              during business start times, often in winter months.

            - **Midday Dip**: Lower prices around noon suggest high solar generation
              (especially in ES) reducing the need for conventional power.

            - **Evening Peak**: High prices in evening hours (17-21) reflect peak residential
              demand when solar generation decreases.

            - **Flat Profile**: Consistent prices throughout the day indicate stable supply/demand
              balance, often during mild weather periods.

            - **High Volatility**: Large price swings within a day suggest supply constraints
              or weather-driven renewable variability.
            """
        )

    st.divider()

    # --- Cluster Statistics ---
    st.subheader("Cluster Statistics")

    col1, col2 = st.columns(2)

    with col1:
        fig_dist = create_cluster_distribution_chart(assignments)
        st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        # Statistics table
        stats_data = []
        unique_clusters = sorted(assignments["cluster"].unique())
        for cluster in unique_clusters:
            cluster_days = assignments[assignments["cluster"] == cluster]
            avg_price = centroids[cluster].mean() if cluster in centroids.columns else 0
            stats_data.append({
                "Cluster": cluster,
                "Days": len(cluster_days),
                "Percentage": f"{len(cluster_days) / len(assignments) * 100:.1f}%",
                "Avg Price": f"{avg_price:.2f} EUR/MWh"
            })

        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    st.divider()

    # --- Temporal Patterns ---
    st.subheader("Temporal Patterns")

    st.markdown(
        """
        How clusters are distributed across time reveals seasonal and weekly patterns
        in electricity market behavior.
        """
    )

    # Monthly heatmap
    fig_monthly = create_monthly_cluster_heatmap(assignments)
    st.plotly_chart(fig_monthly, use_container_width=True)

    # Weekday distribution
    st.markdown("**Cluster Distribution by Day of Week**")

    assignments_copy = assignments.copy()
    assignments_copy["weekday"] = assignments_copy.index.dayofweek
    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    assignments_copy["weekday_name"] = assignments_copy["weekday"].map(
        lambda x: weekday_names[x]
    )

    weekday_cluster = assignments_copy.groupby(
        ["weekday", "weekday_name", "cluster"]
    ).size().reset_index(name="count")

    fig_weekday = px.bar(
        weekday_cluster,
        x="weekday_name",
        y="count",
        color="cluster",
        barmode="stack",
        category_orders={"weekday_name": weekday_names},
        color_discrete_sequence=CLUSTER_COLORS,
        labels={"weekday_name": "Day of Week", "count": "Number of Days", "cluster": "Cluster"}
    )

    fig_weekday.update_layout(
        title="Cluster Distribution by Day of Week",
        template="plotly_white",
        height=400,
        legend_title="Cluster"
    )

    st.plotly_chart(fig_weekday, use_container_width=True)

    st.divider()

    # --- Calendar View ---
    st.subheader("Calendar View")

    st.markdown("View cluster assignments on a calendar to see patterns over time.")

    # Year selector
    available_years = sorted(assignments.index.year.unique())
    selected_year = st.selectbox(
        label="Select Year",
        options=available_years,
        index=len(available_years) - 1  # Default to most recent year
    )

    fig_calendar = create_cluster_calendar_heatmap(
        df=assignments,
        cluster_col="cluster",
        year=selected_year
    )

    st.plotly_chart(fig_calendar, use_container_width=True)

    st.divider()

    # --- Methodology Note ---
    with st.expander("Clustering Methodology"):
        st.markdown(
            """
            **Hierarchical Clustering with Ward Linkage**

            The clustering methodology groups days with similar 24-hour price profiles:

            1. **Feature Extraction**: Each day is represented by its 24-hour price vector.

            2. **Distance Calculation**: Ward's linkage minimizes the total within-cluster variance.

            3. **Dendrogram Analysis**: Optimal cluster count is determined by analyzing the
               dendrogram and silhouette scores.

            4. **Cluster Assignment**: Each day is assigned to the cluster whose centroid
               best matches its price profile.

            **Applications:**
            - Understanding market regimes
            - Input for forecasting models (Mix-of-Experts)
            - Anomaly detection for unusual price days
            """
        )

    # --- Footer ---
    st.markdown("---")
    st.caption(
        "Cluster analysis based on day-ahead prices from ENTSO-E Transparency Platform."
    )


# --- Entry Point ---
if __name__ == "__main__":
    main()
