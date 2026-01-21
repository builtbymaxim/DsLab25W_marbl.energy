"""
Module: data_loader.py
Description: Functions for loading and preprocessing masterset data.
Author: MARBL Dashboard Team
Date: 2026-01-16
"""

# --- Imports ---
from pathlib import Path
from typing import Optional, List

import pandas as pd
import streamlit as st


# --- Constants ---
# Path to the data directory (relative to dashboard folder)
DATA_DIR = Path(__file__).parent.parent.parent / "data"

# Valid bidding zones
VALID_ZONES = ["DK1", "ES", "NO2"]

# Zone display names for UI
ZONE_NAMES = {
    "DK1": "Denmark West (DK1)",
    "ES": "Spain (ES)",
    "NO2": "South Norway (NO2)"
}

# Zone descriptions with market characteristics
ZONE_DESCRIPTIONS = {
    "DK1": "Wind-dominated market with high renewable penetration (55% wind generation)",
    "ES": "Solar-dominated market with significant midday price dips from PV generation",
    "NO2": "Hydro-dominated market connected to Continental Europe via subsea cables"
}

# Zone short labels for compact display
ZONE_SHORT = {
    "DK1": "Wind",
    "ES": "Solar",
    "NO2": "Hydro"
}

# Column descriptions for documentation
COLUMN_DESCRIPTIONS = {
    "price_eur_mwh": "Day-ahead electricity price in EUR/MWh",
    "temperature_2m": "Temperature at 2m height in Celsius",
    "wind_speed_10m": "Wind speed at 10m height in m/s",
    "precipitation_mm": "Precipitation in mm",
    "solar_radiation_W": "Solar radiation in W/m2"
}


# --- Data Loading Functions ---
@st.cache_data(ttl=3600)
def load_masterset(zone: str) -> pd.DataFrame:
    """
    Load the masterset CSV file for a given bidding zone.

    The masterset contains hourly price and weather data merged together.
    Data is cached for 1 hour to avoid repeated file reads.

    Parameters
    ----------
    zone : str
        The bidding zone identifier. One of: 'DK1', 'ES', 'NO2'.

    Returns
    -------
    pd.DataFrame
        The masterset with datetime index and columns:
        - price_eur_mwh: electricity price
        - temperature_2m: temperature
        - wind_speed_10m: wind speed
        - precipitation_mm: precipitation
        - solar_radiation_W: solar radiation

    Raises
    ------
    ValueError
        If the zone identifier is not recognized.
    FileNotFoundError
        If the masterset file does not exist.
    """
    # Validate input
    if zone not in VALID_ZONES:
        raise ValueError(f"Zone must be one of {VALID_ZONES}, got: {zone}")

    # Build file path
    file_path = DATA_DIR / "processed" / f"{zone}_masterset.csv"

    # Check file exists
    if not file_path.exists():
        raise FileNotFoundError(f"Masterset not found: {file_path}")

    # Load CSV with datetime index
    df = pd.read_csv(
        file_path,
        index_col=0,
        parse_dates=True
    )

    # Convert index to proper DatetimeIndex (handles timezone-aware timestamps)
    df.index = pd.to_datetime(df.index, utc=True)

    # Ensure index is named consistently
    df.index.name = "timestamp"

    return df


def load_all_mastersets() -> dict:
    """
    Load mastersets for all available zones.

    Returns
    -------
    dict
        Dictionary mapping zone codes to DataFrames.
        Example: {'DK1': df_dk1, 'ES': df_es, 'NO2': df_no2}
    """
    mastersets = {}
    for zone in VALID_ZONES:
        try:
            mastersets[zone] = load_masterset(zone)
        except FileNotFoundError:
            # Skip zones without data
            continue
    return mastersets


@st.cache_data(ttl=3600)
def load_masterset_filtered(
    zone: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Load masterset with optional date filtering.

    Parameters
    ----------
    zone : str
        The bidding zone identifier.
    start_date : str, optional
        Start date in format 'YYYY-MM-DD'. If None, no start filter.
    end_date : str, optional
        End date in format 'YYYY-MM-DD'. If None, no end filter.

    Returns
    -------
    pd.DataFrame
        Filtered masterset DataFrame.
    """
    df = load_masterset(zone)

    # Apply date filters
    if start_date is not None:
        df = df[df.index >= start_date]
    if end_date is not None:
        df = df[df.index <= end_date]

    return df


# --- Aggregation Functions ---
def get_daily_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly prices to daily statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Masterset DataFrame with hourly data.

    Returns
    -------
    pd.DataFrame
        Daily aggregated data with columns:
        - price_mean: average daily price
        - price_min: minimum hourly price
        - price_max: maximum hourly price
        - price_std: standard deviation of hourly prices
    """
    daily = df["price_eur_mwh"].resample("D").agg(["mean", "min", "max", "std"])
    daily.columns = ["price_mean", "price_min", "price_max", "price_std"]
    return daily


def get_daily_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the average price profile by hour of day.

    Parameters
    ----------
    df : pd.DataFrame
        Masterset DataFrame with hourly data.

    Returns
    -------
    pd.DataFrame
        DataFrame with index 0-23 (hours) and columns:
        - price_mean: average price at each hour
        - price_std: standard deviation at each hour
    """
    df_copy = df.copy()
    df_copy["hour"] = df_copy.index.hour

    hourly_profile = df_copy.groupby("hour")["price_eur_mwh"].agg(
        price_mean="mean",
        price_std="std"
    )
    return hourly_profile


# --- Statistics Functions ---
def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Calculate summary statistics for a masterset.

    Parameters
    ----------
    df : pd.DataFrame
        Masterset DataFrame.

    Returns
    -------
    dict
        Dictionary with summary statistics:
        - start_date: first timestamp
        - end_date: last timestamp
        - total_records: number of rows
        - price_mean: average price
        - price_std: price standard deviation
        - price_min: minimum price
        - price_max: maximum price
    """
    summary = {
        "start_date": df.index.min(),
        "end_date": df.index.max(),
        "total_records": len(df),
        "price_mean": df["price_eur_mwh"].mean(),
        "price_std": df["price_eur_mwh"].std(),
        "price_min": df["price_eur_mwh"].min(),
        "price_max": df["price_eur_mwh"].max(),
    }
    return summary


def get_date_range(zone: str) -> tuple:
    """
    Get the available date range for a zone.

    Parameters
    ----------
    zone : str
        The bidding zone identifier.

    Returns
    -------
    tuple
        (min_date, max_date) as datetime objects.
    """
    df = load_masterset(zone)
    return df.index.min(), df.index.max()


# --- Validation Functions ---
def check_data_availability() -> dict:
    """
    Check which zone data files are available.

    Returns
    -------
    dict
        Dictionary mapping zone codes to availability status.
        Example: {'DK1': True, 'ES': True, 'NO2': True}
    """
    availability = {}
    for zone in VALID_ZONES:
        file_path = DATA_DIR / "processed" / f"{zone}_masterset.csv"
        availability[zone] = file_path.exists()
    return availability


# --- Cluster Data Loading Functions ---
# Zone code mapping for prediction files (teammate used different naming)
ZONE_CODE_MAP = {
    "DK1": "DK",
    "ES": "ES",
    "NO2": "NO"
}

# Number of clusters per zone (from analysis)
ZONE_CLUSTERS = {
    "DK1": 6,
    "ES": 3,
    "NO2": 5
}


@st.cache_data(ttl=3600)
def load_cluster_assignments(zone: str) -> pd.DataFrame:
    """
    Load real cluster assignments for a zone.

    Parameters
    ----------
    zone : str
        The bidding zone identifier.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, cluster
    """
    if zone not in VALID_ZONES:
        raise ValueError(f"Zone must be one of {VALID_ZONES}, got: {zone}")

    file_path = DATA_DIR / "processed" / f"{zone}_date_cluster.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Cluster file not found: {file_path}")

    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


@st.cache_data(ttl=3600)
def load_predictions(zone: str, approach: str = "cluster") -> pd.DataFrame:
    """
    Load model predictions for a zone.

    Parameters
    ----------
    zone : str
        The bidding zone identifier.
    approach : str
        Prediction approach: 'cluster', 'naive_one', or 'naive_two'

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: price_real, price_predicted, date_time
    """
    if zone not in VALID_ZONES:
        raise ValueError(f"Zone must be one of {VALID_ZONES}, got: {zone}")

    zone_code = ZONE_CODE_MAP.get(zone, zone)
    file_path = DATA_DIR / "processed" / f"{zone_code}_predictions_{approach}_approach.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {file_path}")

    df = pd.read_csv(file_path)
    df["date_time"] = pd.to_datetime(df["date_time"], utc=True)
    df = df.set_index("date_time").sort_index()
    return df


def load_all_predictions(zone: str) -> dict:
    """
    Load predictions from all approaches for a zone.

    Parameters
    ----------
    zone : str
        The bidding zone identifier.

    Returns
    -------
    dict
        Dictionary mapping approach names to DataFrames.
    """
    approaches = ["cluster", "naive_one", "naive_two"]
    predictions = {}

    for approach in approaches:
        try:
            predictions[approach] = load_predictions(zone, approach)
        except FileNotFoundError:
            continue

    return predictions


def get_cluster_distribution(zone: str) -> dict:
    """
    Get the distribution of clusters for a zone.

    Parameters
    ----------
    zone : str
        The bidding zone identifier.

    Returns
    -------
    dict
        Dictionary mapping cluster IDs to counts.
    """
    df = load_cluster_assignments(zone)
    return df["cluster"].value_counts().sort_index().to_dict()


def get_masterset_with_clusters(zone: str) -> pd.DataFrame:
    """
    Load masterset data merged with cluster assignments.

    Parameters
    ----------
    zone : str
        The bidding zone identifier.

    Returns
    -------
    pd.DataFrame
        Masterset with additional 'cluster' column.
    """
    df = load_masterset(zone)
    clusters = load_cluster_assignments(zone)

    # Add date column to masterset for merging
    df_copy = df.copy()
    df_copy["date"] = df_copy.index.normalize()

    # Normalize cluster dates
    clusters["date"] = clusters["date"].dt.normalize()

    # Merge on date
    df_merged = df_copy.merge(
        clusters[["date", "cluster"]],
        on="date",
        how="left"
    )
    df_merged = df_merged.set_index(df.index)
    df_merged = df_merged.drop(columns=["date"])

    return df_merged
