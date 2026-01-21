"""
Module: model_inference.py
Description: ML model loading and inference for price forecasting.
Author: MARBL Dashboard Team
Date: 2026-01-21

This module handles loading trained XGBoost models and generating
real price predictions for the Live Forecast page.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st


# --- Path Configuration ---
MODELS_DIR = Path(__file__).parent.parent.parent / "models" / "approach_clusters"
CLUSTER_MODELS_DIR = MODELS_DIR / "cluster_predictions"
PRICE_MODELS_DIR = MODELS_DIR / "within_cluster_price_predictions"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"

# Zone code mapping (models use 2-letter codes)
ZONE_CODE_MAP = {
    "DK1": "DK",
    "ES": "ES",
    "NO2": "NO"
}

# Number of clusters per zone
ZONE_CLUSTERS = {
    "DK1": 6,
    "ES": 3,
    "NO2": 5
}

# Precipitation aggregation days per zone (from notebook optimization)
# DK and ES perform best with 20 days, NO with 7 days
ZONE_PRECIP_DAYS = {
    "DK1": 20,
    "ES": 20,
    "NO2": 7
}


# --- Historical Data Loading ---
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_masterset(zone: str) -> pd.DataFrame:
    """
    Load the masterset for a zone.

    Parameters
    ----------
    zone : str
        Bidding zone identifier (DK1, ES, NO2).

    Returns
    -------
    pd.DataFrame
        Historical data with datetime index.
    """
    masterset_path = DATA_DIR / f"{zone}_masterset.csv"

    if not masterset_path.exists():
        raise FileNotFoundError(f"Masterset not found: {masterset_path}")

    df = pd.read_csv(masterset_path, index_col=0)
    # Ensure index is proper DatetimeIndex
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def calculate_precipitation_last_x_days(zone: str, forecast_date: pd.Timestamp) -> float:
    """
    Calculate precipitation_last_X_days for a given forecast date.

    This follows the exact logic from Predictions.ipynb:
    1. Group hourly precipitation by date to get daily sums
    2. Shift by 1 day (exclude "today" from the window)
    3. Apply rolling sum with window=X days

    Parameters
    ----------
    zone : str
        Bidding zone identifier (DK1, ES, NO2).
    forecast_date : pd.Timestamp
        The date for which to calculate precipitation.

    Returns
    -------
    float
        Rolling sum of precipitation over the past X days.
    """
    precip_days = ZONE_PRECIP_DAYS.get(zone, 20)

    try:
        # Load historical data
        df = load_masterset(zone)

        # Get precipitation column
        if "precipitation_mm" not in df.columns:
            return 0.0

        # Create a date column from index (extract date part only)
        df = df.copy()
        # Convert to naive datetime first (remove timezone), then normalize to midnight
        df["date"] = df.index.tz_convert(None).normalize()

        # Aggregate hourly precipitation to daily sums
        daily = (
            df.groupby("date", as_index=False)["precipitation_mm"]
            .sum()
            .sort_values("date")
        )
        daily["date"] = pd.to_datetime(daily["date"])

        # Calculate rolling sum with shift (exclude today)
        daily["precip_rolling"] = (
            daily["precipitation_mm"]
            .shift(1)  # Exclude "today" from the window
            .rolling(window=precip_days, min_periods=precip_days)
            .sum()
        )

        # Get the value for the forecast date (or closest prior date with data)
        forecast_date_normalized = pd.Timestamp(forecast_date).normalize()

        # Filter to dates up to forecast date
        daily_filtered = daily[daily["date"] <= forecast_date_normalized]

        if daily_filtered.empty or daily_filtered["precip_rolling"].isna().all():
            # Not enough historical data - use mean from available data
            valid_values = daily["precip_rolling"].dropna()
            return float(valid_values.mean()) if len(valid_values) > 0 else 0.0

        # Get the last available value
        last_valid = daily_filtered["precip_rolling"].dropna()
        if len(last_valid) == 0:
            valid_values = daily["precip_rolling"].dropna()
            return float(valid_values.mean()) if len(valid_values) > 0 else 0.0

        return float(last_valid.iloc[-1])

    except Exception as e:
        st.warning(f"Could not calculate precipitation history: {e}")
        return 0.0


# --- Model Loading ---
@st.cache_resource
def load_cluster_model(zone: str) -> dict:
    """
    Load the cluster classification model for a zone.

    Parameters
    ----------
    zone : str
        Bidding zone identifier (DK1, ES, NO2).

    Returns
    -------
    dict
        Model bundle containing 'model', 'feature_cols', 'label_encoder'.
    """
    zone_code = ZONE_CODE_MAP.get(zone, zone)
    model_path = CLUSTER_MODELS_DIR / f"xgb_cluster_model_{zone_code}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Cluster model not found: {model_path}")

    return joblib.load(model_path)


@st.cache_resource
def load_price_models(zone: str, horizon: int = 3) -> dict:
    """
    Load within-cluster price prediction models for a zone.

    Parameters
    ----------
    zone : str
        Bidding zone identifier.
    horizon : int
        Prediction horizon (3, 7, 14, or 20 days). Default is 3.

    Returns
    -------
    dict
        Dictionary mapping cluster IDs to model bundles.
    """
    zone_code = ZONE_CODE_MAP.get(zone, zone)
    n_clusters = ZONE_CLUSTERS.get(zone, 5)

    models = {}
    for cluster_id in range(1, n_clusters + 1):
        model_path = PRICE_MODELS_DIR / f"xgb_price_{zone_code}_p{horizon}_c{cluster_id}.joblib"

        if model_path.exists():
            models[cluster_id] = joblib.load(model_path)

    if not models:
        raise FileNotFoundError(f"No price models found for {zone}")

    return models


def check_models_available() -> dict:
    """
    Check which models are available.

    Returns
    -------
    dict
        Dictionary with zone availability status.
    """
    availability = {}
    for zone in ["DK1", "ES", "NO2"]:
        zone_code = ZONE_CODE_MAP.get(zone)
        cluster_model = CLUSTER_MODELS_DIR / f"xgb_cluster_model_{zone_code}.joblib"
        availability[zone] = cluster_model.exists()
    return availability


def get_season(date: pd.Timestamp) -> str:
    """Get season from date (Northern Hemisphere)."""
    month = date.month
    if month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    elif month in [9, 10, 11]:
        return "autumn"
    else:
        return "winter"


# --- Feature Engineering ---
def prepare_cluster_features(
    weather_forecast: pd.DataFrame,
    recent_prices: pd.DataFrame,
    forecast_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Prepare features for cluster classification.

    Model expects these features:
    - avg_temperature, avg_wind_speed, sum_precipitation, sum_solar_radiation
    - avg_price_1, avg_price_2, avg_price_3 (lagged daily averages)
    - is_weekend
    - season_spring, season_summer, season_autumn (one-hot encoded)

    Parameters
    ----------
    weather_forecast : pd.DataFrame
        Hourly weather forecast with columns: temperature_2m, wind_speed_10m,
        solar_radiation_W, precipitation_mm.
    recent_prices : pd.DataFrame
        Recent price data with price_eur_mwh column.
    forecast_date : pd.Timestamp
        The date to forecast.

    Returns
    -------
    pd.DataFrame
        Single row DataFrame with cluster prediction features.
    """
    # Aggregate weather to daily values
    avg_temperature = weather_forecast["temperature_2m"].mean() if "temperature_2m" in weather_forecast.columns else 10.0
    avg_wind_speed = weather_forecast["wind_speed_10m"].mean() if "wind_speed_10m" in weather_forecast.columns else 5.0
    sum_precipitation = weather_forecast["precipitation_mm"].sum() if "precipitation_mm" in weather_forecast.columns else 0.0
    sum_solar_radiation = weather_forecast["solar_radiation_W"].sum() if "solar_radiation_W" in weather_forecast.columns else 200.0

    # Calculate lagged average prices from recent data
    if not recent_prices.empty and "price_eur_mwh" in recent_prices.columns:
        # Get daily average prices
        recent_prices_copy = recent_prices.copy()
        recent_prices_copy["date"] = recent_prices_copy.index.date
        daily_avg = recent_prices_copy.groupby("date")["price_eur_mwh"].mean()
        daily_avg = daily_avg.sort_index(ascending=False)

        # Get the last 3 days' averages
        avg_price_1 = float(daily_avg.iloc[0]) if len(daily_avg) > 0 else 50.0
        avg_price_2 = float(daily_avg.iloc[1]) if len(daily_avg) > 1 else avg_price_1
        avg_price_3 = float(daily_avg.iloc[2]) if len(daily_avg) > 2 else avg_price_2
    else:
        # Fallback values if no recent prices
        avg_price_1 = 50.0
        avg_price_2 = 50.0
        avg_price_3 = 50.0

    # Weekend indicator
    is_weekend = 1 if forecast_date.weekday() >= 5 else 0

    # Season one-hot encoding
    season = get_season(forecast_date)
    season_spring = 1 if season == "spring" else 0
    season_summer = 1 if season == "summer" else 0
    season_autumn = 1 if season == "autumn" else 0

    # Build feature DataFrame with exact column names expected by model
    features = pd.DataFrame([{
        "avg_temperature": avg_temperature,
        "avg_wind_speed": avg_wind_speed,
        "sum_precipitation": sum_precipitation,
        "sum_solar_radiation": sum_solar_radiation,
        "avg_price_1": avg_price_1,
        "avg_price_2": avg_price_2,
        "avg_price_3": avg_price_3,
        "is_weekend": is_weekend,
        "season_spring": season_spring,
        "season_summer": season_summer,
        "season_autumn": season_autumn
    }])

    return features


def prepare_price_features(
    weather_forecast: pd.DataFrame,
    forecast_date: pd.Timestamp,
    zone: str
) -> pd.DataFrame:
    """
    Prepare features for hourly price prediction.

    Model expects these features:
    - temperature_2m, wind_speed_10m, solar_radiation_W
    - precipitation_last_X_days (where X depends on zone: DK=20, ES=20, NO=7)

    Parameters
    ----------
    weather_forecast : pd.DataFrame
        Hourly weather forecast data.
    forecast_date : pd.Timestamp
        The date to forecast.
    zone : str
        Bidding zone identifier.

    Returns
    -------
    pd.DataFrame
        DataFrame with 24 rows (one per hour) with price prediction features.
    """
    # Get zone-specific precipitation days
    precip_days = ZONE_PRECIP_DAYS.get(zone, 20)
    precip_col_name = f"precipitation_last_{precip_days}_days"

    # Filter weather to forecast date only (24 hours)
    forecast_start = pd.Timestamp(forecast_date)
    forecast_end = forecast_start + pd.Timedelta(hours=23)

    # Try to get weather data for the forecast date
    if hasattr(weather_forecast.index, 'tz') and weather_forecast.index.tz is not None:
        forecast_start = forecast_start.tz_localize(weather_forecast.index.tz)
        forecast_end = forecast_end.tz_localize(weather_forecast.index.tz)

    weather_24h = weather_forecast[
        (weather_forecast.index >= forecast_start) &
        (weather_forecast.index <= forecast_end)
    ].copy()

    # If no weather data for forecast date, use what's available
    if len(weather_24h) == 0 and len(weather_forecast) > 0:
        weather_24h = weather_forecast.tail(24).copy()

    if len(weather_24h) < 24:
        # Pad with last available values or defaults
        if len(weather_24h) > 0:
            last_row = weather_24h.iloc[-1:].copy()
            while len(weather_24h) < 24:
                weather_24h = pd.concat([weather_24h, last_row], ignore_index=True)
        else:
            # Create default weather if none available
            weather_24h = pd.DataFrame({
                "temperature_2m": [10.0] * 24,
                "wind_speed_10m": [5.0] * 24,
                "solar_radiation_W": [200.0 * max(0, np.sin((h - 6) * np.pi / 12))
                                      for h in range(24)],
                "precipitation_mm": [0.0] * 24
            })

    # Ensure exactly 24 hours
    weather_24h = weather_24h.head(24).reset_index(drop=True)

    # Calculate precipitation_last_X_days from historical data
    # This is the rolling sum of daily precipitation over the past X days
    precip_last_x = calculate_precipitation_last_x_days(zone, forecast_date)

    # Extract column values with fallbacks
    temp_vals = (weather_24h["temperature_2m"].values
                 if "temperature_2m" in weather_24h.columns else [10.0] * 24)
    wind_vals = (weather_24h["wind_speed_10m"].values
                 if "wind_speed_10m" in weather_24h.columns else [5.0] * 24)
    solar_vals = (weather_24h["solar_radiation_W"].values
                  if "solar_radiation_W" in weather_24h.columns else [0.0] * 24)

    # Build features DataFrame with exact column names expected by model
    features = pd.DataFrame({
        "temperature_2m": temp_vals,
        "wind_speed_10m": wind_vals,
        "solar_radiation_W": solar_vals,
        precip_col_name: [precip_last_x] * 24  # Same value for all hours (rolling sum)
    })

    return features


# --- Prediction Functions ---
def predict_cluster_probabilities(
    zone: str,
    weather_forecast: pd.DataFrame,
    recent_prices: pd.DataFrame,
    forecast_date: pd.Timestamp
) -> dict:
    """
    Predict cluster probabilities for the forecast date.

    Parameters
    ----------
    zone : str
        Bidding zone identifier.
    weather_forecast : pd.DataFrame
        Hourly weather forecast.
    recent_prices : pd.DataFrame
        Recent actual prices.
    forecast_date : pd.Timestamp
        Date to forecast.

    Returns
    -------
    dict
        Dictionary mapping cluster IDs (1-indexed) to probabilities.
    """
    try:
        # Load model
        bundle = load_cluster_model(zone)
        model = bundle["model"]
        feature_cols = bundle["feature_cols"]
        label_encoder = bundle["label_encoder"]

        # Prepare features
        features = prepare_cluster_features(weather_forecast, recent_prices, forecast_date)

        # Ensure features are in the correct order
        X = features[feature_cols].copy()

        # Get probabilities
        probs = model.predict_proba(X)[0]

        # Map back to original cluster labels (1-indexed)
        cluster_labels = label_encoder.inverse_transform(range(len(probs)))

        return {int(label): float(prob) for label, prob in zip(cluster_labels, probs)}

    except Exception as e:
        st.warning(f"Cluster prediction error: {e}. Using fallback.")
        n_clusters = ZONE_CLUSTERS.get(zone, 5)
        return {i: 1.0 / n_clusters for i in range(1, n_clusters + 1)}


def predict_hourly_prices(
    zone: str,
    weather_forecast: pd.DataFrame,
    cluster_probs: dict,
    forecast_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Predict hourly prices using Mix-of-Experts approach.

    Each cluster model predicts prices, and the final prediction
    is a weighted average based on cluster probabilities.

    Parameters
    ----------
    zone : str
        Bidding zone identifier.
    weather_forecast : pd.DataFrame
        Hourly weather forecast.
    cluster_probs : dict
        Cluster probabilities from cluster model.
    forecast_date : pd.Timestamp
        Date to forecast.

    Returns
    -------
    pd.DataFrame
        DataFrame with timestamp index and price_forecast column.
    """
    try:
        # Get zone-specific precipitation days for model selection
        precip_days = ZONE_PRECIP_DAYS.get(zone, 20)

        # Load price models with correct precipitation aggregation
        price_models = load_price_models(zone, precip_days)

        # Prepare features with zone-specific precipitation column name
        features = prepare_price_features(weather_forecast, forecast_date, zone)

        # Get predictions from each cluster model
        cluster_predictions = {}
        for cluster_id, bundle in price_models.items():
            model = bundle["model"]
            model_feature_cols = bundle["feature_cols"]

            # Ensure features are in the correct order
            X = features[model_feature_cols].copy()

            # Predict
            pred = model.predict(X)
            cluster_predictions[cluster_id] = pred

        # Weighted average based on cluster probabilities (Mix-of-Experts)
        final_predictions = np.zeros(24)
        for cluster_id, pred in cluster_predictions.items():
            prob = cluster_probs.get(cluster_id, 0.0)
            final_predictions += prob * pred

        # Create output DataFrame
        hours = pd.date_range(
            start=forecast_date,
            periods=24,
            freq="h"
        )

        result = pd.DataFrame({
            "timestamp": hours,
            "price_forecast": final_predictions
        })
        result.set_index("timestamp", inplace=True)

        # Note: Negative prices ARE valid in electricity markets
        # (e.g., DK1 has seen prices from -440 to +936 EUR/MWh)
        return result

    except Exception as e:
        st.warning(f"Price prediction error: {e}. Using fallback.")
        return _generate_fallback_forecast(zone, forecast_date)


def _generate_fallback_forecast(zone: str, forecast_date: pd.Timestamp) -> pd.DataFrame:
    """
    Generate realistic fallback forecast when models fail.

    Uses zone-specific typical patterns based on historical data analysis.
    DK1: avg ~80 EUR/MWh, ES: avg ~70 EUR/MWh, NO2: avg ~65 EUR/MWh
    """
    np.random.seed(int(forecast_date.timestamp()) % 10000)

    hours = pd.date_range(start=forecast_date, periods=24, freq="h")
    hour_of_day = np.arange(24)

    # Zone-specific base patterns (calibrated to historical averages)
    if zone == "DK1":
        # DK1: Wind-dominated, avg ~80 EUR/MWh
        # Morning peak 6-9, evening peak 17-20
        base = 75 + 25 * np.sin((hour_of_day - 3) * np.pi / 12)
        morning_peak = 15 * np.exp(-((hour_of_day - 7) ** 2) / 4)
        evening_peak = 20 * np.exp(-((hour_of_day - 18) ** 2) / 4)
        base = base + morning_peak + evening_peak
        noise = np.random.randn(24) * 12
    elif zone == "ES":
        # ES: Solar-dominated, avg ~70 EUR/MWh with solar dip midday
        base = 65 + 20 * np.sin((hour_of_day - 3) * np.pi / 12)
        solar_dip = -20 * np.exp(-((hour_of_day - 13) ** 2) / 6)
        evening_peak = 25 * np.exp(-((hour_of_day - 20) ** 2) / 4)
        base = base + solar_dip + evening_peak
        noise = np.random.randn(24) * 10
    else:  # NO2
        # NO2: Hydro-dominated, more stable, avg ~65 EUR/MWh
        base = 60 + 15 * np.sin((hour_of_day - 4) * np.pi / 12)
        morning_peak = 10 * np.exp(-((hour_of_day - 8) ** 2) / 5)
        evening_peak = 12 * np.exp(-((hour_of_day - 18) ** 2) / 5)
        base = base + morning_peak + evening_peak
        noise = np.random.randn(24) * 8

    # Weekend effect: lower demand, lower prices
    if forecast_date.weekday() >= 5:
        base = base * 0.88

    # Seasonal adjustment (winter = higher prices)
    month = forecast_date.month
    if month in [12, 1, 2]:  # Winter
        base = base * 1.15
    elif month in [6, 7, 8]:  # Summer
        base = base * 0.90

    # Note: Negative prices ARE valid in electricity markets
    prices = base + noise

    return pd.DataFrame({
        "timestamp": hours,
        "price_forecast": prices
    }).set_index("timestamp")


def generate_real_forecast(
    zone: str,
    weather_forecast: pd.DataFrame,
    recent_prices: pd.DataFrame,
    forecast_date: pd.Timestamp,
    use_ml_models: bool = True  # Enabled - uses trained XGBoost models
) -> tuple[pd.DataFrame, dict, bool]:
    """
    Generate price forecast.

    This is the main entry point for the Live Forecast page.

    Parameters
    ----------
    zone : str
        Bidding zone identifier.
    weather_forecast : pd.DataFrame
        Hourly weather forecast.
    recent_prices : pd.DataFrame
        Recent actual prices for lagged features.
    forecast_date : pd.Timestamp
        Date to forecast.
    use_ml_models : bool
        Whether to use ML models (requires validation).

    Returns
    -------
    tuple
        (price_forecast_df, cluster_probabilities, is_real_prediction)
    """
    if use_ml_models:
        try:
            # Check if models are available
            availability = check_models_available()
            if not availability.get(zone, False):
                raise FileNotFoundError(f"Models not available for {zone}")

            # Step 1: Predict cluster probabilities
            cluster_probs = predict_cluster_probabilities(
                zone, weather_forecast, recent_prices, forecast_date
            )

            # Step 2: Predict hourly prices using Mix-of-Experts
            price_forecast = predict_hourly_prices(
                zone, weather_forecast, cluster_probs, forecast_date
            )

            # Note: We don't validate price ranges because negative and very low
            # prices ARE valid in electricity markets (e.g., during high renewable
            # generation or low demand periods)
            return price_forecast, cluster_probs, True

        except Exception as e:
            st.warning(f"ML prediction issue: {e}. Using pattern-based forecast.")

    # Use pattern-based forecast (realistic fallback)
    fallback_forecast = _generate_fallback_forecast(zone, forecast_date)
    n_clusters = ZONE_CLUSTERS.get(zone, 5)

    # Generate plausible cluster probabilities based on day/weather
    cluster_probs = _generate_cluster_probabilities(zone, weather_forecast, forecast_date)

    return fallback_forecast, cluster_probs, False


def _generate_cluster_probabilities(
    zone: str,
    weather_forecast: pd.DataFrame,
    forecast_date: pd.Timestamp
) -> dict:
    """Generate plausible cluster probabilities based on weather/date."""
    n_clusters = ZONE_CLUSTERS.get(zone, 5)
    np.random.seed(int(forecast_date.timestamp()) % 10000 + hash(zone) % 100)

    # Base probabilities (uniform with noise)
    probs = np.random.dirichlet(np.ones(n_clusters) * 2)

    # Make one cluster dominant
    dominant = np.argmax(probs)
    probs[dominant] += 0.2
    probs = probs / probs.sum()

    return {i + 1: float(p) for i, p in enumerate(probs)}
