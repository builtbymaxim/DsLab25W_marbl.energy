# MARBL Energy Dashboard

Interactive dashboard for visualizing European electricity price trends and forecasts.

## Project Overview

This dashboard is part of the Data Science Lab 2025/26 project at WU Vienna,
developed for MARBL (Market Analytics).

**Features:**
- Historical price and weather data exploration
- Cluster analysis visualization (price pattern detection)
- Day-ahead price forecasting with XGBoost models

**Bidding Zones:**
- DK1: Denmark West (wind-dominated)
- ES: Spain (solar-dominated)
- NO2: South Norway (hydro-dominated)

## Installation

1. Navigate to the dashboard directory:
   ```bash
   cd dashboard
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Dashboard

Start the Streamlit server:

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

## Project Structure

```
dashboard/
├── app.py                      # Main entry point
├── pages/                      # Dashboard pages
│   ├── 1_Market_Overview.py    # Historical data exploration
│   ├── 2_Cluster_Analysis.py   # Pattern visualization
│   └── 3_Live_Forecast.py      # Price predictions
├── utils/                      # Utility modules
│   ├── __init__.py
│   ├── data_loader.py          # Data loading functions
│   └── visualizations.py       # Chart creation functions
├── assets/                     # Static files
│   └── style.css               # Custom styling
├── .streamlit/                 # Streamlit configuration
│   └── config.toml
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Data Requirements

The dashboard expects the following data files in the parent `data/` directory:

```
data/
├── processed/
│   ├── DK1_masterset.csv
│   ├── ES_masterset.csv
│   └── NO2_masterset.csv
├── live/                       # (Optional) Live weather forecasts
│   └── {ZONE}_forecast.csv
└── figures/                    # (Optional) Analysis output figures
    └── *.png
```

## Pages

### 1. Market Overview
Explore historical electricity prices and weather data:
- Price time series with daily/hourly aggregation
- Average daily price profiles
- Weather variable charts
- Price-weather correlation scatter plots

### 2. Cluster Analysis
View identified price patterns:
- Cluster centroid profiles (24-hour patterns)
- Cluster distribution statistics
- Temporal patterns (monthly, weekly)
- Calendar heatmap view

Note: Currently uses example data. Replace with actual cluster outputs
when available from analysis notebooks.

### 3. Live Forecast
Day-ahead price predictions:
- 24-hour price forecast chart
- Cluster probability predictions
- Weather forecast inputs
- Model methodology explanation

Note: Currently uses mock predictions. Replace with actual model inference
when XGBoost models are trained.

## Integration Points

### Adding Real Cluster Data

To integrate actual cluster analysis results, modify `pages/2_Cluster_Analysis.py`:

1. Replace `generate_example_centroids()` with loading from:
   - Parquet/CSV files with cluster centroids
   - Or compute from stored cluster assignments

2. Replace `generate_example_cluster_assignments()` with loading from:
   - Cluster assignment CSV with columns: date, cluster

### Adding Real Model Predictions

To integrate trained XGBoost models, modify `pages/3_Live_Forecast.py`:

1. Add model loading in `utils/model_inference.py`:
   ```python
   import joblib

   @st.cache_resource
   def load_cluster_classifier():
       return joblib.load("models/cluster_classifier.pkl")

   @st.cache_resource
   def load_price_model(cluster_id):
       return joblib.load(f"models/price_models/cluster_{cluster_id}_model.pkl")
   ```

2. Replace mock functions with actual inference calls.

### Adding Live Weather Data

To use real weather forecasts from WeatherAPI:

1. Load forecast CSVs from `data/live/{ZONE}_forecast.csv`
2. Replace `generate_mock_weather_forecast()` with actual data loading

## Configuration

Streamlit settings are in `.streamlit/config.toml`:
- Theme colors
- Server port
- Usage stats (disabled)

## Deployment

### Streamlit Community Cloud

1. Push code to GitHub
2. Go to share.streamlit.io
3. Connect your repository
4. Set main file path: `dashboard/app.py`

### Local Network

```bash
streamlit run app.py --server.address 0.0.0.0
```

## Dependencies

- streamlit: Web app framework
- pandas: Data manipulation
- numpy: Numerical computing
- plotly: Interactive visualizations
- scikit-learn: ML utilities
- xgboost: Prediction models
- joblib: Model serialization

## Authors

- Maximilian Dieringer
- Harald Körbel
- Maxim Gomez Valverde
- Daniel Klaric

## License

Project developed for WU Vienna Data Science Lab 2025/26.
