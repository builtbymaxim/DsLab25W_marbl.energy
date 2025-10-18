from pathlib import Path
import pandas as pd


def load_timeseries_csv(path: Path, ts_col: str = "timestamp", tz_utc: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    if ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col], utc=tz_utc)
        df = df.set_index(ts_col).sort_index()
    return df
