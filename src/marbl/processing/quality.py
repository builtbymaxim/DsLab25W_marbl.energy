import pandas as pd


def coverage_pct(df: pd.DataFrame, col: str) -> float:
    n = len(df)
    if n == 0:
        return 0.0
    return 100.0 * df[col].notna().sum() / n


def missing_runs(df: pd.DataFrame, freq: str = "H") -> pd.DataFrame:
    # Erwartet DatetimeIndex
    full = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq=freq))
    gaps = full.index[full.isna().all(axis=1)]
    if len(gaps) == 0:
        return pd.DataFrame(columns=["start", "end", "len"])
    runs = []
    start = gaps[0]
    prev = gaps[0]
    for ts in gaps[1:]:
        if ts == prev + pd.tseries.frequencies.to_offset(freq):
            prev = ts
        else:
            runs.append((start, prev, (prev - start) / pd.Timedelta(freq) + 1))
            start = ts
            prev = ts
    runs.append((start, prev, (prev - start) / pd.Timedelta(freq) + 1))
    return pd.DataFrame(runs, columns=["start", "end", "len"])
