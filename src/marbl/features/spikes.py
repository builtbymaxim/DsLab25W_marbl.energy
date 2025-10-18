import pandas as pd


def detect_spikes(
    s: pd.Series, abs_pos: float, abs_neg: float, quantile: float | None = 0.99
) -> pd.DataFrame:
    df = pd.DataFrame({"price": s})
    flags = (df["price"] >= abs_pos) | (df["price"] <= abs_neg)
    if quantile:
        q_hi = df["price"].quantile(quantile)
        q_lo = df["price"].quantile(1 - quantile)
        flags = flags | (df["price"] >= q_hi) | (df["price"] <= q_lo)
    out = df.copy()
    out["is_spike"] = flags.fillna(False)
    return out
