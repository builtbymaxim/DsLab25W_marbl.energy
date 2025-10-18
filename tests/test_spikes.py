import pandas as pd
from marbl.features.spikes import detect_spikes


def test_detect_spikes_threshold():
    s = pd.Series([0, 10, 300, -60, 5], index=pd.date_range("2020-01-01", periods=5, freq="H"))
    df = detect_spikes(s, abs_pos=250, abs_neg=-50, quantile=None)
    assert df["is_spike"].sum() == 2
