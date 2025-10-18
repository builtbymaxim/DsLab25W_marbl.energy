import pandas as pd
import matplotlib.pyplot as plt


def hourly_profile(s: pd.Series, title: str = "Hourly profile"):
    g = s.groupby(s.index.hour).mean()
    plt.figure()
    g.plot()
    plt.title(title)
    plt.xlabel("Hour")
    plt.ylabel(s.name or "")
    plt.tight_layout()
