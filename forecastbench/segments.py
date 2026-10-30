"""Syntetos-Boylan demand classes (ADI / CV^2) plus a cold-start flag, computed at an origin."""
import numpy as np
import pandas as pd


def classify(data, idx, origin, window_days, cold_start_days, adi_cut, cv2_cut):
    fs = data.first_sale[idx]
    rows = []
    for k, i in enumerate(idx):
        start = max(fs[k] - 1, origin - window_days)
        h = data.Y[i, start:origin]
        nz = h[h > 0]
        adi = len(h) / len(nz) if len(nz) else float("inf")
        cv2 = float((nz.std() / nz.mean()) ** 2) if len(nz) > 1 else 0.0
        rows.append((adi, cv2))
    df = pd.DataFrame(rows, columns=["adi", "cv2"])
    df["age"] = origin - fs + 1
    regular, stable = df["adi"] < adi_cut, df["cv2"] < cv2_cut
    df["segment"] = np.select(
        [regular & stable, regular & ~stable, ~regular & stable],
        ["smooth", "erratic", "intermittent"], "lumpy")
    df["cold_start"] = df["age"] < cold_start_days
    df["group"] = np.where(df["cold_start"], "cold_start", df["segment"])
    return df
