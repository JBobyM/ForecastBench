"""Load M5 into dense arrays. Column j of every (n, D) array is day d = j + 1."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HIER_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]


@dataclass
class M5Data:
    meta: pd.DataFrame       # one row per series: id + hierarchy columns
    Y: np.ndarray            # (n, D) units sold, NaN after the last observed day
    price: np.ndarray        # (n, D) sell price, NaN when not listed
    cal: pd.DataFrame        # one row per day d = 1..D
    first_sale: np.ndarray   # (n,) first day d with units > 0
    last_obs: int            # last observed day d
    codes: dict              # integer codes for categorical features


def load_m5(raw_dir, category, n_series, n_cold_start_extra, min_history_days, origins, cold_start_days, **_):
    raw = Path(raw_dir)
    sales = pd.read_csv(raw / "sales_train_evaluation.csv")
    cal = pd.read_csv(raw / "calendar.csv", parse_dates=["date"])
    prices = pd.read_csv(raw / "sell_prices.csv")

    cal["d"] = cal["d"].str[2:].astype(int)
    cal = cal.sort_values("d").reset_index(drop=True)
    cal["event"] = cal["event_name_1"].notna().astype(np.int8)
    D = int(cal["d"].max())

    sales = sales[sales["cat_id"] == category].reset_index(drop=True)
    d_cols = [c for c in sales.columns if c.startswith("d_")]
    last_obs = int(d_cols[-1][2:])
    units = sales[d_cols].to_numpy(dtype=np.float32)
    sold = units > 0
    never = ~sold.any(axis=1)
    first = np.argmax(sold, axis=1) + 1
    recent = units[:, -365:].sum(axis=1)

    order = [i for i in np.argsort(-recent) if not never[i]]
    top = order[:n_series]
    top_set = set(top)
    # Cold starts: series that are new at each origin, so every backtest has a real cold-start slice.
    # Ranked by sales up to that origin only (no peeking at the test window).
    taken, per = set(top_set), int(np.ceil(n_cold_start_extra / len(origins)))
    for o in origins:
        pre = units[:, :o].sum(axis=1)
        cand = [i for i in np.argsort(-pre) if not never[i] and i not in taken
                and o - cold_start_days + 1 < first[i] <= o - min_history_days + 1]
        taken |= set(cand[:per])
    keep = np.array(sorted(taken))

    sales = sales.iloc[keep].reset_index(drop=True)
    units, first = units[keep], first[keep]
    n = len(sales)

    Y = np.full((n, D), np.nan, dtype=np.float32)
    Y[:, :last_obs] = units

    meta = sales[["id"] + HIER_COLS].copy()
    meta["key"] = meta["item_id"] + "_" + meta["store_id"]
    prices["key"] = prices["item_id"] + "_" + prices["store_id"]
    piv = prices[prices["key"].isin(meta["key"])].pivot(index="key", columns="wm_yr_wk", values="sell_price")
    price = piv.reindex(index=meta["key"], columns=cal["wm_yr_wk"].to_numpy()).to_numpy(dtype=np.float32)

    codes = {c: pd.factorize(meta[c])[0].astype(np.int32) for c in ["item_id", "dept_id", "store_id"]}
    codes["state"] = meta["state_id"].map({"CA": 0, "TX": 1, "WI": 2}).to_numpy()
    return M5Data(meta=meta, Y=Y, price=price, cal=cal, first_sale=first.astype(int), last_obs=last_obs, codes=codes)


def active(data, origin, min_history_days):
    """Series that have sold for at least min_history_days by the origin."""
    return np.where(data.first_sale <= origin - min_history_days + 1)[0]
