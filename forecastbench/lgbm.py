"""Global LightGBM: one model for every series and every horizon.

All lag and rolling features end 28 days before the target day, so a single model
forecasts the full 28-day horizon with no recursion and no leakage.
"""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

FEATURES = ["lag_28", "lag_35", "lag_42", "lag_49", "lag_56", "rmean_7", "rmean_28", "rmean_56",
            "rstd_28", "price", "price_rel", "price_change", "wday", "month", "event", "snap",
            "age", "dept", "store", "item"]
CATEGORICAL = ["dept", "store", "item"]
LABELS = {
    "lag_28": "sales 4 weeks earlier", "lag_35": "sales 5 weeks earlier", "lag_42": "sales 6 weeks earlier",
    "lag_49": "sales 7 weeks earlier", "lag_56": "sales 8 weeks earlier",
    "rmean_7": "recent weekly average", "rmean_28": "recent monthly average", "rmean_56": "2-month average",
    "rstd_28": "recent volatility", "price": "price", "price_rel": "price vs. its usual level",
    "price_change": "price change vs. last week", "wday": "day of week", "month": "month",
    "event": "holiday or event", "snap": "SNAP benefit days", "age": "time since launch",
    "dept": "department", "store": "store", "item": "item",
}


def _features(data, idx, days):
    n, p = len(idx), np.asarray(days) - 1
    Y0 = np.nan_to_num(data.Y[idx]).astype(np.float64)
    C = np.zeros((n, Y0.shape[1] + 1)); C[:, 1:] = np.cumsum(Y0, 1)
    C2 = np.zeros_like(C); C2[:, 1:] = np.cumsum(Y0 ** 2, 1)

    def rsum(Cm, w):  # sum over the w days ending 28 days before each target day
        return Cm[:, p - 27] - Cm[:, p - 27 - w]

    f = {f"lag_{k}": Y0[:, p - k] for k in (28, 35, 42, 49, 56)}
    for w in (7, 28, 56):
        f[f"rmean_{w}"] = rsum(C, w) / w
    f["rstd_28"] = np.sqrt(np.maximum(rsum(C2, 28) / 28 - f["rmean_28"] ** 2, 0))
    P = data.price[idx]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f["price"] = P[:, p]
        f["price_rel"] = P[:, p] / np.nanmax(P, axis=1, keepdims=True)
        f["price_change"] = P[:, p] / P[:, p - 7] - 1
    cal = data.cal.iloc[p]
    for col in ("wday", "month", "event"):
        f[col] = np.broadcast_to(cal[col].to_numpy()[None, :], (n, len(p)))
    snap = data.cal[["snap_CA", "snap_TX", "snap_WI"]].to_numpy()[p]
    f["snap"] = snap[:, data.codes["state"][idx]].T
    f["age"] = (p + 1)[None, :] - data.first_sale[idx][:, None]
    for key, col in (("dept", "dept_id"), ("store", "store_id"), ("item", "item_id")):
        f[key] = np.broadcast_to(data.codes[col][idx][:, None], (n, len(p)))

    X = pd.DataFrame({k: np.asarray(f[k]).ravel() for k in FEATURES})
    for c in FEATURES:
        X[c] = X[c].astype(np.int32 if c in CATEGORICAL else np.float32)
    return X, np.repeat(np.arange(n), len(p)), np.tile(p + 1, n)


def forecast(data, idx, origin, h, qs, cfg, seed, drivers=False):
    n = len(idx)
    tr_days = np.arange(origin - cfg["train_days"] + 1, origin + 1)
    X, s, d = _features(data, idx, tr_days)
    y = data.Y[idx][:, tr_days - 1].ravel()
    keep = d >= data.first_sale[idx][s]                  # train only on days after launch
    ds = lgb.Dataset(X[keep], y[keep], categorical_feature=CATEGORICAL, free_raw_data=False)
    Xte, _, _ = _features(data, idx, np.arange(origin + 1, origin + h + 1))

    base = {**cfg["params"], "seed": seed, "deterministic": True, "force_row_wise": True, "verbose": -1}

    def fit(extra):
        return lgb.train({**base, **extra}, ds, num_boost_round=cfg["num_boost_round"])

    mean_model = fit({"objective": "tweedie", "tweedie_variance_power": cfg["tweedie_power"]})
    point = np.clip(mean_model.predict(Xte), 0, None).reshape(n, h)
    Q = np.stack([fit({"objective": "quantile", "alpha": q}).predict(Xte).reshape(n, h) for q in qs], axis=2)
    out = {"point": point, "Q": np.clip(np.sort(Q, axis=2), 0, None)}

    if drivers:  # SHAP-style contributions of the mean model, averaged over the horizon
        contrib = mean_model.predict(Xte, pred_contrib=True)[:, :-1].reshape(n, h, -1).mean(axis=1)
        top = np.argsort(-np.abs(contrib), axis=1)[:, :3]
        out["drivers"] = [[{"feature": LABELS[FEATURES[j]], "effect": "raises" if contrib[k, j] > 0 else "lowers"}
                           for j in top[k]] for k in range(n)]
    return out
