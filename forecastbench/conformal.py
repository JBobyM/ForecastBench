"""Conformal recalibration of prediction intervals (the "self-calibrating" part).

Bottom level: normalized conformalized quantile regression (CQR), pooled by demand group.
Scores are divided by each series' MASE scale so one adjustment works across volumes.
Calibration always uses the 28 days that end at the forecast origin, never the test window.
"""
import numpy as np


def split_quantile(scores, alpha):
    v = np.asarray(scores, dtype=float).ravel()
    v = np.sort(v[np.isfinite(v)])
    if len(v) == 0:
        return 0.0
    k = int(np.ceil((len(v) + 1) * (1 - alpha)))
    return float(v[min(k, len(v)) - 1])


def cqr_fit(lo, hi, y, scale, groups, alpha, min_points=200):
    s = np.maximum(lo - y, y - hi) / scale[:, None]
    qhat = {"__all__": split_quantile(s, alpha)}
    for g in np.unique(groups):
        sg = s[groups == g]
        if sg.size >= min_points:          # small groups fall back to the pooled value
            qhat[g] = split_quantile(sg, alpha)
    return qhat


def cqr_apply(lo, hi, scale, groups, qhat):
    adj = np.array([qhat.get(g, qhat["__all__"]) for g in groups])[:, None] * scale[:, None]
    new_lo = np.clip(lo - adj, 0, None)
    return new_lo, np.maximum(hi + adj, new_lo)
