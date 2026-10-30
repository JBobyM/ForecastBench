"""Classical models: seasonal naive, Croston-SBA, TSB.

Quantiles come from empirical in-sample residuals. They are deliberately left uncalibrated;
conformal recalibration (conformal.py) is what makes the intervals honest.
"""
import warnings

import numpy as np


def _history(data, idx, origin, L):
    H = data.Y[idx, origin - L:origin].astype(np.float64)
    days = np.arange(origin - L, origin) + 1
    H[days[None, :] < data.first_sale[idx][:, None]] = np.nan   # before launch = unknown, not zero
    return H


def _resid_quantiles(point, resid, qs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        off = np.nan_to_num(np.nanquantile(resid, qs, axis=1).T)
    Q = point[:, :, None] + off[:, None, :]
    return np.clip(np.sort(Q, axis=2), 0, None)


def _init(H):
    valid, occ = ~np.isnan(H), H > 0
    n_valid, n_occ = valid.sum(1).clip(min=1), occ.sum(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        size = np.nan_to_num(np.nanmean(np.where(occ, H, np.nan), axis=1))
    return valid, occ, n_valid, n_occ, size


def seasonal_naive(data, idx, origin, h, qs, resid_window, **_):
    H = _history(data, idx, origin, resid_window + 7)
    last7 = np.nan_to_num(H[:, -7:])
    point = np.tile(last7, int(np.ceil(h / 7)))[:, :h]
    resid = H[:, 7:] - H[:, :-7]
    return {"point": point, "Q": _resid_quantiles(point, resid, qs)}


def tsb(data, idx, origin, h, qs, window, resid_window, tsb_alpha, tsb_beta, **_):
    """Teunter-Syntetos-Babai: smooths sale probability and sale size separately."""
    H = _history(data, idx, origin, window)
    valid, occ, n_valid, n_occ, z = _init(H)
    p = n_occ / n_valid
    fitted = np.full(H.shape, np.nan)
    for t in range(H.shape[1]):
        fitted[:, t] = p * z
        v, o = valid[:, t], occ[:, t]
        p = np.where(v, p + tsb_beta * (o - p), p)
        z = np.where(v & o, z + tsb_alpha * (np.nan_to_num(H[:, t]) - z), z)
    point = np.repeat((p * z)[:, None], h, axis=1)
    return {"point": point, "Q": _resid_quantiles(point, (H - fitted)[:, -resid_window:], qs)}


def croston_sba(data, idx, origin, h, qs, window, resid_window, croston_alpha, **_):
    """Croston with the Syntetos-Boylan bias correction."""
    a = croston_alpha
    H = _history(data, idx, origin, window)
    valid, occ, n_valid, n_occ, z = _init(H)
    q = n_valid / np.maximum(n_occ, 1)
    k = np.ones(len(idx))
    fitted = np.full(H.shape, np.nan)
    for t in range(H.shape[1]):
        fitted[:, t] = (1 - a / 2) * z / q
        v, o = valid[:, t], occ[:, t]
        hit = v & o
        z = np.where(hit, z + a * (np.nan_to_num(H[:, t]) - z), z)
        q = np.where(hit, q + a * (k - q), q)
        k = np.where(hit, 1, np.where(v, k + 1, k))
    point = np.repeat(((1 - a / 2) * z / q)[:, None], h, axis=1)
    return {"point": point, "Q": _resid_quantiles(point, (H - fitted)[:, -resid_window:], qs)}
