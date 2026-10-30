"""Accuracy metrics. Scales are computed from history since the series' first sale (M5 convention)."""
import numpy as np


def _diffs(H):
    first = np.argmax(H > 0, axis=1)
    D = np.diff(H, axis=1)
    valid = np.arange(H.shape[1] - 1)[None, :] >= first[:, None]
    return D, valid


def mase_scale(H):
    D, v = _diffs(H)
    return np.maximum((np.abs(D) * v).sum(1) / np.maximum(v.sum(1), 1), 1e-2)


def rmsse_scale(H):
    D, v = _diffs(H)
    return np.maximum((D ** 2 * v).sum(1) / np.maximum(v.sum(1), 1), 1e-4)


def rmsse(y, f, rscale):
    return np.sqrt(((y - f) ** 2).mean(1) / rscale)


def wrmsse(agg, y, f):
    """M5 WRMSSE over the levels in agg (weights recomputed for this subset). Returns (overall, per level)."""
    per = {lvl: float(a["w"] @ rmsse(a["M"] @ y, a["M"] @ f, a["rscale"])) for lvl, a in agg.items()}
    return float(np.mean(list(per.values()))), per


def mase(y, f, scale):
    return float((np.abs(y - f).mean(1) / scale).mean())


def scaled_pinball(y, Q, qs, scale):
    e = y[..., None] - Q
    qs = np.asarray(qs)
    loss = np.maximum(qs * e, (qs - 1) * e)
    return float((loss.mean(axis=(1, 2)) / scale).mean())


def coverage(y, lo, hi):
    return float(((y >= lo) & (y <= hi)).mean())


def scaled_width(lo, hi, scale):
    return float(((hi - lo) / scale[:, None]).mean())
