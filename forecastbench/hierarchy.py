"""M5 hierarchy for one category: aggregation matrices, bottom-up coherence, per-level intervals.

Point forecasts are reconciled bottom-up (aggregates = sums of item-store forecasts).
Quantiles do not add up, so each aggregate level gets its own conformal interval around
the coherent point forecast.
"""
import numpy as np
from scipy import sparse

from . import conformal, metrics

LEVELS = {
    "total": [], "state": ["state_id"], "store": ["store_id"], "dept": ["dept_id"],
    "state_dept": ["state_id", "dept_id"], "store_dept": ["store_id", "dept_id"],
    "item": ["item_id"], "item_state": ["item_id", "state_id"], "item_store": ["id"],
}


def prepare(meta_sub, hist, rev):
    """Per level: aggregation matrix, scales from aggregated history, revenue weights."""
    meta_sub = meta_sub.reset_index(drop=True)
    out = {}
    for lvl, cols in LEVELS.items():
        codes = meta_sub.groupby(cols, sort=False).ngroup().to_numpy() if cols else np.zeros(len(meta_sub), int)
        M = sparse.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))),
                              shape=(codes.max() + 1, len(codes)))
        H = M @ hist
        w = M @ rev
        w = w / w.sum() if w.sum() > 0 else np.full(M.shape[0], 1.0 / M.shape[0])
        out[lvl] = {"M": M, "mscale": metrics.mase_scale(H), "rscale": metrics.rmsse_scale(H), "w": w}
    return out


def level_qhat(agg, point, y, alpha):
    """Split-conformal half-width per aggregate level, from the calibration window."""
    q = {}
    for lvl, a in agg.items():
        if lvl == "item_store":
            continue
        s = np.abs(a["M"] @ y - a["M"] @ point) / a["mscale"][:, None]
        q[lvl] = conformal.split_quantile(s, alpha)
    return q


def level_coverage(agg, point, y, qhat):
    cov = {}
    for lvl, q in qhat.items():
        a = agg[lvl]
        F, A = a["M"] @ point, a["M"] @ y
        half = q * a["mscale"][:, None]
        cov[lvl] = metrics.coverage(A, np.maximum(F - half, 0), F + half)
    return cov
