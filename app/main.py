"""ForecastBench API: serves precomputed, recalibrated forecasts and the benchmark results.

No model runs at request time.
"""
import json
import math
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parent
SERVE = Path(os.environ.get("SERVE_DIR", ROOT.parent / "artifacts" / "serve"))

series = pd.read_parquet(SERVE / "series.parquet").set_index("id")
forecasts = dict(tuple(pd.read_parquet(SERVE / "forecasts.parquet").groupby("id")))
history = dict(tuple(pd.read_parquet(SERVE / "history.parquet").groupby("id")))


def clean(x):
    """Make values JSON-safe (NaN and inf become null)."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


benchmark = clean(json.load(open(SERVE / "benchmark.json")))
app = FastAPI(title="ForecastBench", version="1.0")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "series": len(series)}


@app.get("/api/benchmark")
def get_benchmark():
    return benchmark


@app.get("/api/series")
def list_series(q: str = "", group: str = "", limit: int = Query(50, le=500)):
    df = series
    if q:
        df = df[df.index.str.contains(q, case=False, regex=False)]
    if group:
        df = df[df["group"] == group]
    return [{"id": i, "group": g, "model": m} for i, g, m in zip(df.index[:limit], df["group"][:limit], df["model"][:limit])]


@app.get("/api/forecast/{series_id}")
def get_forecast(series_id: str):
    if series_id not in series.index:
        raise HTTPException(404, f"unknown series {series_id}")
    s = series.loc[series_id]
    fc, hist = forecasts[series_id], history[series_id]
    return clean({
        "id": series_id, **{k: s[k] for k in ["item_id", "store_id", "dept_id", "group", "segment", "model", "reason"]},
        "adi": float(s["adi"]), "cv2": float(s["cv2"]), "age_days": int(s["age_days"]),
        "drivers": json.loads(s["drivers"]),
        "forecast": fc[["date", "mean", "lo", "hi"]].round(3).to_dict("records"),
        "history": hist[["date", "y"]].to_dict("records"),
    })
