"""Run ForecastBench.

  python -m forecastbench.benchmark --task 0     one origin (Slurm array task). Tasks 0..2 = test origins, 3 = serve origin
  python -m forecastbench.benchmark --aggregate  leaderboard + quality gate from the finished tasks
  python -m forecastbench.benchmark              everything, one after another
  add --quick for a 300-series, 1-origin smoke test (writes to artifacts_quick/)
"""
import argparse
import hashlib
import json
import os
import platform
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import chronos_arm, classical, conformal, hierarchy, lgbm, metrics, segments
from .data import active, load_m5

MODEL_DESC = {
    "seasonal_naive": "seasonal naive, which repeats last week",
    "croston_sba": "Croston-SBA, a classic intermittent-demand method",
    "tsb": "TSB, which forecasts how often the item sells times its typical sale size",
    "lightgbm": "a global LightGBM model trained on all series at once",
    "chronos": "Chronos-Bolt, a pretrained forecasting model used zero-shot",
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_config(path, quick):
    cfg = yaml.safe_load(open(path))
    if quick:
        cfg["data"]["n_series"], cfg["data"]["n_cold_start_extra"] = 300, 60
        cfg["backtest"]["test_origins"] = cfg["backtest"]["test_origins"][-1:]
        cfg["models"]["lightgbm"]["num_boost_round"] = 100
        cfg["output_dir"] += "_quick"
    return cfg


def registry(cfg):
    h, qs, c = cfg["backtest"]["horizon"], cfg["quantiles"], cfg["models"]["classical"]
    return {
        "seasonal_naive": lambda d, i, o, drv: classical.seasonal_naive(d, i, o, h, qs, **c),
        "croston_sba": lambda d, i, o, drv: classical.croston_sba(d, i, o, h, qs, **c),
        "tsb": lambda d, i, o, drv: classical.tsb(d, i, o, h, qs, **c),
        "lightgbm": lambda d, i, o, drv: lgbm.forecast(d, i, o, h, qs, cfg["models"]["lightgbm"], cfg["seed"], drv),
        "chronos": lambda d, i, o, drv: chronos_arm.forecast(d, i, o, h, qs, cfg["models"]["chronos"]),
    }


def make_side(data, cfg, origin, with_actuals):
    h = cfg["backtest"]["horizon"]
    idx = active(data, origin, cfg["data"]["min_history_days"])
    seg = segments.classify(data, idx, origin, **cfg["segments"])
    hist = np.nan_to_num(data.Y[idx, :origin])
    rev = np.nansum(data.Y[idx, origin - 28:origin] * np.nan_to_num(data.price[idx, origin - 28:origin]), axis=1)
    agg = hierarchy.prepare(data.meta.iloc[idx], hist, rev)
    bottom = agg["item_store"]
    return {"origin": origin, "idx": idx, "seg": seg, "groups": seg["group"].to_numpy(), "agg": agg,
            "scale": bottom["mscale"], "rscale": bottom["rscale"], "w": bottom["w"],
            "y": data.Y[idx, origin:origin + h].astype(np.float64) if with_actuals else None}


def interval_index(cfg):
    qs, a = np.asarray(cfg["quantiles"]), cfg["interval_alpha"]
    lo_i, hi_i = int(np.argmin(abs(qs - a / 2))), int(np.argmin(abs(qs - (1 - a / 2))))
    assert np.isclose(qs[lo_i], a / 2) and np.isclose(qs[hi_i], 1 - a / 2), "interval quantiles missing"
    return lo_i, hi_i


def run_origin(data, cfg, origin, evaluate):
    h, alpha = cfg["backtest"]["horizon"], cfg["interval_alpha"]
    lo_i, hi_i = interval_index(cfg)
    cal = make_side(data, cfg, origin - h, True)       # calibration window ends at the origin
    test = make_side(data, cfg, origin, evaluate)
    chronos_arm.load(cfg["models"]["chronos"])          # keep the one-time model load out of the timings

    raw = {}
    for name, fn in registry(cfg).items():
        t0 = time.perf_counter()
        fc = {"cal": fn(data, cal["idx"], cal["origin"], False),
              "test": fn(data, test["idx"], test["origin"], not evaluate)}   # drivers only for serving
        secs = time.perf_counter() - t0
        raw[name] = {**fc, "sec_per_1k": 1000 * secs / (len(cal["idx"]) + len(test["idx"]))}
        log(f"origin {origin}: {name} done in {secs:.0f}s")

    # Routed system: each series goes to one model by its demand group (rule fixed in the config)
    route, nq = cfg["routing"], len(cfg["quantiles"])
    routed = {}
    for tag, side in (("cal", cal), ("test", test)):
        choice = np.array([route[g] for g in side["groups"]])
        point, Q = np.zeros((len(choice), h)), np.zeros((len(choice), h, nq))
        for m in np.unique(choice):
            mask = choice == m
            point[mask], Q[mask] = raw[m][tag]["point"][mask], raw[m][tag]["Q"][mask]
        routed[tag] = {"point": point, "Q": Q}
        routed[tag + "_choice"] = choice
    share = pd.Series(routed["test_choice"]).value_counts(normalize=True)
    routed["sec_per_1k"] = float(sum(share[m] * raw[m]["sec_per_1k"] for m in share.index))
    raw["routed"] = routed

    results = {}
    for name, r in raw.items():
        qhat = conformal.cqr_fit(r["cal"]["Q"][..., lo_i], r["cal"]["Q"][..., hi_i], cal["y"],
                                 cal["scale"], cal["groups"], alpha)
        lo, hi = conformal.cqr_apply(r["test"]["Q"][..., lo_i], r["test"]["Q"][..., hi_i],
                                     test["scale"], test["groups"], qhat)
        results[name] = {"point": r["test"]["point"], "Q": r["test"]["Q"], "lo": lo, "hi": hi,
                         "level_q": hierarchy.level_qhat(cal["agg"], r["cal"]["point"], cal["y"], alpha),
                         "sec_per_1k": r["sec_per_1k"]}
    return test, raw, results


def score(name, res, test, cfg):
    y, qs, o = test["y"], cfg["quantiles"], test["origin"]
    lo_i, hi_i = interval_index(cfg)
    overall, per_level = metrics.wrmsse(test["agg"], y, res["point"])
    row = {"origin": o, "model": name, "wrmsse": overall,
           "mase": metrics.mase(y, res["point"], test["scale"]),
           "scaled_pinball": metrics.scaled_pinball(y, res["Q"], qs, test["scale"]),
           "coverage_raw": metrics.coverage(y, res["Q"][..., lo_i], res["Q"][..., hi_i]),
           "coverage_cal": metrics.coverage(y, res["lo"], res["hi"]),
           "width_cal": metrics.scaled_width(res["lo"], res["hi"], test["scale"]),
           "sec_per_1k": res["sec_per_1k"], "n_series": int(len(y))}
    lvl_cov = hierarchy.level_coverage(test["agg"], res["point"], y, res["level_q"])
    levels = [{"origin": o, "model": name, "level": lvl, "wrmsse": v,
               "coverage_cal": lvl_cov.get(lvl, row["coverage_cal"])} for lvl, v in per_level.items()]
    r = metrics.rmsse(y, res["point"], test["rscale"])
    segs = []
    for g in np.unique(test["groups"]):
        m = test["groups"] == g
        w = test["w"][m]
        segs.append({"origin": o, "model": name, "group": g, "n_series": int(m.sum()),
                     "wrmsse": float(w @ r[m] / w.sum()) if w.sum() > 0 else float(r[m].mean()),
                     "mase": metrics.mase(y[m], res["point"][m], test["scale"][m]),
                     "coverage_cal": metrics.coverage(y[m], res["lo"][m], res["hi"][m])})
    return row, levels, segs


def reason(s, model):
    if s.group == "cold_start":
        why = f"Only {s.age} days of sales history, too short for the lag features other models need"
    elif s.segment == "intermittent":
        why = f"Sells on about 1 day in {s.adi:.1f}, with fairly steady sale sizes (CV² {s.cv2:.2f})"
    elif s.segment == "lumpy":
        why = f"Sells on about 1 day in {s.adi:.1f}, and sale sizes swing a lot (CV² {s.cv2:.2f})"
    elif s.segment == "smooth":
        why = f"Sells almost every day with stable volumes (CV² {s.cv2:.2f})"
    else:
        why = f"Sells almost every day, but volumes swing a lot (CV² {s.cv2:.2f})"
    return f"{why}. Routed to {MODEL_DESC[model]}."


def write_serving(data, cfg, test, raw, results, out):
    h, o, idx = cfg["backtest"]["horizon"], test["origin"], test["idx"]
    res, choice, seg = results["routed"], raw["routed"]["test_choice"], test["seg"]
    drivers = raw["lightgbm"]["test"].get("drivers")
    series = data.meta.iloc[idx][["id", "item_id", "dept_id", "store_id", "state_id"]].reset_index(drop=True)
    series["group"], series["segment"] = seg["group"].to_numpy(), seg["segment"].to_numpy()
    series["adi"] = seg["adi"].replace(np.inf, np.nan).to_numpy()
    series["cv2"], series["age_days"], series["model"] = seg["cv2"].to_numpy(), seg["age"].to_numpy(), choice
    series["reason"] = [reason(s, m) for s, m in zip(seg.itertuples(), choice)]
    series["drivers"] = [json.dumps(drivers[k]) if m == "lightgbm" and drivers else "[]" for k, m in enumerate(choice)]

    fdates = data.cal["date"].iloc[o:o + h].dt.strftime("%Y-%m-%d").to_numpy()
    fc = pd.DataFrame({"id": np.repeat(series["id"].to_numpy(), h), "date": np.tile(fdates, len(idx)),
                       "mean": res["point"].ravel(), "lo": res["lo"].ravel(), "hi": res["hi"].ravel()})
    hdates = data.cal["date"].iloc[o - 112:o].dt.strftime("%Y-%m-%d").to_numpy()
    hist = pd.DataFrame({"id": np.repeat(series["id"].to_numpy(), 112), "date": np.tile(hdates, len(idx)),
                         "y": data.Y[idx, o - 112:o].ravel()})
    serve = out / "serve"
    serve.mkdir(parents=True, exist_ok=True)
    series.to_parquet(serve / "series.parquet", index=False)
    fc.to_parquet(serve / "forecasts.parquet", index=False)
    hist.to_parquet(serve / "history.parquet", index=False)
    log(f"serve files written for {len(idx)} series")


def run_task(cfg, task):
    origins = cfg["backtest"]["test_origins"] + [cfg["backtest"]["serve_origin"]]
    origin, evaluate = origins[task], task < len(cfg["backtest"]["test_origins"])
    out = Path(cfg["output_dir"])
    log(f"task {task}: origin {origin} ({'backtest' if evaluate else 'serving'})")
    data = load_m5(**cfg["data"])
    log(f"loaded {len(data.meta)} series")
    test, raw, results = run_origin(data, cfg, origin, evaluate)
    if not evaluate:
        write_serving(data, cfg, test, raw, results, out)
        return
    rows, levels, segs = [], [], []
    for name, res in results.items():
        r, lv, sg = score(name, res, test, cfg)
        rows.append(r); levels += lv; segs += sg
    (out / "runs").mkdir(parents=True, exist_ok=True)
    json.dump({"rows": rows, "levels": levels, "segments": segs}, open(out / "runs" / f"origin_{origin}.json", "w"))
    log(f"origin {origin} scored")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def aggregate(cfg, config_path):
    out = Path(cfg["output_dir"])
    runs = [out / "runs" / f"origin_{o}.json" for o in cfg["backtest"]["test_origins"]]
    missing = [str(p) for p in runs if not p.exists()]
    if missing:
        raise SystemExit(f"missing backtest results: {missing}")
    parts = [json.load(open(p)) for p in runs]
    rows = pd.DataFrame([r for p in parts for r in p["rows"]])
    levels = pd.DataFrame([r for p in parts for r in p["levels"]])
    segs = pd.DataFrame([r for p in parts for r in p["segments"]])

    lb = rows.drop(columns="origin").groupby("model").mean()
    lb["coverage_cal_min"] = rows.groupby("model")["coverage_cal"].min()
    base = lb.loc[cfg["gate"]["baseline"], "wrmsse"]
    lo, hi = cfg["gate"]["coverage_band"]
    lb["beats_baseline"] = lb["wrmsse"] < base
    lb["coverage_ok"] = lb["coverage_cal"].between(lo, hi)
    lb["promoted"] = lb["beats_baseline"] & lb["coverage_ok"]
    vcpus = cfg["models"]["lightgbm"]["params"]["num_threads"]
    lb["usd_per_1k"] = lb["sec_per_1k"] * vcpus * cfg["cost"]["usd_per_vcpu_second"]
    lb = lb.sort_values("wrmsse").reset_index()

    by_level = levels.drop(columns="origin").groupby(["model", "level"]).mean().reset_index()
    by_segment = segs.drop(columns="origin").groupby(["model", "group"]).mean().reset_index()
    lb.to_csv(out / "leaderboard.csv", index=False)
    by_level.to_csv(out / "by_level.csv", index=False)
    by_segment.to_csv(out / "by_segment.csv", index=False)

    raw_dir = Path(cfg["data"]["raw_dir"])
    versions = {}
    for pkg in ["numpy", "pandas", "lightgbm", "torch", "chronos-forecasting", "scipy"]:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    manifest = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "config_file": str(config_path), "config": cfg,
                "data_sha256": {f: sha256(raw_dir / f) for f in
                                ["sales_train_evaluation.csv", "calendar.csv", "sell_prices.csv"]},
                "versions": versions, "python": platform.python_version(), "host": platform.node(),
                "cost_note": f"usd_per_1k = sec_per_1k x {vcpus} vCPUs x config price; GPU time not priced"}
    json.dump(manifest, open(out / "manifest.json", "w"), indent=2, default=str)

    serve = out / "serve"
    serve.mkdir(parents=True, exist_ok=True)
    gate = {"baseline": cfg["gate"]["baseline"], "coverage_band": [lo, hi],
            "rule": "promoted = beats baseline WRMSSE and recalibrated 80% coverage within the band"}
    json.dump({"gate": gate, "leaderboard": lb.to_dict("records"),
               "by_segment": by_segment.to_dict("records"), "by_level": by_level.to_dict("records")},
              open(serve / "benchmark.json", "w"), default=str)
    pd.set_option("display.width", 200)
    print(lb[["model", "wrmsse", "mase", "scaled_pinball", "coverage_raw", "coverage_cal",
              "sec_per_1k", "usd_per_1k", "promoted"]].round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/benchmark.yaml")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--task", type=int)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.quick)
    np.random.seed(cfg["seed"])
    if args.task is not None:
        run_task(cfg, args.task)
    elif args.aggregate:
        aggregate(cfg, args.config)
    else:
        for t in range(len(cfg["backtest"]["test_origins"]) + 1):
            run_task(cfg, t)
        aggregate(cfg, args.config)


if __name__ == "__main__":
    main()
