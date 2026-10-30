# ForecastBench

A fixed, reproducible demand-forecasting accuracy benchmark on the M5 (Walmart) hierarchy, plus a live forecasting API that serves the winning setup.

The benchmark is the product. Every model is scored the same way, on the same series, on the same frozen backtest, and a quality gate decides which ones are good enough to ship.

## What it compares

| Family | Models |
|---|---|
| Classical | seasonal naive (baseline), Croston-SBA, TSB |
| ML | global LightGBM: Tweedie mean + quantile models (0.1 / 0.5 / 0.9) |
| Pretrained | Chronos-Bolt small, zero-shot (no training on M5) |
| Routed system | each series goes to one model by its demand segment |

**Segment-aware routing.** Each series is classified at the forecast origin with the Syntetos-Boylan rule (ADI 1.32, CV² 0.49) into smooth, erratic, intermittent or lumpy. Series with fewer than 120 days since their first sale are cold starts. Routing is fixed in the config before any test result is seen: cold start goes to Chronos, intermittent and lumpy go to TSB, smooth and erratic go to LightGBM. The routed system is benchmarked against every single model.

**Self-calibrating intervals.** Raw 80% intervals are recalibrated with normalized conformalized quantile regression, pooled by segment. Calibration always uses the 28 days that end at the forecast origin, so the test window is never used to calibrate.

**Hierarchical coherence.** Point forecasts are reconciled bottom-up across 9 levels (item-store, item-state, item, store-dept, state-dept, dept, store, state, total). Quantiles don't add up, so each aggregate level gets its own conformal interval around the coherent point forecast.

**Quality gate.** A model is promoted only if it beats seasonal naive on WRMSSE **and** its recalibrated 80% coverage lands in [75%, 85%].

## Protocol (frozen in configs/benchmark.yaml)

- Data: M5 FOODS category, top 2,500 item-stores by unit sales over the last year, plus the top 300 that launched in the last year (so cold starts are actually represented).
- Backtest: 3 rolling origins (d_1857, d_1885, d_1913), 28-day horizon each.
- Metrics: WRMSSE overall and by level, MASE, scaled pinball loss, raw and recalibrated 80% coverage, interval width, seconds and dollars per 1,000 forecasts. Every metric is also reported per segment, with cold start as its own slice.
- WRMSSE weights are recomputed for this subset, so the scores are **not** comparable to the M5 Kaggle leaderboard.
- manifest.json records the config, SHA-256 of the raw files, package versions and host for every run.

## Results

Run the benchmark, then paste artifacts/leaderboard.csv here.

## Run it (uOttawa ganymede)

~~~bash
# 1. data: download from https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
#    and upload sales_train_evaluation.csv, calendar.csv, sell_prices.csv to data/raw/
# 2. environment (once)
bash setup_env.sh
# 3. smoke test: 300 series, 1 origin (writes artifacts_quick/)
bash submit.sh --quick
# 4. full benchmark: 4 Slurm array tasks in parallel, then the leaderboard job
bash submit.sh
cat artifacts/leaderboard.csv
~~~

Anywhere else: install requirements.txt and run python -m forecastbench.benchmark (runs every origin in sequence).

## Serve it

The API reads the precomputed forecasts in artifacts/serve/ (committed after the full run), so no model runs per request.

~~~bash
uvicorn app.main:app --port 8080
gcloud run deploy forecastbench --source . --region northamerica-northeast1 --allow-unauthenticated --memory 1Gi
~~~

Endpoints: / (UI), /health, /api/benchmark, /api/series?q=&group=, /api/forecast/{id}. Each forecast comes with the model that served it, why the series was routed there, and the top LightGBM drivers when LightGBM served it.

## Layout

~~~
configs/benchmark.yaml   the benchmark definition
forecastbench/           data, segments, classical, lgbm, chronos_arm, conformal, hierarchy, metrics, benchmark
app/                     FastAPI service + UI
run_benchmark.sbatch, aggregate.sbatch, submit.sh, setup_env.sh   ganymede jobs
Dockerfile               serving image
~~~

## Limits

- One category and about 2,800 series, not all 30,490. The pipeline itself isn't tied to that size.
- Cost is compute time at the Cloud Run vCPU price in the config (check https://cloud.google.com/run/pricing). LightGBM's cost includes training. GPU time isn't priced.
- Cold-start series have short but nonzero history; brand-new items with no sales are out of scope.
