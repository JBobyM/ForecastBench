# ForecastBench

A demand-forecasting benchmark on the M5 Walmart data, with a small API that serves the forecasts.

Live demo: https://forecastbench-436839971921.northamerica-northeast1.run.app

The idea is simple. Every model goes through the same test: same series, same backtest windows, same metrics. A model only gets promoted if it beats a seasonal naive forecast and its 80% prediction intervals actually cover about 80% of what happens.

![Pipeline](docs/pipeline.png)

## Results

Three forecast dates, 28 days ahead each, about 2,530 item-store series. Lower is better, except coverage (target 80%).

| Model | WRMSSE | MASE | Pinball (scaled) | Coverage before | Coverage after | Runtime (s / 1k series) | Gate |
|---|---|---|---|---|---|---|---|
| LightGBM, global | **0.650** | 1.020 | **0.323** | 78.0% | 80.7% | 21.3 | promoted |
| Routed by segment | 0.731 | 1.027 | 0.332 | 76.8% | 80.5% | 14.2 | promoted |
| Chronos-Bolt, zero-shot | 0.949 | **1.008** | 0.328 | 79.5% | 80.4% | 3.5 | promoted |
| Seasonal naive | 0.977 | 1.248 | 0.402 | 80.2% | 81.4% | 0.04 | baseline |
| TSB | 1.071 | 1.040 | 0.339 | 74.9% | 80.1% | 0.05 | no |
| Croston-SBA | 1.075 | 1.086 | 0.346 | 78.0% | 80.3% | 0.05 | no |

What I took from it:

- LightGBM wins clearly, 33% better than seasonal naive on WRMSSE. It's also the best on intermittent and lumpy items, which I didn't expect.
- Chronos-Bolt was never trained on M5 and still passes the gate, at a sixth of LightGBM's runtime. It's the best on smooth and erratic series but falls behind once forecasts are summed up the hierarchy.
- Routing each series to a "specialist" model didn't pay off. TSB on intermittent and lumpy items loses to LightGBM. The one route that works is new items to Chronos (0.739 vs 0.768).
- After conformal recalibration every model lands between 80% and 81.4% coverage, so the gate ends up deciding on accuracy alone.
- New items are rare in M5 FOODS (about 21 per forecast date), so take that slice as a rough signal.

Per-level and per-segment numbers are in artifacts/.

## How it works

- **Data.** The FOODS category: the 2,500 best-selling item-stores, plus items that are new at each forecast date.
- **Segments.** Each series is labelled smooth, erratic, intermittent or lumpy (Syntetos-Boylan), or new if it has under 120 days of sales.
- **Models.** Seasonal naive, Croston-SBA, TSB, one global LightGBM (Tweedie mean plus 10/50/90% quantiles), Chronos-Bolt small used zero-shot, and a routed setup: new items go to Chronos, intermittent and lumpy to TSB, the rest to LightGBM. The routing rule was fixed before seeing any results.
- **Intervals.** Recalibrated with conformal prediction on the 28 days just before each forecast date, never on the test window.
- **Hierarchy.** Item-store forecasts are summed bottom-up so stores, states and the total add up. WRMSSE is averaged over 9 levels. The weights are recomputed for this subset, so the scores aren't comparable to the Kaggle leaderboard.

Everything that defines the benchmark is in configs/benchmark.yaml.

## Run it

Download the data from [Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) and put sales_train_evaluation.csv, calendar.csv and sell_prices.csv in data/raw/.

~~~bash
pip install -r requirements.txt
python -m forecastbench.benchmark
~~~

The leaderboard ends up in artifacts/leaderboard.csv. Each forecast date can also run on its own with --task 0 to 3, then --aggregate.

## API

The API serves precomputed forecasts, so nothing is trained or run per request.

~~~bash
uvicorn app.main:app --port 8080
gcloud run deploy forecastbench --source . --region northamerica-northeast1 --allow-unauthenticated
~~~

- / : results and a forecast explorer
- /api/forecast/{id} : 28-day forecast with its interval, the model that served it, why it was routed there, and the main drivers when LightGBM served it
- /api/series, /api/benchmark, /health
