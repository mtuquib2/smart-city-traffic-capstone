# Part 3 – Machine Learning and AI: Intelligent Mobility Solution

Supervised, unsupervised and deep learning models, SHAP explainability, MLflow tracking and registry, a travel-time
recommender, a FastAPI deployment, drift monitoring with PASS/ALERT alerting, and a responsible-AI analysis. All are
built on the Part 2 cleaned I-94 traffic data.

> ## ⚠ Accident dataset: documented proxy used
> **No real accident dataset was sourced.** As instructed in the brief, the classification task predicts a **proxy
> label**, `high_risk` = High/Severe congestion (traffic-volume quartiles) **and** severe (Thunderstorm/Squall) or
> low-visibility (Mist/Fog/Haze/Smoke) weather. It exists only to demonstrate the classification workflow, and is **not
> a prediction of real accidents**. Its risks (circularity, blind spots, sensitivity to cleaning choices) are analysed in
> [`reports/BIAS_FAIRNESS_REPORT.md`](reports/BIAS_FAIRNESS_REPORT.md).

**Reports:** [Final capstone report](reports/FINAL_CAPSTONE_REPORT.md) ([PDF](reports/FINAL_CAPSTONE_REPORT.pdf)) · [Responsible AI report (PDF)](responsible_ai_report.pdf) ([Markdown source](reports/BIAS_FAIRNESS_REPORT.md)) · [Model versions](reports/MODEL_VERSIONS.md) · [Recommendation examples](recommendation_system/recommendation_examples.md) · [Walkthrough notebook](notebooks/part3_results_walkthrough.ipynb)

---

## Results at a glance (test set Jan–Sep 2018)

| Task | Result |
|---|---|
| 1 Classification (proxy risk) | HistGB champion: F1 0.954, ROC AUC 1.000, PR AUC 0.994 · logistic baseline F1 0.909 · **no-ML rule F1 0.885** (label is circular) |
| 1 Regression (traffic volume) | HistGB champion: **MAE 217, R² 0.966** · ridge baseline MAE 550, R² 0.871 |
| 2 K-means | k = 4 regimes: night low · adverse-weather (all hours) · afternoon–evening heavy · morning–midday heavy |
| 2 Association rules | Top lift 3.8: "night at weekends, freezing → Low 95%"; "weekday AM peak, cold → Severe 92%" |
| 3 LSTM (next hour) | LSTM MAE 156 (R² 0.986) vs persistence 588 · explainable HistGB + lags surrogate **MAE 139** · SHAP: last hour 49%, time of day 22% |
| 4 MLflow | 5 experiments, 26 runs, 10 registered versions, `champion` aliases |
| 5 Recommender | Plain-language travel windows by day type and weather, e.g. "consider travelling between 18:00 and 20:00 … 35% lighter" |
| 6 MLOps | FastAPI (5 endpoints) · PSI/error/integrity monitoring · dashboard: Jan & Apr 2018 ALERT, current status PASS, 2/2 injected incidents detected |
| 7 Responsible AI | Snow MAE 2.7×, holidays 1.8× · +69% positives if secondary weather kept · all training ≈ 0.9 Wh |

---

## Project structure

```
part3_machine_learning/
├── run_all.py               # ENTRY POINT: reproduces every result below (logs/run_all.log)
├── config.py                # paths, splits, feature lists, MLflow settings, constants
├── logging_config.py        # console + file handlers (called only by entry points)
├── features.py              # single feature builder for training AND serving
├── data_prep.py             # Part 2 data → features, congestion_category, PROXY high_risk, time split
├── supervised_models.py     # Task 1: classifiers + regressors, metrics, MLflow registry, champions
├── unsupervised.py          # Task 2: K-means + Apriori association rules
├── deep_learning.py         # Task 3: PyTorch LSTM, baselines, lag-feature surrogate
├── explainability.py        # Task 3: SHAP (surrogate, regressor, classifier)
├── mlflow_utils.py          # Task 4 / 6.1-6.2: tracking, registry, version + run exports
├── recommendation_system/   # Task 5: recommender.py (engine + CLI), recommendation_examples.md
├── deployment/              # Task 6.3: app.py (FastAPI), serve.py (uvicorn launcher), demo_client.py
├── monitoring/              # Task 6.4-6.5: monitoring.py, monitoring_report.json, dashboard.html
├── fairness_analysis.py     # Task 7: coverage, proxy-label bias, segment errors, footprint
├── plotting.py              # shared chart style (reuses Part 2 palette)
├── models/                  # champion models, metadata cards, model_registry.json, serving_reference.json
├── mlflow/                  # MLflow logs: mlflow.db (tracking + registry) and mlartifacts/
├── responsible_ai_report.pdf  # Task 7: bias, fairness, governance and sustainability
├── figures/                 # task1_… to task7_… PNG figures
├── reports/                 # FINAL_CAPSTONE_REPORT.md, BIAS_FAIRNESS_REPORT.md, MODEL_VERSIONS.md, metrics/*.csv
├── notebooks/               # part3_results_walkthrough.ipynb (executed)
├── logs/                    # run_all.log, api.log, api_demo.log
├── tests/                   # pytest suite (15 tests)
├── data/                    # modelling_table.csv (generated, git-ignored)
└── requirements.txt
```

## How to run

Requires Python 3.10+ (tested with Python 3.13 on Windows 11, CPU only). Run commands from `part3_machine_learning/`.
Part 3 reads Part 2's cleaned data and regenerates it automatically with the Part 2 pipeline if it is missing.

```bash
pip install -r requirements.txt
```

**Reproduce everything** (≈ 3.5 min on a laptop CPU):

```bash
python run_all.py --fresh-mlflow
```

Run selected stages (`data supervised unsupervised deep explain recommend api monitoring fairness exports`):

```bash
python run_all.py --stages monitoring fairness --log-level DEBUG
```

**Recommendation CLI:**

```bash
python -m recommendation_system.recommender --date 2018-01-16 --earliest 7 --latest 19 --window 2 --weather Snow
```

**Serve the API** (Swagger docs at http://127.0.0.1:8000/docs), then call it from a second terminal:

```bash
python -m deployment.serve
```

```bash
python -m deployment.demo_client --url http://127.0.0.1:8000
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict/traffic-volume -H "Content-Type: application/json" -d "{\"date_time\": \"2018-03-13T08:00:00\", \"temp_c\": -2, \"weather_main\": \"Clear\"}"
```

**Monitoring report and dashboard** (prints a status table; open `monitoring/dashboard.html` in a browser):

```bash
python -m monitoring.monitoring
```

**Browse MLflow** (experiments, runs, registered models and aliases) at http://127.0.0.1:5000:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --port 5000
```

**Tests:**

```bash
python -m pytest
```

> MLflow stores absolute artifact paths. On another machine the database, metrics and CSV exports are readable
> immediately; run `python run_all.py --fresh-mlflow` to regenerate artifacts at the new location.

---

## Machine learning models

| Registered model | Champion (alias `champion`) | Selected on | Served by |
|---|---|---|---|
| `traffic-risk-classifier` | v4 HistGradientBoosting (tuned), threshold 0.834 | validation PR AUC | API `/predict/risk`, recommender caution flag |
| `traffic-volume-regressor` | v3 HistGradientBoosting v1 | validation MAE | API `/predict/traffic-volume`, recommender, monitoring |
| `traffic-volume-lstm` | v1 small LSTM (24-hour lookback) | validation MAE | benchmark (surrogate explains it) |

* **Common features (32):** hour, day of week, month, weekend, holiday, rush hour, sin/cos of hour, day of week and
  month, temperature, rain (log), snow, clouds, weather flags, weather severity, 11 weather one-hots.
* **Split:** train ≤ 2016, validation 2017 (selection), test Jan–Sep 2018 (reporting). Final models refit on ≤ 2017.
* **Congestion category:** traffic-volume quartiles (Low ≤ 1,255 < Medium ≤ 3,430 < High ≤ 4,952 < Severe).

## Logging configuration

| Aspect | Implementation |
|---|---|
| Loggers | `logger = logging.getLogger(__name__)` in every module |
| Handlers | Attached only by entry points (`run_all.py`, each module's `__main__`, `deployment/serve.py`) through `logging_config.configure_logging()`: console (stdout) + `logs/<entry-point>.log` |
| Format | `%(asctime)s \| %(levelname)-8s \| %(module)s \| %(message)s` |
| DEBUG | Intermediate values: quartile thresholds, k-selection scores, epoch losses, scaling parameters, integrity ranges |
| INFO | Milestones: data loaded, model trained or registered, figure or report saved, API requests, monitoring PASS |
| WARNING | Recoverable but noteworthy: imbalanced label, interpolated gaps, proxy-label circularity, drift noted, **MONITORING ALERT**, rejected API requests |
| ERROR | A stage or command cannot continue (with `exc_info=True` in `run_all.py`), invalid CLI requests |
| `print()` | Only for end-user output of CLIs (recommender, monitoring status table, API demo summary) |

## Assumptions and limitations

* **Proxy label, not accidents** (see above). Do not use `/predict/risk` for safety decisions.
* One sensor, westbound only; 2014–15 has large gaps; holidays and snow are rare and have the largest errors.
* Recommendations assume the user supplies forecast weather; they optimise timing on this corridor only.
* Energy figures are estimates (30 W CPU, 0.41 kg CO₂/kWh).
* Built with pandas 3.0.5 although MLflow 3.15 declares `pandas<3`; all features used worked, but pin versions for production.
