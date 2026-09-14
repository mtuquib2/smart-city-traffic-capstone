# Smart City Traffic Intelligence Capstone

AI, ML and Data Science capstone on the **Metro Interstate Traffic Volume** dataset: hourly westbound I-94 traffic at
MnDOT station ATR 301 between Minneapolis and St Paul, Oct 2012 – Sep 2018, with weather and holiday attributes.
The project progresses from exploratory analytics (Part 1) to a reproducible Python pipeline (Part 2) and an intelligent
mobility solution with ML, MLOps and responsible AI (Part 3).

> **Accident data:** no real accident dataset was sourced. Part 3's classification uses the **documented proxy label**
> defined in the brief (High/Severe congestion during severe or low-visibility weather). See
> [part3_machine_learning/README.md](part3_machine_learning/README.md).

**Final report:** [part3_machine_learning/reports/FINAL_CAPSTONE_REPORT.md](part3_machine_learning/reports/FINAL_CAPSTONE_REPORT.md) ·
**Bias & fairness report:** [part3_machine_learning/reports/BIAS_FAIRNESS_REPORT.md](part3_machine_learning/reports/BIAS_FAIRNESS_REPORT.md)

## Repository structure

```
smart-city-traffic-capstone/
├── part1_data_analytics/     # Part 1: SQLite database, SQL analyses, Power BI dashboard, summary report
├── part2_python/             # Part 2: logged cleaning pipeline, feature engineering, figures, CLI app, tests
├── part3_machine_learning/   # Part 3: ML/DL models, SHAP, MLflow, recommender, FastAPI, monitoring, reports
└── README.md
```

| Part | Highlights | Start here |
|---|---|---|
| 1 Data analytics | SQL trend, correlation and probability analyses; Power BI traffic dashboard | [part1_data_analytics/README.md](part1_data_analytics/README.md) |
| 2 Python pipeline | Schema validation, 6 logged cleaning steps, 38 features, 7 figures, query CLI, 22 tests | [part2_python/README.md](part2_python/README.md) |
| 3 Machine learning | Classifier + regressor, K-means + Apriori, LSTM + SHAP, MLflow registry, travel-time recommender, FastAPI, PASS/ALERT monitoring, bias/fairness and sustainability | [part3_machine_learning/README.md](part3_machine_learning/README.md) |

## Tools and technologies

SQLite / DB Browser · Power BI · Python 3.13 · pandas · NumPy · Matplotlib · scikit-learn · PyTorch · SHAP · mlxtend ·
MLflow · FastAPI / uvicorn / pydantic · pytest · Git

## Quick start

```bash
cd part2_python
```

```bash
pip install -r requirements.txt
```

```bash
python pipeline.py
```

```bash
cd ../part3_machine_learning
```

```bash
pip install -r requirements.txt
```

```bash
python run_all.py --fresh-mlflow
```

Each part's README documents its commands, logging configuration (where logs are written and what each level means),
outputs, assumptions and limitations.

## Key findings across the capstone

1. **Time drives traffic.** Commuter peaks at 07:00 and 16:00 on workdays, a weekend midday plateau, and quiet nights.
   Hour and day type dominate every model (SHAP) and every high-lift association rule.
2. **Weather matters less than expected, except snow.** Only snow clearly reduces daytime traffic (−8%), and it is also
   where models err most (MAE 2.7× average).
3. **Simple and well-engineered beats complex.** A gradient-boosting model with lag features (MAE 139) outperformed an
   LSTM (MAE 156) with ~11× less training time.
4. **Proxy labels need scrutiny.** Near-perfect risk-classifier scores come from the label's circular definition, not
   from predicting danger.
5. **Monitoring must be context-aware.** A naive drift check alerted every month because of a weather-feed change. The
   tiered, seasonal design flagged only the genuinely anomalous winter months of 2018 and the injected faults.

## Version history

Commits are incremental and descriptive for each part. Part 2 was merged in with its original task-by-task history
preserved (`git log --graph` shows the subtree merge).
