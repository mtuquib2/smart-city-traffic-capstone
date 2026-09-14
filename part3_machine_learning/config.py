"""
config.py - Central configuration for Part 3 (paths, splits, feature lists, MLflow).

Every tunable constant lives here so experiments are reproducible and documented.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
PART2_ROOT = REPO_ROOT / "part2_python"

# Part 2 modules (pipeline, feature_engineering) are reused rather than duplicated.
if str(PART2_ROOT) not in sys.path:
    sys.path.insert(0, str(PART2_ROOT))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PART2_RAW_DATA = PART2_ROOT / "data" / "raw" / "Metro_Interstate_Traffic_Volume.csv"
PART2_CLEAN_DATA = PART2_ROOT / "data" / "processed" / "traffic_clean.csv"

DATA_DIR = PROJECT_ROOT / "data"
MODELLING_TABLE = DATA_DIR / "modelling_table.csv"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
MONITORING_DIR = PROJECT_ROOT / "monitoring"
LOGS_DIR = PROJECT_ROOT / "logs"
MODEL_REGISTRY_FILE = MODELS_DIR / "model_registry.json"

# ---------------------------------------------------------------------------
# MLflow (SQLite backend enables the model registry; artifacts stored locally)
# ---------------------------------------------------------------------------
MLFLOW_DIR = PROJECT_ROOT / "mlflow"          # no __init__.py: must never shadow the mlflow library
MLFLOW_DB = MLFLOW_DIR / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"
MLFLOW_ARTIFACT_ROOT = MLFLOW_DIR / "mlartifacts"
EXPERIMENT_CLASSIFICATION = "traffic-risk-classification"
EXPERIMENT_REGRESSION = "traffic-volume-regression"
EXPERIMENT_UNSUPERVISED = "traffic-unsupervised"
EXPERIMENT_DEEP_LEARNING = "traffic-volume-lstm"
EXPERIMENT_MONITORING = "traffic-monitoring"
REGISTERED_CLASSIFIER = "traffic-risk-classifier"
REGISTERED_REGRESSOR = "traffic-volume-regressor"
REGISTERED_LSTM = "traffic-volume-lstm"
CHAMPION_ALIAS = "champion"

# ---------------------------------------------------------------------------
# Reproducibility and time-based splits (no shuffling: future never leaks into the past)
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TRAIN_END = pd.Timestamp("2016-12-31 23:59:59")        # train: Oct 2012 - Dec 2016
VALIDATION_END = pd.Timestamp("2017-12-31 23:59:59")   # validation: 2017 (model selection)
# test: Jan - Sep 2018 (final, untouched evaluation and monitoring "production" period)

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
CONGESTION_ORDER = ["Low", "Medium", "High", "Severe"]
HIGH_CONGESTION = ["High", "Severe"]

# ---------------------------------------------------------------------------
# Common feature set shared by the classification and regression models
# ---------------------------------------------------------------------------
WEATHER_CATEGORIES = [
    "Clear", "Clouds", "Drizzle", "Fog", "Haze", "Mist", "Rain", "Smoke", "Snow", "Squall", "Thunderstorm",
]
TIME_FEATURES = [
    "hour", "day_of_week", "month", "is_weekend", "is_holiday", "is_rush_hour",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
]
WEATHER_FEATURES = [
    "temp_c", "rain_1h_log", "snow_1h", "clouds_all",
    "is_precipitation", "is_low_visibility", "is_severe_weather", "is_freezing", "weather_severity",
] + [f"weather_{c}" for c in WEATHER_CATEGORIES]
MODEL_FEATURES = TIME_FEATURES + WEATHER_FEATURES

# ---------------------------------------------------------------------------
# Sustainability estimate (Task 7): laptop CPU package power and grid carbon intensity
# ---------------------------------------------------------------------------
CPU_POWER_WATTS = 30.0            # Intel Core Ultra 7 258V under sustained load (approx.)
GRID_KG_CO2_PER_KWH = 0.41        # Singapore grid emission factor (EMA, approx.)
