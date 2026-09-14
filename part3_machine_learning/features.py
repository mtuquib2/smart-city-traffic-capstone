"""
features.py - The single feature builder used for training AND serving.

Using one function for both the training table and API/recommender inputs removes
train/serve skew: a request is converted to exactly the same columns, in the same
order, that the models were fitted on.

Required input columns
----------------------
date_time (datetime), temp_c, rain_1h, snow_1h, clouds_all, weather_main, is_holiday
Optional: weather_description (used for 'heavy'/'freezing' severity keywords)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import MODEL_FEATURES, WEATHER_CATEGORIES
from feature_engineering import (  # Part 2 definitions keep weather logic consistent
    EVENING_RUSH_HOURS,
    LOW_VISIBILITY_WEATHER,
    MORNING_RUSH_HOURS,
    PRECIPITATION_WEATHER,
    SEVERE_DESCRIPTION_KEYWORDS,
    SEVERE_WEATHER,
)

logger = logging.getLogger(__name__)

REQUIRED_INPUT_COLUMNS = ["date_time", "temp_c", "rain_1h", "snow_1h", "clouds_all", "weather_main", "is_holiday"]


def make_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with exactly MODEL_FEATURES (in order) for the given records."""
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required input columns: {missing}")

    dt = pd.to_datetime(df["date_time"])
    out = pd.DataFrame(index=df.index)

    # --- time
    out["hour"] = dt.dt.hour
    out["day_of_week"] = dt.dt.dayofweek
    out["month"] = dt.dt.month
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["is_holiday"] = df["is_holiday"].astype(int)
    workday = (out["is_weekend"] == 0) & (out["is_holiday"] == 0)
    rush_window = out["hour"].isin(MORNING_RUSH_HOURS) | out["hour"].isin(EVENING_RUSH_HOURS)
    out["is_rush_hour"] = (workday & rush_window).astype(int)
    for name, source, period in (("hour", "hour", 24), ("dow", "day_of_week", 7), ("month", "month", 12)):
        angle = 2 * np.pi * out[source] / period
        out[f"{name}_sin"] = np.sin(angle)
        out[f"{name}_cos"] = np.cos(angle)

    # --- weather
    weather = df["weather_main"].astype(str).str.strip().str.title()
    unknown = sorted(set(weather) - set(WEATHER_CATEGORIES))
    if unknown:
        logger.warning("Unknown weather_main value(s) %s in %d row(s): encoded as all-zero weather dummies",
                       unknown, int(weather.isin(unknown).sum()))
    description = df["weather_description"].astype(str).str.lower() if "weather_description" in df else pd.Series("", index=df.index)

    out["temp_c"] = df["temp_c"].astype(float)
    out["rain_1h_log"] = np.log1p(df["rain_1h"].astype(float).clip(lower=0))
    out["snow_1h"] = df["snow_1h"].astype(float)
    out["clouds_all"] = df["clouds_all"].astype(float)
    out["is_precipitation"] = (
        weather.isin(PRECIPITATION_WEATHER) | (df["rain_1h"] > 0) | (df["snow_1h"] > 0)
    ).astype(int)
    out["is_low_visibility"] = weather.isin(LOW_VISIBILITY_WEATHER).astype(int)
    out["is_severe_weather"] = (
        weather.isin(SEVERE_WEATHER) | description.str.contains("|".join(SEVERE_DESCRIPTION_KEYWORDS), regex=True)
    ).astype(int)
    out["is_freezing"] = (out["temp_c"] <= 0).astype(int)
    out["weather_severity"] = np.select(
        [
            out["is_severe_weather"] == 1,
            (out["is_precipitation"] == 1) & (out["is_freezing"] == 1),
            (out["is_precipitation"] == 1) | (out["is_low_visibility"] == 1),
        ],
        [3, 2, 1],
        default=0,
    )
    for category in WEATHER_CATEGORIES:
        out[f"weather_{category}"] = (weather == category).astype(int)

    return out[MODEL_FEATURES]
