"""
feature_engineering.py - Create ML-ready features from the cleaned traffic data.

Feature groups
--------------
Time        hour, day_of_week, day_name, month, year, is_weekend, is_holiday,
            is_rush_hour, and cyclical sin/cos encodings of hour, day of week
            and month (so 23:00 sits next to 00:00, Sunday next to Monday).
Weather     temp_c, one-hot weather_main (weather_*), is_precipitation,
            is_low_visibility, is_severe_weather, is_freezing, weather_severity.
Numerical   temp_c_scaled (z-score), clouds_all_scaled (min-max),
            rain_1h_scaled (log1p then min-max), snow_1h_scaled (min-max).
Target      congestion_level / congestion_code - see add_congestion_target().

This module never configures logging handlers; it only obtains a module logger.
Intermediate values (scaling parameters, quartile thresholds) are logged at
DEBUG so they appear only when the pipeline runs with --log-level DEBUG.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NO_HOLIDAY_LABEL = "None"
KELVIN_OFFSET = 273.15

# Weekday commuter peaks identified in Part 1 (Power BI): 6-9 AM and 3-6 PM.
MORNING_RUSH_HOURS = range(6, 10)
EVENING_RUSH_HOURS = range(15, 19)

PRECIPITATION_WEATHER = {"Rain", "Drizzle", "Snow", "Thunderstorm", "Squall"}
LOW_VISIBILITY_WEATHER = {"Mist", "Fog", "Haze", "Smoke"}
SEVERE_WEATHER = {"Thunderstorm", "Squall"}
SEVERE_DESCRIPTION_KEYWORDS = ("heavy", "very heavy", "freezing", "sleet")

CONGESTION_LABELS = ["Low", "Moderate", "High", "Very High"]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar, holiday, rush-hour and cyclical time features."""
    df = df.copy()
    dt = df["date_time"]

    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek  # 0 = Monday ... 6 = Sunday
    df["day_name"] = dt.dt.day_name()
    df["month"] = dt.dt.month
    df["year"] = dt.dt.year
    df["date"] = dt.dt.normalize()
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # The raw feed labels only the 00:00 record of a holiday, so propagate the
    # flag to every hour on that calendar date.
    holiday_dates = df.loc[df["holiday"] != NO_HOLIDAY_LABEL, "date"].unique()
    df["is_holiday"] = df["date"].isin(holiday_dates).astype(int)
    logger.debug("Holiday dates found: %d (propagated to all hours of each date)", len(holiday_dates))

    is_workday = (df["is_weekend"] == 0) & (df["is_holiday"] == 0)
    in_rush_window = df["hour"].isin(MORNING_RUSH_HOURS) | df["hour"].isin(EVENING_RUSH_HOURS)
    df["is_rush_hour"] = (is_workday & in_rush_window).astype(int)

    # Cyclical encodings: map a periodic value onto the unit circle.
    for column, period in (("hour", 24), ("day_of_week", 7), ("month", 12)):
        angle = 2 * np.pi * df[column] / period
        df[f"{column}_sin"] = np.sin(angle)
        df[f"{column}_cos"] = np.cos(angle)
        logger.debug("Cyclical encoding for '%s' uses period=%d", column, period)

    logger.info("Added time features: hour, day_of_week, is_weekend, is_holiday, is_rush_hour, cyclical sin/cos")
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Temperature conversion, one-hot weather categories and derived weather indicators."""
    df = df.copy()

    df["temp_c"] = (df["temp"] - KELVIN_OFFSET).round(2)

    weather_dummies = pd.get_dummies(df["weather_main"], prefix="weather", dtype=int)
    logger.debug("One-hot encoded weather_main into %d columns: %s", weather_dummies.shape[1], list(weather_dummies.columns))
    df = pd.concat([df, weather_dummies], axis=1)

    description = df["weather_description"].astype(str)
    df["is_precipitation"] = (
        df["weather_main"].isin(PRECIPITATION_WEATHER) | (df["rain_1h"] > 0) | (df["snow_1h"] > 0)
    ).astype(int)
    df["is_low_visibility"] = df["weather_main"].isin(LOW_VISIBILITY_WEATHER).astype(int)
    df["is_severe_weather"] = (
        df["weather_main"].isin(SEVERE_WEATHER)
        | description.str.contains("|".join(SEVERE_DESCRIPTION_KEYWORDS), regex=True)
    ).astype(int)
    df["is_freezing"] = (df["temp_c"] <= 0).astype(int)

    # Ordinal driving-conditions score: 0 good, 1 low visibility or light
    # precipitation, 2 precipitation below freezing (snow/ice risk), 3 severe.
    df["weather_severity"] = np.select(
        [
            df["is_severe_weather"] == 1,
            (df["is_precipitation"] == 1) & (df["is_freezing"] == 1),
            (df["is_precipitation"] == 1) | (df["is_low_visibility"] == 1),
        ],
        [3, 2, 1],
        default=0,
    )
    logger.debug("weather_severity distribution: %s", df["weather_severity"].value_counts().sort_index().to_dict())
    logger.info(
        "Added weather features: temp_c, %d weather_* dummies, is_precipitation, is_low_visibility, "
        "is_severe_weather, is_freezing, weather_severity",
        weather_dummies.shape[1],
    )
    return df


def _min_max(series: pd.Series, name: str) -> pd.Series:
    lo, hi = float(series.min()), float(series.max())
    logger.debug("Min-max scaling '%s': min=%.4f max=%.4f", name, lo, hi)
    if hi == lo:
        return pd.Series(0.0, index=series.index)
    return (series - lo) / (hi - lo)


def add_scaled_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalised / standardised versions of continuous variables."""
    df = df.copy()

    mean, std = float(df["temp_c"].mean()), float(df["temp_c"].std())
    logger.debug("Z-score scaling 'temp_c': mean=%.4f std=%.4f", mean, std)
    df["temp_c_scaled"] = (df["temp_c"] - mean) / std

    df["clouds_all_scaled"] = _min_max(df["clouds_all"], "clouds_all")
    # Rainfall is extremely right-skewed (mostly 0), so compress with log1p first.
    df["rain_1h_scaled"] = _min_max(np.log1p(df["rain_1h"]), "log1p(rain_1h)")
    df["snow_1h_scaled"] = _min_max(df["snow_1h"], "snow_1h")

    logger.info("Added scaled features: temp_c_scaled (z-score), clouds_all_scaled, rain_1h_scaled, snow_1h_scaled (min-max)")
    return df


def add_congestion_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create a data-driven congestion category from traffic_volume quartiles.

    Logic
    -----
    Thresholds are the 25th, 50th and 75th percentiles of the cleaned hourly
    traffic_volume, so each class holds roughly a quarter of all hours:

        Low        volume <  Q1
        Moderate   Q1 <= volume < Q2 (median)
        High       Q2 <= volume < Q3
        Very High  volume >= Q3

    Quartiles are used (rather than fixed cut-offs) so the classes adapt to the
    data, are balanced for classification, and extend Part 1's Low/Medium/High
    banding with a fourth level that isolates peak-hour congestion.
    NOTE: for a train/test ML split the thresholds should be fitted on the
    training data only to avoid leakage.
    """
    df = df.copy()
    q1, q2, q3 = np.percentile(df["traffic_volume"], [25, 50, 75])
    logger.debug("Congestion quartile thresholds: Q1=%.1f, Q2(median)=%.1f, Q3=%.1f", q1, q2, q3)

    volume = df["traffic_volume"]
    codes = np.select([volume < q1, volume < q2, volume < q3], [0, 1, 2], default=3)
    df["congestion_code"] = codes
    df["congestion_level"] = pd.Categorical.from_codes(codes, categories=CONGESTION_LABELS, ordered=True)

    logger.debug("Congestion class counts: %s", df["congestion_level"].value_counts().reindex(CONGESTION_LABELS).to_dict())
    logger.info("Added congestion target: congestion_level (%s) based on traffic_volume quartiles", ", ".join(CONGESTION_LABELS))
    return df


def get_congestion_thresholds(traffic_volume: pd.Series) -> tuple[float, float, float]:
    """Return the (Q1, median, Q3) thresholds used by add_congestion_target()."""
    q1, q2, q3 = np.percentile(traffic_volume, [25, 50, 75])
    return float(q1), float(q2), float(q3)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature-engineering steps and log the dataset shape before and after."""
    logger.info("Feature engineering input shape: %d rows x %d columns", df.shape[0], df.shape[1])

    if not pd.api.types.is_datetime64_any_dtype(df["date_time"]):
        df = df.assign(date_time=pd.to_datetime(df["date_time"]))

    df = add_time_features(df)
    df = add_weather_features(df)
    df = add_scaled_features(df)
    df = add_congestion_target(df)

    logger.info("Feature engineering output shape: %d rows x %d columns", df.shape[0], df.shape[1])
    return df
