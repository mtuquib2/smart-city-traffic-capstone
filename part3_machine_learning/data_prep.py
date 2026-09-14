"""
data_prep.py - Build the Part 3 modelling table from the Part 2 cleaned dataset.

Steps
-----
1. Load Part 2's cleaned hourly data (re-running the Part 2 cleaning pipeline if it
   has not been generated yet).
2. Add engineered features with features.make_feature_frame().
3. Add the congestion_category (quartiles of traffic_volume) and the PROXY
   accident-risk label `high_risk`, exactly as specified in the capstone brief.
4. Assign a time-based split: train (<=2016), validation (2017), test (2018).

IMPORTANT: no accident dataset was available. `high_risk` is a documented proxy
(High/Severe congestion during severe or low-visibility weather) used only to
demonstrate a classification workflow. It is NOT a prediction of real accidents.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import (
    CONGESTION_ORDER,
    DATA_DIR,
    HIGH_CONGESTION,
    MODELLING_TABLE,
    PART2_CLEAN_DATA,
    PART2_RAW_DATA,
    TRAIN_END,
    VALIDATION_END,
)
from feature_engineering import NO_HOLIDAY_LABEL, SEVERE_WEATHER
from features import make_feature_frame

logger = logging.getLogger(__name__)


def load_clean_data() -> pd.DataFrame:
    """Load Part 2's cleaned data, regenerating it with the Part 2 pipeline if absent."""
    if not PART2_CLEAN_DATA.exists():
        logger.warning("Part 2 cleaned data not found at %s; running the Part 2 cleaning pipeline", PART2_CLEAN_DATA.name)
        import pipeline as part2_pipeline  # imported lazily: only needed when regenerating

        raw = part2_pipeline.load_raw_data(PART2_RAW_DATA)
        cleaned = part2_pipeline.clean_data(part2_pipeline.validate_schema(raw))
        part2_pipeline.save_dataframe(cleaned, PART2_CLEAN_DATA, "cleaned dataset")

    df = pd.read_csv(PART2_CLEAN_DATA, parse_dates=["date_time"], keep_default_na=False, na_values=[""])
    logger.info("Loaded Part 2 cleaned data: %d rows, %d columns", *df.shape)
    return df


def add_calendar_context(df: pd.DataFrame) -> pd.DataFrame:
    """Temperature in Celsius and a whole-day holiday flag (raw feed marks only 00:00)."""
    df = df.copy()
    df["temp_c"] = (df["temp"] - 273.15).round(2)
    df["date"] = df["date_time"].dt.normalize()
    holiday_dates = df.loc[df["holiday"] != NO_HOLIDAY_LABEL, "date"].unique()
    df["is_holiday"] = df["date"].isin(holiday_dates).astype(int)
    logger.info("Added temp_c and whole-day is_holiday flag (%d holiday dates)", len(holiday_dates))
    return df


def add_congestion_category(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Congestion category from data-driven quartiles of traffic_volume (brief's definition, <= boundaries)."""
    df = df.copy()
    q1, q2, q3 = df["traffic_volume"].quantile([0.25, 0.5, 0.75]).values
    logger.debug("Congestion quartile thresholds: q1=%.1f q2=%.1f q3=%.1f", q1, q2, q3)

    def bucket(v: float) -> str:
        if v <= q1:
            return "Low"
        elif v <= q2:
            return "Medium"
        elif v <= q3:
            return "High"
        return "Severe"

    df["congestion_category"] = df["traffic_volume"].apply(bucket)
    df["congestion_category"] = pd.Categorical(df["congestion_category"], categories=CONGESTION_ORDER, ordered=True)
    counts = df["congestion_category"].value_counts().reindex(CONGESTION_ORDER).to_dict()
    logger.info("Added congestion_category from quartiles (q1=%.0f, q2=%.0f, q3=%.0f): %s", q1, q2, q3, counts)
    return df, {"q1": float(q1), "q2": float(q2), "q3": float(q3)}


def add_proxy_risk_label(df: pd.DataFrame) -> pd.DataFrame:
    """Proxy accident-risk label: High/Severe congestion AND severe or low-visibility weather."""
    df = df.copy()
    high_congestion = df["congestion_category"].isin(HIGH_CONGESTION)
    risky_weather = df["weather_main"].isin(SEVERE_WEATHER) | (df["is_low_visibility"] == 1)
    df["high_risk"] = (high_congestion & risky_weather).astype(int)
    rate = df["high_risk"].mean()
    logger.info("Added PROXY high_risk label: %d positive hours (%.2f%%)", int(df["high_risk"].sum()), 100 * rate)
    if rate < 0.10:
        logger.warning("high_risk label is imbalanced (%.2f%% positive): using class weighting and PR-AUC", 100 * rate)
    return df


def assign_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["split"] = np.select(
        [df["date_time"] <= TRAIN_END, df["date_time"] <= VALIDATION_END], ["train", "validation"], default="test"
    )
    sizes = df["split"].value_counts().reindex(["train", "validation", "test"]).to_dict()
    logger.info("Time-based split sizes: %s", sizes)
    return df


def build_modelling_table(save: bool = True) -> pd.DataFrame:
    """Run all preparation steps and (optionally) cache the result to data/modelling_table.csv."""
    df = add_calendar_context(load_clean_data())
    features = make_feature_frame(df)
    df = pd.concat([df.drop(columns=[c for c in features.columns if c in df.columns]), features], axis=1)
    logger.info("Built %d model features; table shape %d rows x %d columns", features.shape[1], *df.shape)

    df, thresholds = add_congestion_category(df)
    df = add_proxy_risk_label(df)
    df = assign_split(df)
    df.attrs["congestion_thresholds"] = thresholds

    if save:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(MODELLING_TABLE, index=False)
        logger.info("Saved modelling table to %s", MODELLING_TABLE.relative_to(MODELLING_TABLE.parents[1]).as_posix())
    return df


def load_modelling_table() -> pd.DataFrame:
    """Load the cached modelling table, building it first if needed."""
    if not MODELLING_TABLE.exists():
        logger.info("Modelling table not cached yet; building it")
        return build_modelling_table(save=True)
    df = pd.read_csv(MODELLING_TABLE, parse_dates=["date_time", "date"], keep_default_na=False, na_values=[""])
    df["congestion_category"] = pd.Categorical(df["congestion_category"], categories=CONGESTION_ORDER, ordered=True)
    logger.info("Loaded modelling table: %d rows, %d columns", *df.shape)
    return df


if __name__ == "__main__":
    from logging_config import configure_logging

    configure_logging("data_prep")
    build_modelling_table(save=True)
