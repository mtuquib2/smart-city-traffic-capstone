"""Unit tests for feature_engineering.py (run with `pytest`)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import feature_engineering as fe  # noqa: E402


@pytest.fixture
def clean_df() -> pd.DataFrame:
    hours = pd.date_range("2016-12-24 00:00", periods=48, freq="h")  # Sat 24 & Sun 25 Dec
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "holiday": ["None"] * 24 + ["Christmas Day"] + ["None"] * 23,
            "temp": rng.uniform(260, 290, 48),
            "rain_1h": [0.0] * 47 + [2.5],
            "snow_1h": [0.0] * 48,
            "clouds_all": rng.integers(0, 101, 48),
            "weather_main": ["Clear"] * 40 + ["Snow"] * 4 + ["Thunderstorm"] * 4,
            "weather_description": ["sky is clear"] * 40 + ["heavy snow"] * 4 + ["thunderstorm"] * 4,
            "date_time": hours,
            "traffic_volume": rng.integers(100, 7000, 48),
        }
    )


def test_build_features_preserves_rows_and_adds_columns(clean_df):
    out = fe.build_features(clean_df)
    assert len(out) == len(clean_df)
    for col in ["hour", "day_of_week", "is_weekend", "hour_sin", "hour_cos", "temp_c_scaled", "congestion_level"]:
        assert col in out.columns


def test_holiday_flag_propagates_to_whole_day(clean_df):
    out = fe.build_features(clean_df)
    assert out.loc[out["date"] == pd.Timestamp("2016-12-25"), "is_holiday"].eq(1).all()
    assert out.loc[out["date"] == pd.Timestamp("2016-12-24"), "is_holiday"].eq(0).all()


def test_cyclical_hour_encoding_on_unit_circle(clean_df):
    out = fe.build_features(clean_df)
    assert np.allclose(out["hour_sin"] ** 2 + out["hour_cos"] ** 2, 1.0)


def test_congestion_classes_are_quartile_based(clean_df):
    out = fe.build_features(clean_df)
    q1, q2, q3 = fe.get_congestion_thresholds(clean_df["traffic_volume"])
    assert (out.loc[out["congestion_level"] == "Low", "traffic_volume"] < q1).all()
    assert (out.loc[out["congestion_level"] == "Very High", "traffic_volume"] >= q3).all()


def test_scaled_features_ranges(clean_df):
    out = fe.build_features(clean_df)
    assert out["clouds_all_scaled"].between(0, 1).all()
    assert abs(out["temp_c_scaled"].mean()) < 1e-9


def test_threshold_logged_only_at_debug(clean_df, caplog):
    with caplog.at_level("INFO"):
        fe.build_features(clean_df)
    assert not any("quartile thresholds" in r.getMessage() for r in caplog.records)
    caplog.clear()
    with caplog.at_level("DEBUG"):
        fe.build_features(clean_df)
    assert any("quartile thresholds" in r.getMessage() and r.levelname == "DEBUG" for r in caplog.records)
