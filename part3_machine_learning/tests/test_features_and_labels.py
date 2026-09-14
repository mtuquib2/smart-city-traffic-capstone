"""Feature builder, congestion category and proxy label behave as specified."""

import numpy as np
import pandas as pd
import pytest

from config import MODEL_FEATURES
from data_prep import add_congestion_category, add_proxy_risk_label
from features import make_feature_frame


def conditions(**overrides) -> pd.DataFrame:
    row = {"date_time": pd.Timestamp("2018-03-13 08:00"), "temp_c": -2.0, "rain_1h": 0.0, "snow_1h": 0.0,
           "clouds_all": 90, "weather_main": "Mist", "weather_description": "mist", "is_holiday": 0}
    row.update(overrides)
    return pd.DataFrame([row])


def test_feature_frame_has_exact_columns_in_order():
    assert list(make_feature_frame(conditions()).columns) == MODEL_FEATURES


def test_cyclical_encodings_and_flags():
    out = make_feature_frame(conditions()).iloc[0]
    assert out["hour_sin"] ** 2 + out["hour_cos"] ** 2 == pytest.approx(1.0)
    assert out["is_rush_hour"] == 1 and out["is_weekend"] == 0
    assert out["is_low_visibility"] == 1 and out["weather_Mist"] == 1 and out["is_freezing"] == 1


def test_holiday_suppresses_rush_hour():
    assert make_feature_frame(conditions(is_holiday=1)).iloc[0]["is_rush_hour"] == 0


def test_unknown_weather_is_all_zero_dummies(caplog):
    out = make_feature_frame(conditions(weather_main="Tornado")).iloc[0]
    assert out[[c for c in MODEL_FEATURES if c.startswith("weather_") and c != "weather_severity"]].sum() == 0
    assert "Unknown weather_main" in caplog.text


def test_congestion_category_uses_inclusive_quartile_boundaries():
    df = pd.DataFrame({"traffic_volume": [100, 200, 300, 400, 500]})
    out, t = add_congestion_category(df)
    assert out.loc[out["traffic_volume"] == t["q1"], "congestion_category"].iloc[0] == "Low"   # v <= q1 -> Low
    assert out["congestion_category"].iloc[-1] == "Severe"


def test_proxy_label_requires_both_congestion_and_risky_weather():
    df = pd.DataFrame({
        "congestion_category": ["Severe", "Severe", "Low", "High"],
        "weather_main": ["Mist", "Clear", "Fog", "Thunderstorm"],
        "is_low_visibility": [1, 0, 1, 0],
    })
    assert add_proxy_risk_label(df)["high_risk"].tolist() == [1, 0, 0, 1]
