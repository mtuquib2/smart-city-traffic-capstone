"""Unit tests for the cleaning stages in pipeline.py (run with `pytest`)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline  # noqa: E402


def make_raw(**overrides) -> pd.DataFrame:
    rows = {
        "holiday": ["None", "None", "None"],
        "temp": [280.0, 0.0, 281.0],
        "rain_1h": [0.0, 9831.3, 0.0],
        "snow_1h": [0.0, 0.0, 0.0],
        "clouds_all": [40, 75, 90],
        "weather_main": ["Clouds", "Clouds", "SQUALLS"],
        "weather_description": ["scattered clouds", "Sky is Clear", "SQUALLS"],
        "date_time": ["2016-01-04 08:00:00", "2016-01-04 09:00:00", "2016-01-04 10:00:00"],
        "traffic_volume": [5500, 4800, 4400],
    }
    rows.update(overrides)
    return pd.DataFrame(rows)


def test_load_missing_file_raises_pipeline_error(tmp_path):
    with pytest.raises(pipeline.PipelineError):
        pipeline.load_raw_data(tmp_path / "does_not_exist.csv")


def test_load_empty_file_raises_pipeline_error(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(pipeline.PipelineError):
        pipeline.load_raw_data(empty)


def test_schema_validation_detects_missing_column():
    with pytest.raises(pipeline.SchemaValidationError):
        pipeline.validate_schema(make_raw().drop(columns=["traffic_volume"]))


def test_standardise_categoricals_unifies_casing():
    out = pipeline.standardise_categoricals(make_raw())
    assert out.loc[2, "weather_main"] == "Squall"
    assert out.loc[1, "weather_description"] == "sky is clear"


def test_parse_datetimes_drops_malformed_rows():
    raw = make_raw(date_time=["2016-01-04 08:00:00", "not a date", "2016-13-45 99:00:00"])
    out = pipeline.parse_datetimes(raw)
    assert len(out) == 1


def test_remove_duplicates_collapses_repeated_hours():
    raw = make_raw(date_time=["2016-01-04 08:00:00"] * 3, traffic_volume=[5500] * 3)
    out = pipeline.remove_duplicates(pipeline.parse_datetimes(raw))
    assert len(out) == 1
    assert out.loc[0, "n_weather_conditions"] == 3


def test_impossible_values_are_imputed_with_monthly_median():
    df = pipeline.parse_datetimes(pipeline.standardise_categoricals(make_raw()))
    out = pipeline.handle_impossible_values(df)
    assert out.loc[1, "temp"] == pytest.approx(280.5)  # median of the valid January readings
    assert out["rain_1h"].max() <= pipeline.RAIN_MAX_MM_PER_HOUR


def test_impossible_values_log_warning(caplog):
    df = pipeline.parse_datetimes(pipeline.standardise_categoricals(make_raw()))
    with caplog.at_level("WARNING"):
        pipeline.handle_impossible_values(df)
    assert any("Imputed 1 row(s) in 'temp'" in r.getMessage() for r in caplog.records)
