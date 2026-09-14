"""Tests for the traffic_app.py command-line application (run with `pytest`)."""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import feature_engineering as fe  # noqa: E402
import traffic_app  # noqa: E402


@pytest.fixture
def processed_csv(tmp_path) -> Path:
    hours = pd.date_range("2016-05-02 00:00", periods=24 * 14, freq="h")  # two weeks from a Monday
    rng = np.random.default_rng(1)
    daytime = hours.hour.isin(range(6, 19))
    raw = pd.DataFrame(
        {
            "holiday": "None",
            "temp": rng.uniform(280, 295, len(hours)),
            "rain_1h": 0.0,
            "snow_1h": 0.0,
            "clouds_all": rng.integers(0, 101, len(hours)),
            "weather_main": np.where(rng.random(len(hours)) < 0.2, "Snow", "Clear"),
            "weather_description": "sky is clear",
            "date_time": hours,
            "traffic_volume": np.where(daytime, 5000, 800) + rng.integers(-200, 200, len(hours)),
        }
    )
    path = tmp_path / "features.csv"
    fe.build_features(raw).to_csv(path, index=False)
    return path


def run(capsys, processed_csv, tmp_path, *args):
    code = traffic_app.main(["--data", str(processed_csv), "--log-file", str(tmp_path / "app.log"), *args])
    logging.getLogger().handlers.clear()
    return code, capsys.readouterr()


@pytest.mark.parametrize(
    "args",
    [
        ["lookup", "--datetime", "2016-05-03 08:00"],
        ["peaks", "--top", "3"],
        ["compare"],
        ["recommend", "--day", "tue"],
        ["weather", "--condition", "snow"],
    ],
)
def test_commands_succeed(capsys, processed_csv, tmp_path, args):
    code, captured = run(capsys, processed_csv, tmp_path, *args)
    assert code == 0
    assert captured.out.strip()
    log_text = (tmp_path / "app.log").read_text(encoding="utf-8")
    assert f"Command invoked: {args[0]}" in log_text


def test_malformed_date_logs_error_without_traceback(capsys, processed_csv, tmp_path):
    code, captured = run(capsys, processed_csv, tmp_path, "lookup", "--datetime", "03/05/2016 8am")
    assert code == 2
    log_text = (tmp_path / "app.log").read_text(encoding="utf-8")
    assert "ERROR" in log_text and "Malformed date/time" in log_text
    assert "Traceback" not in log_text + captured.err


def test_invalid_day_is_rejected(capsys, processed_csv, tmp_path):
    code, _ = run(capsys, processed_csv, tmp_path, "recommend", "--day", "funday")
    assert code == 2


def test_missing_data_file_is_reported(capsys, tmp_path):
    code, _ = run(capsys, tmp_path / "nope.csv", tmp_path, "compare")
    assert code == 1
    assert "Run `python pipeline.py` first" in (tmp_path / "app.log").read_text(encoding="utf-8")
