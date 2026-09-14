"""Drift statistics and PASS/ALERT rules."""

import numpy as np
import pandas as pd

from monitoring.monitoring import evaluate_batch, psi_categorical, psi_numeric


def test_psi_is_near_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(10, 5, 5000))
    assert psi_numeric(reference, pd.Series(rng.normal(10, 5, 5000))) < 0.05
    assert psi_numeric(reference, pd.Series(rng.normal(25, 5, 5000))) > 0.25


def test_psi_categorical_detects_mix_change():
    reference = pd.Series(["Clear"] * 80 + ["Snow"] * 20)
    assert psi_categorical(reference, reference) < 1e-6
    assert psi_categorical(reference, pd.Series(["Clear"] * 20 + ["Snow"] * 80)) > 0.25


def _batch(volume_factor=1.0, temp_shift=0.0, n=720, seed=1):
    rng = np.random.default_rng(seed)
    dt = pd.date_range("2018-07-01", periods=n, freq="h")
    volume = rng.uniform(500, 6000, n)
    return pd.DataFrame({"date_time": dt, "traffic_volume": volume * volume_factor, "predicted_volume": volume + rng.normal(0, 150, n),
                         "temp_c": rng.normal(22, 4, n) + temp_shift, "clouds_all": rng.choice([1, 40, 90], n),
                         "rain_1h": 0.0, "snow_1h": 0.0, "weather_main": rng.choice(["Clear", "Clouds"], n)})


RANGES = {"temp_c": (-30.0, 36.0), "clouds_all": (0.0, 100.0), "rain_1h": (0.0, 60.0), "snow_1h": (0.0, 0.6)}


def test_normal_batch_passes():
    reference = _batch(seed=2).assign(date_time=pd.date_range("2017-07-01", periods=720, freq="h"))
    result = evaluate_batch("normal", _batch(), reference, baseline_mae=150, training_ranges=RANGES)
    assert result["status"] == "PASS"


def test_sensor_undercount_raises_error_drift_alert():
    reference = _batch(seed=2).assign(date_time=pd.date_range("2017-07-01", periods=720, freq="h"))
    result = evaluate_batch("undercount", _batch(volume_factor=0.6), reference, baseline_mae=150, training_ranges=RANGES)
    assert result["status"] == "ALERT" and any("Error drift" in a for a in result["alerts"])


def test_unit_fault_raises_integrity_alert():
    reference = _batch(seed=2).assign(date_time=pd.date_range("2017-07-01", periods=720, freq="h"))
    fahrenheit = _batch()
    fahrenheit["temp_c"] = fahrenheit["temp_c"] * 9 / 5 + 32
    result = evaluate_batch("units", fahrenheit, reference, baseline_mae=150, training_ranges=RANGES)
    assert any("Data integrity" in a for a in result["alerts"])
