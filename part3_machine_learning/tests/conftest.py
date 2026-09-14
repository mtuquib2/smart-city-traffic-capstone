import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402  (also puts part2_python on sys.path)

MODELS_READY = all((config.MODELS_DIR / name).exists() for name in (
    "traffic_volume_regressor.joblib", "traffic_risk_classifier.joblib", "serving_reference.json"))

requires_models = pytest.mark.skipif(not MODELS_READY, reason="run `python run_all.py` first to train models")
