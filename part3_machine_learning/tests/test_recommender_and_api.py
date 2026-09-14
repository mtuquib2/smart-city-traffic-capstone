"""Recommendation engine and FastAPI deployment mock-up (need trained models)."""

import warnings
from datetime import date

import pytest

from conftest import requires_models

warnings.filterwarnings("ignore", category=DeprecationWarning)


@requires_models
def test_recommendation_is_plain_language_and_within_range():
    from recommendation_system.recommender import TravelRecommender, TripRequest

    rec = TravelRecommender().recommend(TripRequest(date(2018, 3, 13), 7, 19, 2, "Snow"))
    assert rec.message.startswith("For a weekday journey")
    assert "consider travelling between" in rec.message
    assert 7 <= int(rec.best_window["start"][:2]) <= 18
    assert rec.best_window["mean_predicted_volume"] <= rec.busiest_window["mean_predicted_volume"]


@requires_models
def test_holiday_is_detected_from_calendar():
    from recommendation_system.recommender import TravelRecommender, TripRequest

    assert TravelRecommender().recommend(TripRequest(date(2017, 7, 4), 8, 20, 1, "Clear")).day_type == "holiday"


@requires_models
def test_invalid_request_is_rejected():
    from recommendation_system.recommender import TripRequest

    with pytest.raises(ValueError):
        TripRequest(date(2018, 3, 13), 20, 7, 1, "Clear").validate()


@requires_models
def test_api_endpoints():
    from fastapi.testclient import TestClient

    from deployment.app import app

    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        body = {"date_time": "2018-03-13T08:00:00", "temp_c": 1.0, "weather_main": "Clear"}
        volume = client.post("/predict/traffic-volume", json=body)
        assert volume.status_code == 200 and volume.json()["predicted_volume"] > 3000
        risk = client.post("/predict/risk", json={**body, "weather_main": "Mist"})
        assert risk.status_code == 200 and "PROXY" in risk.json()["disclaimer"]
        bad = client.post("/predict/traffic-volume", json={**body, "weather_main": "Tornado"})
        assert bad.status_code == 422
        rec = client.post("/recommend", json={"travel_date": "2018-03-13", "earliest_hour": 6, "latest_hour": 20})
        assert rec.status_code == 200 and "consider travelling" in rec.json()["message"]
