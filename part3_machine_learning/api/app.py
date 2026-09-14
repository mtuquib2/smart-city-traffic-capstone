"""
api/app.py - FastAPI service serving the champion models and the recommendation engine.

Endpoints
---------
GET  /health                      liveness + loaded model versions
GET  /model-info                  metadata and test metrics of served models
POST /predict/traffic-volume      hourly traffic volume for a timestamp + weather
POST /predict/risk                PROXY high-risk probability (not a real accident prediction)
POST /recommend                   plain-language travel-window recommendation

Run (from part3_machine_learning/):
    python -m api.serve                     # uvicorn on http://127.0.0.1:8000, docs at /docs
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.encoders import jsonable_encoder  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from config import MODELS_DIR, MONITORING_DIR  # noqa: E402
from features import make_feature_frame  # noqa: E402
from recommender import TravelRecommender, TripRequest  # noqa: E402

logger = logging.getLogger(__name__)

WeatherMain = Literal["Clear", "Clouds", "Drizzle", "Fog", "Haze", "Mist", "Rain", "Smoke", "Snow", "Squall", "Thunderstorm"]
PREDICTION_LOG = MONITORING_DIR / "api_predictions.jsonl"
PROXY_DISCLAIMER = ("high_risk is a documented PROXY (High/Severe congestion during severe or low-visibility weather). "
                    "It is not trained on accident data and must not be used as an accident prediction.")


class TrafficConditions(BaseModel):
    date_time: datetime = Field(..., examples=["2018-03-13T08:00:00"])
    temp_c: float = Field(..., ge=-45, le=45, examples=[2.5])
    weather_main: WeatherMain = Field(..., examples=["Snow"])
    rain_1h: float = Field(0.0, ge=0, le=305)
    snow_1h: float = Field(0.0, ge=0, le=100)
    clouds_all: int = Field(75, ge=0, le=100)
    is_holiday: bool | None = Field(None, description="Defaults to the known US federal holiday calendar in the data")


class VolumePrediction(BaseModel):
    prediction_id: str
    predicted_volume: int
    congestion_category: str
    model: str
    model_version: str


class RiskPrediction(BaseModel):
    prediction_id: str
    probability: float
    high_risk: bool
    decision_threshold: float
    model: str
    model_version: str
    disclaimer: str = PROXY_DISCLAIMER


class RecommendationRequest(BaseModel):
    travel_date: date = Field(..., examples=["2018-03-13"])
    earliest_hour: int = Field(6, ge=0, le=23)
    latest_hour: int = Field(21, ge=0, le=23)
    window_hours: int = Field(1, ge=1, le=12)
    weather: WeatherMain = "Clear"
    temp_c: float | None = Field(None, ge=-45, le=45)
    is_holiday: bool | None = None


class ServingState:
    recommender: TravelRecommender
    regressor_meta: dict[str, Any]
    classifier_meta: dict[str, Any]
    started: float


state = ServingState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.recommender = TravelRecommender()
    state.regressor_meta = json.loads((MODELS_DIR / "traffic_volume_regressor.json").read_text(encoding="utf-8"))
    state.classifier_meta = json.loads((MODELS_DIR / "traffic_risk_classifier.json").read_text(encoding="utf-8"))
    state.started = time.time()
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("API started: regressor %s v%s, classifier %s v%s", state.regressor_meta["candidate"],
                state.regressor_meta["version"], state.classifier_meta["candidate"], state.classifier_meta["version"])
    yield
    logger.info("API shutting down")


app = FastAPI(title="I-94 Traffic Intelligence API", version="1.0.0", lifespan=lifespan,
              description="Deployment mock-up for the capstone traffic models. Risk outputs use a PROXY label.")


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    logger.warning("Rejected invalid request to %s: %s", request.url.path,
                   "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()))
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})


def _features(conditions: TrafficConditions) -> pd.DataFrame:
    is_holiday = conditions.is_holiday
    if is_holiday is None:
        is_holiday = conditions.date_time.date().isoformat() in state.recommender.holidays
    frame = pd.DataFrame([{**conditions.model_dump(), "is_holiday": int(is_holiday),
                           "weather_description": conditions.weather_main.lower()}])
    return make_feature_frame(frame).astype(float)


def _audit(endpoint: str, payload: dict[str, Any], output: dict[str, Any]) -> None:
    record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "endpoint": endpoint,
              "input": payload, "output": output}
    with PREDICTION_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "uptime_seconds": round(time.time() - state.started, 1),
            "models": {"traffic-volume-regressor": state.regressor_meta["version"],
                       "traffic-risk-classifier": state.classifier_meta["version"]}}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    keep = ("version", "candidate", "algorithm", "trained_on", "validation", "test", "mlflow_run_id")
    return {"traffic-volume-regressor": {k: state.regressor_meta[k] for k in keep},
            "traffic-risk-classifier": {**{k: state.classifier_meta[k] for k in keep},
                                        "decision_threshold": state.classifier_meta["decision_threshold"],
                                        "label": PROXY_DISCLAIMER}}


@app.post("/predict/traffic-volume", response_model=VolumePrediction)
def predict_volume(conditions: TrafficConditions) -> VolumePrediction:
    volume = max(0, int(round(float(state.recommender.regressor.predict(_features(conditions))[0]))))
    result = VolumePrediction(prediction_id=str(uuid.uuid4()), predicted_volume=volume,
                              congestion_category=state.recommender.congestion_category(volume),
                              model=state.regressor_meta["candidate"], model_version=str(state.regressor_meta["version"]))
    logger.info("POST /predict/traffic-volume %s %s -> %d (%s)", conditions.date_time, conditions.weather_main, volume,
                result.congestion_category)
    _audit("/predict/traffic-volume", conditions.model_dump(), result.model_dump())
    return result


@app.post("/predict/risk", response_model=RiskPrediction)
def predict_risk(conditions: TrafficConditions) -> RiskPrediction:
    probability = float(state.recommender.classifier.predict_proba(_features(conditions))[0, 1])
    threshold = float(state.classifier_meta["decision_threshold"])
    result = RiskPrediction(prediction_id=str(uuid.uuid4()), probability=round(probability, 4),
                            high_risk=probability >= threshold, decision_threshold=round(threshold, 4),
                            model=state.classifier_meta["candidate"], model_version=str(state.classifier_meta["version"]))
    logger.info("POST /predict/risk %s %s -> p=%.3f high_risk=%s", conditions.date_time, conditions.weather_main,
                probability, result.high_risk)
    _audit("/predict/risk", conditions.model_dump(), result.model_dump())
    return result


@app.post("/recommend")
def recommend(body: RecommendationRequest) -> dict[str, Any]:
    try:
        recommendation = state.recommender.recommend(TripRequest(**body.model_dump()))
    except ValueError as exc:
        logger.warning("Rejected recommendation request: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    output = {"day_type": recommendation.day_type, "message": recommendation.message,
              "best_window": recommendation.best_window, "alternatives": recommendation.alternatives,
              "busiest_window": recommendation.busiest_window, "hourly_forecast": recommendation.hourly}
    _audit("/recommend", body.model_dump(), {"best_window": recommendation.best_window})
    return output
