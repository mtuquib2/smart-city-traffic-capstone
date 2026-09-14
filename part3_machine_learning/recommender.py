"""
recommender.py - Task 5: Travel-time recommendation system for the I-94 corridor.

The dataset covers a single corridor, so the system recommends *when* to travel rather
than *which route*. For a requested date, acceptable departure window, trip window length
and forecast weather it:

1. Builds one feature row per candidate hour with the shared feature builder (day type,
   holiday, cyclical time, weather) and predicts traffic volume with the champion regressor.
2. Maps predictions to congestion categories (the same quartile thresholds as the labels)
   and scores the proxy-risk classifier for a weather/congestion caution flag.
3. Ranks every contiguous travel window by mean predicted volume, and compares the best
   window with the busiest one and with the historical median for that day type and hour.
4. Produces a plain-language recommendation plus a structured result for the API.

Serving reference
-----------------
`models/serving_reference.json` (built from the training data by `build_serving_reference`)
holds congestion thresholds, typical temperatures by month/hour, known holiday dates and
historical medians, so the recommender and API run without the full dataset.

CLI example
-----------
    python recommender.py --date 2018-03-13 --earliest 7 --latest 19 --window 2 --weather Snow
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config import CONGESTION_ORDER, MODELS_DIR, REPORTS_DIR, WEATHER_CATEGORIES
from features import make_feature_frame

logger = logging.getLogger(__name__)

SERVING_REFERENCE = MODELS_DIR / "serving_reference.json"
DEFAULT_CLOUDS = {"Clear": 1, "Clouds": 75}
DEFAULT_RAIN_MM = {"Rain": 1.0, "Drizzle": 0.3, "Thunderstorm": 2.0, "Squall": 1.0}
WEATHER_ADVICE = {
    "Snow": "Snow lowers daytime volumes by about 8% historically but makes each journey slower and riskier; allow extra time.",
    "Thunderstorm": "Thunderstorms barely change volumes but reduce safety; avoid travelling during the storm if you can.",
    "Squall": "Squalls are rare and hazardous; delay travel if possible.",
    "Fog": "Fog reduces visibility; drive with extra following distance.",
    "Mist": "Mist can reduce visibility, especially in the early morning.",
    "Haze": "Haze can reduce visibility.",
    "Smoke": "Smoke can reduce visibility and air quality.",
    "Rain": "Rain has little effect on volumes (about -1%) but wet roads lengthen stopping distances.",
    "Drizzle": "Drizzle has little effect on volumes.",
}


# ---------------------------------------------------------------------------
# Serving reference
# ---------------------------------------------------------------------------
def day_type_of(day: date, is_holiday: bool) -> str:
    if is_holiday:
        return "holiday"
    return "weekend" if day.weekday() >= 5 else "weekday"


def build_serving_reference(df: pd.DataFrame) -> dict[str, Any]:
    """Summarise training-period data into a small JSON the recommender/API can load at start-up."""
    train = df[df["split"] != "test"]  # everything the deployed models were trained on (<= 2017)
    day_type = np.where(train["is_holiday"] == 1, "holiday", np.where(train["is_weekend"] == 1, "weekend", "weekday"))
    medians = (train.assign(day_type=day_type).groupby(["day_type", "hour"])["traffic_volume"].median().round())
    q1, q2, q3 = df["traffic_volume"].quantile([0.25, 0.5, 0.75]).values
    temps = train.assign(month=train["date_time"].dt.month).groupby(["month", "hour"])["temp_c"].median().round(1)
    classifier_meta = json.loads((MODELS_DIR / "traffic_risk_classifier.json").read_text(encoding="utf-8"))
    reference = {
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "congestion_thresholds": {"q1": float(q1), "q2": float(q2), "q3": float(q3)},
        "risk_decision_threshold": float(classifier_meta["decision_threshold"]),
        "holiday_dates": sorted(str(d)[:10] for d in df.loc[df["is_holiday"] == 1, "date"].unique()),
        "median_volume": {f"{dt}|{int(h)}": float(v) for (dt, h), v in medians.items()},
        "typical_temp_c": {f"{int(m)}|{int(h)}": float(v) for (m, h), v in temps.items()},
        "data_coverage": {"start": str(df["date_time"].min()), "end": str(df["date_time"].max())},
    }
    SERVING_REFERENCE.write_text(json.dumps(reference, indent=2), encoding="utf-8")
    logger.info("Saved serving reference (%d holiday dates, %d median cells) to models/%s",
                len(reference["holiday_dates"]), len(reference["median_volume"]), SERVING_REFERENCE.name)
    return reference


# ---------------------------------------------------------------------------
# Recommendation engine
# ---------------------------------------------------------------------------
@dataclass
class TripRequest:
    travel_date: date
    earliest_hour: int = 6
    latest_hour: int = 21
    window_hours: int = 1
    weather: str = "Clear"
    temp_c: float | None = None
    is_holiday: bool | None = None

    def validate(self) -> None:
        if not 0 <= self.earliest_hour <= 23 or not 0 <= self.latest_hour <= 23:
            raise ValueError("earliest_hour and latest_hour must be between 0 and 23")
        if self.earliest_hour > self.latest_hour:
            raise ValueError("earliest_hour must not be later than latest_hour")
        if not 1 <= self.window_hours <= self.latest_hour - self.earliest_hour + 1:
            raise ValueError("window_hours must fit inside the requested time range")
        if self.weather.title() not in WEATHER_CATEGORIES:
            raise ValueError(f"weather must be one of {WEATHER_CATEGORIES}")


@dataclass
class Recommendation:
    request: dict[str, Any]
    day_type: str
    best_window: dict[str, Any]
    alternatives: list[dict[str, Any]]
    busiest_window: dict[str, Any]
    message: str
    hourly: list[dict[str, Any]] = field(default_factory=list)


class TravelRecommender:
    def __init__(self) -> None:
        self.regressor = joblib.load(MODELS_DIR / "traffic_volume_regressor.joblib")
        self.classifier = joblib.load(MODELS_DIR / "traffic_risk_classifier.joblib")
        self.reference = json.loads(SERVING_REFERENCE.read_text(encoding="utf-8"))
        self.holidays = set(self.reference["holiday_dates"])
        logger.info("Travel recommender ready (regressor + risk classifier + serving reference)")

    # -- helpers ---------------------------------------------------------------
    def congestion_category(self, volume: float) -> str:
        t = self.reference["congestion_thresholds"]
        return "Low" if volume <= t["q1"] else "Medium" if volume <= t["q2"] else "High" if volume <= t["q3"] else "Severe"

    def hourly_forecast(self, req: TripRequest) -> pd.DataFrame:
        weather = req.weather.title()
        is_holiday = req.is_holiday if req.is_holiday is not None else str(req.travel_date) in self.holidays
        hours = list(range(req.earliest_hour, req.latest_hour + 1))
        temps = [req.temp_c if req.temp_c is not None
                 else self.reference["typical_temp_c"].get(f"{req.travel_date.month}|{h}", 8.0) for h in hours]
        inputs = pd.DataFrame({
            "date_time": [pd.Timestamp(req.travel_date) + pd.Timedelta(hours=h) for h in hours],
            "temp_c": temps,
            "rain_1h": DEFAULT_RAIN_MM.get(weather, 0.0),
            "snow_1h": 0.0,
            "clouds_all": DEFAULT_CLOUDS.get(weather, 90),
            "weather_main": weather,
            "weather_description": weather.lower(),
            "is_holiday": int(is_holiday),
        })
        X = make_feature_frame(inputs).astype(float)
        day_type = day_type_of(req.travel_date, is_holiday)
        out = pd.DataFrame({
            "hour": hours,
            "predicted_volume": np.clip(self.regressor.predict(X), 0, None).round(),
            "risk_probability": self.classifier.predict_proba(X)[:, 1].round(3),
            "historical_median": [self.reference["median_volume"].get(f"{day_type}|{h}", np.nan) for h in hours],
            "temp_c": temps,
        })
        out["congestion"] = out["predicted_volume"].apply(self.congestion_category)
        out["elevated_risk"] = out["risk_probability"] >= self.reference["risk_decision_threshold"]
        return out

    @staticmethod
    def _windows(hourly: pd.DataFrame, length: int) -> pd.DataFrame:
        rows = []
        for start in range(len(hourly) - length + 1):
            chunk = hourly.iloc[start:start + length]
            rows.append({
                "start_hour": int(chunk["hour"].iloc[0]),
                "end_hour": int(chunk["hour"].iloc[-1]) + 1,
                "mean_volume": float(chunk["predicted_volume"].mean()),
                "worst_congestion": max(chunk["congestion"], key=CONGESTION_ORDER.index),
                "elevated_risk": bool(chunk["elevated_risk"].any()),
                "historical_median": float(chunk["historical_median"].mean()),
            })
        return pd.DataFrame(rows)

    # -- public API ------------------------------------------------------------
    def recommend(self, req: TripRequest) -> Recommendation:
        req.validate()
        logger.info("Recommendation requested: %s", {k: str(v) for k, v in asdict(req).items()})
        hourly = self.hourly_forecast(req)
        windows = self._windows(hourly, req.window_hours)
        # Rank by predicted volume; windows flagged for elevated (proxy) risk are ranked after safe ones.
        ranked = windows.sort_values(["elevated_risk", "mean_volume"]).reset_index(drop=True)
        best, busiest = ranked.iloc[0], windows.sort_values("mean_volume", ascending=False).iloc[0]
        alternatives = ranked.iloc[1:4]
        is_holiday = req.is_holiday if req.is_holiday is not None else str(req.travel_date) in self.holidays
        day_type = day_type_of(req.travel_date, is_holiday)
        message = self._message(req, day_type, best, busiest, alternatives, hourly)
        logger.info("Recommended %02d:00-%02d:00 (~%.0f vehicles/hour, %s)", best["start_hour"], best["end_hour"],
                    best["mean_volume"], best["worst_congestion"])
        if bool(windows["elevated_risk"].all()):
            logger.warning("Every candidate window carries elevated proxy risk for %s in %s", req.travel_date, req.weather)
        return Recommendation(
            request={k: str(v) for k, v in asdict(req).items()}, day_type=day_type,
            best_window=_window_dict(best), alternatives=[_window_dict(r) for _, r in alternatives.iterrows()],
            busiest_window=_window_dict(busiest), message=message, hourly=hourly.to_dict(orient="records"),
        )

    def _message(self, req, day_type, best, busiest, alternatives, hourly) -> str:
        journey = {"weekday": "weekday", "weekend": "weekend", "holiday": "public-holiday"}[day_type]
        span = f"{best['start_hour']:02d}:00 and {best['end_hour']:02d}:00"
        weather_text = "" if req.weather.title() in ("Clear", "Clouds") else f" in {req.weather.lower()}"
        saving = 1 - best["mean_volume"] / busiest["mean_volume"] if busiest["mean_volume"] else 0
        parts = [
            f"For a {journey} journey on {req.travel_date:%A %d %B %Y}{weather_text}, consider travelling between {span}, "
            f"when traffic is forecast at about {best['mean_volume']:,.0f} vehicles per hour "
            f"({best['worst_congestion']} congestion)."
        ]
        if saving > 0.05:
            parts.append(f"That is roughly {saving:.0%} lighter than the busiest option in your range, "
                         f"{busiest['start_hour']:02d}:00-{busiest['end_hour']:02d}:00 (~{busiest['mean_volume']:,.0f}).")
        else:
            parts.append("Traffic is similar across your whole range, so timing will make little difference.")
        if not np.isnan(best["historical_median"]):
            parts.append(f"Historically this {journey} slot averages about {best['historical_median']:,.0f} vehicles per hour.")
        if len(alternatives):
            alt = ", ".join(f"{r['start_hour']:02d}:00-{r['end_hour']:02d}:00 (~{r['mean_volume']:,.0f})"
                            for _, r in alternatives.head(2).iterrows())
            parts.append(f"Good alternatives: {alt}.")
        if best["worst_congestion"] == "Severe":
            parts.append("Every window in your range is expected to be heavily congested; widen your time range if you can.")
        advice = WEATHER_ADVICE.get(req.weather.title())
        if advice:
            parts.append(advice)
        if bool(hourly["elevated_risk"].any()):
            risky = hourly.loc[hourly["elevated_risk"], "hour"]
            parts.append(f"Caution: {len(risky)} hour(s) combine heavy traffic with poor weather "
                         f"(proxy risk flag, e.g. {int(risky.iloc[0]):02d}:00); prefer windows outside them.")
        return " ".join(parts)


def _window_dict(row: pd.Series) -> dict[str, Any]:
    return {"start": f"{int(row['start_hour']):02d}:00", "end": f"{int(row['end_hour']):02d}:00",
            "mean_predicted_volume": round(float(row["mean_volume"])), "congestion": row["worst_congestion"],
            "elevated_risk": bool(row["elevated_risk"]),
            "historical_median": None if np.isnan(row["historical_median"]) else round(float(row["historical_median"]))}


# ---------------------------------------------------------------------------
# Scenario report
# ---------------------------------------------------------------------------
SCENARIOS = [
    ("Weekday commute, clear", TripRequest(date(2018, 3, 13), 6, 20, 1, "Clear")),
    ("Weekday, snow, 2-hour window", TripRequest(date(2018, 1, 16), 7, 19, 2, "Snow")),
    ("Weekday evening options, rain", TripRequest(date(2018, 6, 19), 15, 22, 1, "Rain")),
    ("Weekday morning, mist", TripRequest(date(2018, 4, 10), 6, 11, 1, "Mist")),
    ("Saturday errands, clouds", TripRequest(date(2018, 5, 12), 8, 20, 2, "Clouds")),
    ("Public holiday (Independence Day)", TripRequest(date(2018, 7, 4), 8, 20, 1, "Clear")),
]


def write_scenario_report(recommender: TravelRecommender) -> None:
    lines = ["# Recommendation Examples", "",
             "Generated by `recommender.py` with the champion regressor and proxy-risk classifier.",
             "Forecast weather is supplied by the user; temperatures default to the historical median for the month and hour.", ""]
    for title, req in SCENARIOS:
        rec = recommender.recommend(req)
        lines += [f"## {title}", "",
                  f"**Request:** {req.travel_date:%a %d %b %Y}, {req.earliest_hour:02d}:00-{req.latest_hour + 1:02d}:00, "
                  f"{req.window_hours}-hour window, {req.weather}", "", f"> {rec.message}", "",
                  "| Hour | Predicted volume | Congestion | Historical median | Proxy risk p |", "|---|---|---|---|---|"]
        for h in rec.hourly:
            flag = " ⚠" if h["elevated_risk"] else ""
            median = "" if pd.isna(h["historical_median"]) else f"{h['historical_median']:,.0f}"
            lines.append(f"| {h['hour']:02d}:00 | {h['predicted_volume']:,.0f} | {h['congestion']} | {median} | "
                         f"{h['risk_probability']:.2f}{flag} |")
        lines.append("")
    path = REPORTS_DIR / "recommendation_examples.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %d recommendation scenarios to reports/%s", len(SCENARIOS), path.name)


def run(df: pd.DataFrame | None = None) -> None:
    from data_prep import load_modelling_table

    df = load_modelling_table() if df is None else df
    build_serving_reference(df)
    write_scenario_report(TravelRecommender())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recommend lower-traffic travel windows on the I-94 corridor")
    parser.add_argument("--date", required=True, help="Travel date, YYYY-MM-DD")
    parser.add_argument("--earliest", type=int, default=6, help="Earliest departure hour (0-23)")
    parser.add_argument("--latest", type=int, default=21, help="Latest departure hour (0-23)")
    parser.add_argument("--window", type=int, default=1, help="Length of the travel window in hours")
    parser.add_argument("--weather", default="Clear", help=f"Forecast weather: {', '.join(WEATHER_CATEGORIES)}")
    parser.add_argument("--temp", type=float, default=None, help="Forecast temperature in °C (default: typical)")
    parser.add_argument("--holiday", action="store_true", help="Treat the date as a public holiday")
    args = parser.parse_args(argv)
    try:
        travel_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        request = TripRequest(travel_date, args.earliest, args.latest, args.window, args.weather, args.temp,
                              True if args.holiday else None)
        recommendation = TravelRecommender().recommend(request)
    except ValueError as exc:
        logger.error("Invalid recommendation request: %s", exc)
        return 2
    except FileNotFoundError as exc:
        logger.error("Model or serving reference missing (%s). Run `python run_all.py` first.", exc)
        return 1
    print(recommendation.message)
    print()
    print(pd.DataFrame(recommendation.hourly)[["hour", "predicted_volume", "congestion", "historical_median",
                                               "risk_probability"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    from logging_config import configure_logging

    configure_logging("recommender", mode="a")
    sys.exit(main())
