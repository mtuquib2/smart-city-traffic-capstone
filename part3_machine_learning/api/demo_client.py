"""
api/demo_client.py - Exercise every API endpoint and save the responses as deployment evidence.

By default the app is called in-process with FastAPI's TestClient (no server needed).
Pass --url http://127.0.0.1:8000 to call a running `python -m api.serve` over HTTP instead.

    python -m api.demo_client [--url http://127.0.0.1:8000]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx  # noqa: E402

from config import REPORTS_DIR  # noqa: E402

logger = logging.getLogger(__name__)

CALLS = [
    ("GET", "/health", None),
    ("GET", "/model-info", None),
    ("POST", "/predict/traffic-volume", {"date_time": "2018-03-13T08:00:00", "temp_c": -2.0, "weather_main": "Clear",
                                        "clouds_all": 5}),
    ("POST", "/predict/traffic-volume", {"date_time": "2018-03-17T03:00:00", "temp_c": -5.0, "weather_main": "Snow",
                                        "snow_1h": 0.2, "clouds_all": 90}),
    ("POST", "/predict/risk", {"date_time": "2018-04-10T07:00:00", "temp_c": 4.0, "weather_main": "Mist", "clouds_all": 90}),
    ("POST", "/predict/risk", {"date_time": "2018-04-10T07:00:00", "temp_c": 4.0, "weather_main": "Clear", "clouds_all": 1}),
    ("POST", "/recommend", {"travel_date": "2018-01-16", "earliest_hour": 7, "latest_hour": 19, "window_hours": 2,
                            "weather": "Snow"}),
    ("POST", "/predict/traffic-volume", {"date_time": "not-a-date", "temp_c": 999, "weather_main": "Tornado"}),
]


def run(base_url: str | None = None) -> list[dict]:
    if base_url:
        client_cm = httpx.Client(base_url=base_url, timeout=30)
    else:
        from fastapi.testclient import TestClient

        from api.app import app

        client_cm = TestClient(app)
    results = []
    with client_cm as client:
        for method, path, body in CALLS:
            response = client.request(method, path, json=body)
            logger.info("%s %s -> HTTP %d", method, path, response.status_code)
            payload = response.json()
            if path == "/recommend" and response.status_code == 200:
                payload = {k: v for k, v in payload.items() if k != "hourly_forecast"}  # keep evidence file short
            results.append({"request": {"method": method, "path": path, "body": body},
                            "status_code": response.status_code, "response": payload})
    out = REPORTS_DIR / "api_demo_responses.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Saved %d API request/response pairs to reports/%s", len(results), out.name)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Call every API endpoint and save the responses")
    parser.add_argument("--url", default=None, help="Base URL of a running API (default: in-process TestClient)")
    args = parser.parse_args()
    results = run(args.url)
    for item in results:
        print(f"{item['request']['method']:4} {item['request']['path']:26} HTTP {item['status_code']}")
    return 0


if __name__ == "__main__":
    from logging_config import configure_logging

    configure_logging("api_demo")
    sys.exit(main())
