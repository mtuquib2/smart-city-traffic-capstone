"""
deployment/serve.py - Entry point that configures logging and launches the API with uvicorn.

    python -m deployment.serve [--host 127.0.0.1] [--port 8000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn  # noqa: E402

from logging_config import configure_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the traffic models with FastAPI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    configure_logging("api", mode="a")
    # log_config=None keeps uvicorn on our handlers/format instead of its own.
    uvicorn.run("deployment.app:app", host=args.host, port=args.port, log_config=None)


if __name__ == "__main__":
    main()
