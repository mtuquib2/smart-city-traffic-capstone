"""
logging_config.py - Logging setup for Part 3 entry points.

Library modules only call ``logging.getLogger(__name__)``. ``configure_logging`` is
called exclusively from ``if __name__ == "__main__":`` blocks (run_all.py, the
individual stage scripts, the API launcher) so handlers are attached once.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from config import LOGS_DIR

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(module)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
NOISY_LIBRARIES = ("matplotlib", "PIL", "fontTools", "mlflow", "alembic", "urllib3", "numba", "shap", "httpx",
                   "git", "sqlalchemy", "h5py", "tensorflow", "torch")


def configure_logging(log_name: str, level: str = "INFO", mode: str = "w") -> Path:
    """Attach console (stdout) and file handlers to the root logger; return the log path."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{log_name}.log"
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, mode=mode, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(numeric_level)
    root.addHandler(console)
    root.addHandler(file_handler)

    for name in NOISY_LIBRARIES:
        logging.getLogger(name).setLevel(logging.WARNING)
    # Font fallback notices (e.g. no 'semibold' face for a glyph) are cosmetic, not pipeline warnings.
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    # MLflow prints artifact progress bars to stderr unless disabled.
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    return log_path
