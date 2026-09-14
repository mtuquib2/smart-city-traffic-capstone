"""
plotting.py - Shared Matplotlib helpers for Part 3 figures.

Importing Part 2's visualizations module applies the same colour-blind-safe style
(surface, ink, gridline and categorical palette) so the whole portfolio looks consistent.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt

from config import FIGURES_DIR, PROJECT_ROOT
from visualizations import (  # noqa: F401  (re-exported style tokens; import applies rcParams)
    BASELINE,
    BLUE_RAMP,
    GRIDLINE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    ORDINAL_BLUES,
    SERIES_1,
    SERIES_2,
    SURFACE,
    thousands,
)

logger = logging.getLogger(__name__)

# Fixed categorical order (never cycled): blue, orange, aqua, yellow, magenta, green, violet, red
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
DPI = 150


def subtitle(ax: plt.Axes, text: str) -> None:
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK_SECONDARY, fontsize=9.5, va="bottom")


def save_figure(fig: plt.Figure, filename: str, subdir: str | None = None) -> Path:
    """Save a figure under figures/ (optionally a subfolder), close it and log the path."""
    folder = FIGURES_DIR / subdir if subdir else FIGURES_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    try:
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
    finally:
        plt.close(fig)
    logger.info("Saved figure: %s", path.relative_to(PROJECT_ROOT).as_posix())
    return path
