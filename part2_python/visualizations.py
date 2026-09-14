"""
visualizations.py - Matplotlib figures that reveal traffic patterns.

Figures (written to figures/)
-----------------------------
fig01_hourly_profile_weekday_weekend.png   Traffic demand by hour, workday vs weekend
fig02_heatmap_day_hour.png                 Average volume for every day-of-week x hour
fig03_weather_impact_daytime.png           Workday daytime traffic by weather condition
fig04_temperature_vs_traffic.png           Temperature vs traffic (daytime workdays)
fig05_traffic_distribution.png             Distribution of hourly volume + congestion thresholds
fig06_congestion_share_by_hour.png         Share of congestion levels by hour of day
fig07_monthly_trend.png                    Monthly average traffic over time

This module only obtains a module logger; handlers are configured by the entry point.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: figures are written to files, never shown

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

from feature_engineering import CONGESTION_LABELS, get_congestion_thresholds  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Visual style (validated colour-blind-safe reference palette)
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES_1 = "#2a78d6"  # blue  - workdays / primary series
SERIES_2 = "#eb6834"  # orange - weekends / comparison series
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORDINAL_BLUES = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # Low -> Very High

DAYTIME_HOURS = range(7, 19)  # 07:00-18:59, commuter day
MIN_WEATHER_SAMPLES = 50
DPI = 150

plt.rcParams.update(
    {
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK_PRIMARY,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 22,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.8,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "legend.frameon": False,
        "legend.labelcolor": INK_SECONDARY,
        "lines.linewidth": 2,
    }
)

thousands = FuncFormatter(lambda x, _: f"{x:,.0f}")


def _subtitle(ax: plt.Axes, text: str) -> None:
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK_SECONDARY, fontsize=9.5, va="bottom")


def _save(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    """Save and close a figure, logging the confirmed file path."""
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
    finally:
        plt.close(fig)
    logger.info("Saved figure: %s", _relative(path))
    return path


def _relative(path: Path) -> str:
    """Show paths relative to the working directory so logs stay portable."""
    try:
        return Path(path).resolve().relative_to(Path.cwd()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _workdays(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["is_weekend"] == 0) & (df["is_holiday"] == 0)]


# ---------------------------------------------------------------------------
# Individual figures
# ---------------------------------------------------------------------------
def plot_hourly_profile(df: pd.DataFrame, output_dir: Path) -> Path:
    workday = _workdays(df).groupby("hour")["traffic_volume"]
    weekend = df[df["is_weekend"] == 1].groupby("hour")["traffic_volume"]
    hours = np.arange(24)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    for grouped, colour, label in ((workday, SERIES_1, "Workdays (Mon-Fri, excl. holidays)"), (weekend, SERIES_2, "Weekends")):
        mean = grouped.mean().reindex(hours)
        q25, q75 = grouped.quantile(0.25).reindex(hours), grouped.quantile(0.75).reindex(hours)
        ax.fill_between(hours, q25, q75, color=colour, alpha=0.12, linewidth=0)
        ax.plot(hours, mean, color=colour, label=label, marker="o", markersize=4)
        logger.debug("%s hourly mean peak: hour %d (%.0f vehicles)", label, mean.idxmax(), mean.max())

    wd_mean = workday.mean()
    for peak_hour in (wd_mean.loc[:11].idxmax(), wd_mean.loc[12:].idxmax()):
        ax.annotate(
            f"{peak_hour:02d}:00  {wd_mean[peak_hour]:,.0f}",
            (peak_hour, wd_mean[peak_hour]), xytext=(0, 10), textcoords="offset points",
            ha="center", color=INK_PRIMARY, fontsize=9,
        )

    ax.set_title("Workdays have two sharp commuter peaks; weekends build to a single midday plateau")
    _subtitle(ax, "Mean hourly traffic volume on I-94 westbound, 2012-2018. Shaded band = interquartile range.")
    ax.set_xticks(range(0, 24, 2), [f"{h:02d}:00" for h in range(0, 24, 2)])
    ax.set_xlim(-0.5, 23.5)
    ax.yaxis.set_major_formatter(thousands)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Vehicles per hour")
    ax.set_ylim(0, wd_mean.max() * 1.12)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower center", ncols=2, bbox_to_anchor=(0.5, 0.0))
    return _save(fig, output_dir, "fig01_hourly_profile_weekday_weekend.png")


def plot_day_hour_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    pivot = df.pivot_table(index="day_of_week", columns="hour", values="traffic_volume", aggfunc="mean")
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    cmap = LinearSegmentedColormap.from_list("blues", BLUE_RAMP)

    fig, ax = plt.subplots(figsize=(11, 4.4))
    image = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    ax.set_xticks(range(24), [f"{h:02d}" for h in range(24)])
    ax.set_yticks(range(7), day_labels)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, 24, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 7, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)

    colourbar = fig.colorbar(image, ax=ax, pad=0.015, fraction=0.03, format=thousands)
    colourbar.outline.set_visible(False)
    colourbar.set_label("Mean vehicles per hour", color=INK_SECONDARY)
    colourbar.ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY)

    ax.set_title("Peak congestion is a Monday-Friday, 06:00-08:00 and 15:00-17:00 phenomenon")
    _subtitle(ax, "Mean traffic volume by day of week and hour of day (darker = busier)")
    ax.set_xlabel("Hour of day")
    return _save(fig, output_dir, "fig02_heatmap_day_hour.png")


def plot_weather_impact(df: pd.DataFrame, output_dir: Path) -> Path:
    """Compare weather conditions within workday daytime hours only.

    Restricting to the same hours removes the confound that some conditions
    (e.g. mist, fog) occur mostly at night when traffic is naturally low.
    """
    subset = _workdays(df)
    subset = subset[subset["hour"].isin(DAYTIME_HOURS)]
    stats = subset.groupby("weather_main")["traffic_volume"].agg(["mean", "size"])
    excluded = stats[stats["size"] < MIN_WEATHER_SAMPLES].index.tolist()
    stats = stats[stats["size"] >= MIN_WEATHER_SAMPLES].sort_values("mean")
    baseline = stats.loc["Clear", "mean"] if "Clear" in stats.index else stats["mean"].median()
    logger.debug("Daytime workday mean volume by weather: %s", stats["mean"].round(0).to_dict())
    if excluded:
        logger.debug("Weather categories excluded from figure (< %d samples): %s", MIN_WEATHER_SAMPLES, excluded)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    positions = np.arange(len(stats))
    bar_colours = [SERIES_2 if condition == "Snow" else SERIES_1 for condition in stats.index]
    ax.barh(positions, stats["mean"], color=bar_colours, height=0.72, edgecolor=SURFACE, linewidth=2)
    ax.axvline(baseline, color=INK_PRIMARY, linestyle="--", linewidth=1)
    ax.text(baseline - 30, len(stats) - 0.4, f"Clear sky {baseline:,.0f}", color=INK_PRIMARY, fontsize=8.5,
            va="bottom", ha="right")
    label_x = stats["mean"].max() * 1.04  # aligned label column to the right of every bar
    for pos, (condition, row) in zip(positions, stats.iterrows()):
        diff = (row["mean"] / baseline - 1) * 100
        ax.text(
            label_x, pos, f"{row['mean']:,.0f}   {diff:+.1f}% vs clear   n={int(row['size']):,}",
            va="center", color=INK_SECONDARY, fontsize=8.5,
        )
    ax.set_yticks(positions, stats.index)
    ax.set_xlim(0, stats["mean"].max() * 1.45)
    ax.set_ylim(-0.6, len(stats) - 0.1)
    ax.spines["bottom"].set_bounds(0, 6000)
    ax.set_xticks(range(0, 6001, 1000))
    ax.xaxis.set_major_formatter(thousands)
    ax.grid(axis="y", visible=False)
    ax.set_title("Snow is the only weather condition that clearly suppresses daytime traffic")
    _subtitle(ax, "Mean traffic volume on workdays, 07:00-18:59, by primary weather condition (vs clear sky)")
    ax.set_xlabel("Mean vehicles per hour")
    return _save(fig, output_dir, "fig03_weather_impact_daytime.png")


def plot_temperature_vs_traffic(df: pd.DataFrame, output_dir: Path) -> Path:
    subset = _workdays(df)
    subset = subset[subset["hour"].isin(DAYTIME_HOURS)]
    correlation = subset["temp_c"].corr(subset["traffic_volume"])
    logger.debug("Pearson correlation temp_c vs traffic_volume (workday daytime): %.3f", correlation)

    bins = np.arange(-30, 40, 5)
    binned = subset.groupby(pd.cut(subset["temp_c"], bins), observed=True)["traffic_volume"].agg(["median", "size"])
    binned = binned[binned["size"] >= 30]
    centres = [interval.mid for interval in binned.index]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    cmap = LinearSegmentedColormap.from_list("blues", [SURFACE] + BLUE_RAMP)
    hexes = ax.hexbin(subset["temp_c"], subset["traffic_volume"], gridsize=45, cmap=cmap, mincnt=1, linewidths=0)
    ax.plot(centres, binned["median"], color=SERIES_2, marker="o", markersize=6, label="Median per 5 °C band")
    colourbar = fig.colorbar(hexes, ax=ax, pad=0.015, fraction=0.03)
    colourbar.outline.set_visible(False)
    colourbar.set_label("Hours in cell", color=INK_SECONDARY)
    colourbar.ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY)

    ax.set_title("Daytime traffic barely responds to temperature; it dips only in deep cold (below -15 °C)")
    _subtitle(ax, f"Workday hours 07:00-18:59. Pearson r = {correlation:.2f}")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Vehicles per hour")
    ax.yaxis.set_major_formatter(thousands)
    ax.grid(False)
    ax.legend(loc="lower right")
    return _save(fig, output_dir, "fig04_temperature_vs_traffic.png")


def plot_traffic_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    q1, q2, q3 = get_congestion_thresholds(df["traffic_volume"])
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.hist(df["traffic_volume"], bins=np.arange(0, 7600, 200), color=SERIES_1, edgecolor=SURFACE, linewidth=1)

    y_top = ax.get_ylim()[1] * 1.12
    ax.set_ylim(0, y_top)
    edges = [0, q1, q2, q3, df["traffic_volume"].max()]
    for threshold, name in ((q1, "Q1"), (q2, "Median"), (q3, "Q3")):
        ax.axvline(threshold, color=INK_PRIMARY, linestyle="--", linewidth=1)
        ax.text(threshold + 40, y_top * 0.985, f"{name} = {threshold:,.0f}", ha="left", va="top", fontsize=8.5,
                color=INK_PRIMARY)
    for label, lo, hi in zip(CONGESTION_LABELS, edges[:-1], edges[1:]):
        ax.text((lo + hi) / 2, y_top * 0.9, label, ha="center", va="top", fontsize=9.5, color=INK_SECONDARY,
                weight="semibold")

    ax.set_title("Hourly volume forms three regimes: night (<1,000), shoulder (~2,800) and daytime (4,000-6,000)")
    _subtitle(ax, "Distribution of hourly traffic volume. Dashed lines = quartile thresholds that define congestion_level")
    ax.set_xlabel("Vehicles per hour")
    ax.set_ylabel("Number of hours")
    ax.xaxis.set_major_formatter(thousands)
    ax.yaxis.set_major_formatter(thousands)
    ax.grid(axis="x", visible=False)
    return _save(fig, output_dir, "fig05_traffic_distribution.png")


def plot_congestion_share_by_hour(df: pd.DataFrame, output_dir: Path) -> Path:
    share = pd.crosstab(df["hour"], df["congestion_level"], normalize="index").reindex(columns=CONGESTION_LABELS)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    bottom = np.zeros(len(share))
    for label, colour in zip(CONGESTION_LABELS, ORDINAL_BLUES):
        ax.bar(share.index, share[label], bottom=bottom, color=colour, edgecolor=SURFACE, linewidth=1.5, width=0.9, label=label)
        bottom += share[label].to_numpy()
    ax.set_title("Very High congestion is concentrated in 06:00-08:00 and 14:00-17:00")
    _subtitle(ax, "Share of hours in each congestion level, by hour of day (all days)")
    ax.set_xticks(range(0, 24, 2), [f"{h:02d}:00" for h in range(0, 24, 2)])
    ax.set_xlim(-0.6, 23.6)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Share of hours")
    ax.grid(False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(1.0, 1.0), title="Congestion level",
              title_fontsize=9)
    return _save(fig, output_dir, "fig06_congestion_share_by_hour.png")


def plot_monthly_trend(df: pd.DataFrame, output_dir: Path) -> Path:
    monthly = df.set_index("date_time")["traffic_volume"].resample("MS").agg(["mean", "size"])
    # Months with very little data (collection gaps) are shown as breaks, not as misleading dips.
    monthly.loc[monthly["size"] < 24 * 7, "mean"] = np.nan
    n_gap = int(monthly["mean"].isna().sum())
    logger.debug("Monthly trend: %d month(s) with < 7 days of data shown as gaps", n_gap)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(monthly.index, monthly["mean"], color=SERIES_1, marker="o", markersize=3.5)
    gap = monthly["mean"].isna()
    if gap.any():
        gap_months = monthly.index[gap]
        ax.axvspan(gap_months.min(), gap_months.max() + pd.offsets.MonthEnd(1), color=GRIDLINE, alpha=0.5, linewidth=0)
        ax.text(gap_months.min() + (gap_months.max() - gap_months.min()) / 2, ax.get_ylim()[0] + 20,
                "Sensor data gap", ha="center", va="bottom", fontsize=8.5, color=INK_SECONDARY)
    ax.set_title("Average traffic is stable year to year, with dips each winter")
    _subtitle(ax, "Monthly mean hourly traffic volume, Oct 2012 - Sep 2018")
    ax.set_ylabel("Mean vehicles per hour")
    ax.yaxis.set_major_formatter(thousands)
    ax.grid(axis="x", visible=False)
    return _save(fig, output_dir, "fig07_monthly_trend.png")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
FIGURE_FUNCTIONS = (
    plot_hourly_profile,
    plot_day_hour_heatmap,
    plot_weather_impact,
    plot_temperature_vs_traffic,
    plot_traffic_distribution,
    plot_congestion_share_by_hour,
    plot_monthly_trend,
)


def create_all_figures(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Generate every figure; raise if any figure could not be produced."""
    output_dir = Path(output_dir)
    logger.info("Generating %d figures in %s", len(FIGURE_FUNCTIONS), _relative(output_dir))
    saved: list[Path] = []
    failed: list[str] = []
    for plot in FIGURE_FUNCTIONS:
        try:
            saved.append(plot(df, output_dir))
        except (OSError, ValueError, KeyError) as exc:
            logger.error("Could not create figure with %s: %s", plot.__name__, exc, exc_info=True)
            failed.append(plot.__name__)
    if failed:
        raise RuntimeError(f"{len(failed)} figure(s) failed: {failed}")
    logger.info("All %d figures saved successfully", len(saved))
    return saved
