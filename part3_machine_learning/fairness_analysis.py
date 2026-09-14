"""
fairness_analysis.py - Task 7: evidence for the bias, fairness, governance and sustainability report.

1. Coverage   : hours observed vs expected by year, long gaps, rare categories.
2. Proxy label: how the high_risk rate varies by year, weather, time and day type, and how many
                potential positives were lost when Part 2 kept only the primary weather condition.
3. Errors     : regression MAE/bias and classifier recall/false-positive rate per segment on the
                2018 test set (time of day, day type, weather group, season, holiday).
4. Footprint  : training time, estimated energy/CO2, model size and inference latency per model.

Outputs are written to reports/metrics/ and figures/task7_responsible_ai/ and cited in
reports/BIAS_FAIRNESS_REPORT.md.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

import plotting as P
from config import (
    CPU_POWER_WATTS,
    GRID_KG_CO2_PER_KWH,
    METRICS_DIR,
    MODEL_FEATURES,
    MODEL_REGISTRY_FILE,
    MODELS_DIR,
    PART2_RAW_DATA,
)
from data_prep import load_modelling_table
from feature_engineering import LOW_VISIBILITY_WEATHER, SEVERE_WEATHER

logger = logging.getLogger(__name__)

HOUR_BANDS = [(0, 5, "Night 00-05"), (6, 9, "AM peak 06-09"), (10, 14, "Midday 10-14"), (15, 18, "PM peak 15-18"),
              (19, 23, "Evening 19-23")]
WEATHER_GROUPS = {"Clear": "Clear", "Clouds": "Cloudy", "Rain": "Rain/Drizzle", "Drizzle": "Rain/Drizzle", "Snow": "Snow",
                  "Mist": "Low visibility", "Fog": "Low visibility", "Haze": "Low visibility", "Smoke": "Low visibility",
                  "Thunderstorm": "Thunderstorm/Squall", "Squall": "Thunderstorm/Squall"}
SEASONS = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring", 6: "Summer", 7: "Summer",
           8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}


def add_segments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_band"] = ""
    for start, end, label in HOUR_BANDS:
        df.loc[df["hour"].between(start, end), "hour_band"] = label
    df["day_type"] = np.where(df["is_holiday"] == 1, "Holiday", np.where(df["is_weekend"] == 1, "Weekend", "Weekday"))
    df["weather_group"] = df["weather_main"].map(WEATHER_GROUPS).fillna("Other")
    df["season"] = df["date_time"].dt.month.map(SEASONS)
    return df


# ---------------------------------------------------------------------------
# 1. Coverage
# ---------------------------------------------------------------------------
def coverage(df: pd.DataFrame) -> dict[str, Any]:
    years = df["date_time"].dt.year
    per_year = []
    for year, group in df.groupby(years):
        start = max(pd.Timestamp(f"{year}-01-01"), df["date_time"].min().floor("D"))
        end = min(pd.Timestamp(f"{year}-12-31 23:00"), df["date_time"].max())
        expected = int((end - start) / pd.Timedelta(hours=1)) + 1
        per_year.append({"year": int(year), "hours_observed": len(group), "hours_expected": expected,
                         "coverage": len(group) / expected, "high_risk_rate": group["high_risk"].mean()})
    per_year = pd.DataFrame(per_year)
    gaps = df["date_time"].diff().dt.total_seconds().div(3600).sub(1)
    long_gap = df.loc[gaps.idxmax()]
    rare = df["weather_main"].value_counts()
    result = {
        "per_year": per_year,
        "longest_gap_hours": int(gaps.max()),
        "longest_gap_end": str(long_gap["date_time"]),
        "rare_weather": rare[rare < 250].to_dict(),
        "holiday_days": int(df.loc[df["is_holiday"] == 1, "date"].nunique()),
    }
    per_year.to_csv(METRICS_DIR / "fairness_coverage_by_year.csv", index=False, float_format="%.4f")
    logger.info("Coverage: lowest year coverage %.0f%% (%d); longest gap %d hours ending %s; rare weather %s",
                100 * per_year["coverage"].min(), int(per_year.loc[per_year["coverage"].idxmin(), "year"]),
                result["longest_gap_hours"], result["longest_gap_end"], result["rare_weather"])
    return result


# ---------------------------------------------------------------------------
# 2. Proxy label
# ---------------------------------------------------------------------------
def proxy_label_analysis(df: pd.DataFrame) -> dict[str, Any]:
    tables = {}
    for column in ("weather_group", "hour_band", "day_type", "season"):
        table = df.groupby(column)["high_risk"].agg(rate="mean", positives="sum", hours="size").reset_index()
        table.insert(0, "segment_type", column)
        tables[column] = table.rename(columns={column: "segment"})
    combined = pd.concat(tables.values(), ignore_index=True)
    combined.to_csv(METRICS_DIR / "fairness_proxy_label_rates.csv", index=False, float_format="%.4f")

    positives = df[df["high_risk"] == 1]
    mist_share = float((positives["weather_main"] == "Mist").mean())

    # Positives lost to de-duplication: hours whose SECONDARY weather condition was risky.
    raw = pd.read_csv(PART2_RAW_DATA, keep_default_na=False, na_values=[""], parse_dates=["date_time"])
    raw["weather_main"] = raw["weather_main"].str.strip().str.title().replace({"Squalls": "Squall"})
    risky_any = raw.assign(risky=raw["weather_main"].isin(SEVERE_WEATHER | LOW_VISIBILITY_WEATHER)) \
        .groupby("date_time")["risky"].any()
    merged = df.set_index("date_time").join(risky_any.rename("any_condition_risky"), how="left")
    high_congestion = merged["congestion_category"].isin(["High", "Severe"])
    would_be_positive = high_congestion & merged["any_condition_risky"].fillna(False)
    lost = int((would_be_positive & (merged["high_risk"] == 0)).sum())
    result = {"rates": combined, "mist_share_of_positives": mist_share, "positives": int(df["high_risk"].sum()),
              "positives_lost_to_dedup": lost,
              "clear_or_cloudy_positive_rate": float(df.loc[df["weather_group"].isin(["Clear", "Cloudy"]), "high_risk"].mean())}
    logger.info("Proxy label: %d positives, %.0f%% of them Mist; clear/cloudy hours can never be positive (rate %.3f)",
                result["positives"], 100 * mist_share, result["clear_or_cloudy_positive_rate"])
    logger.warning("Proxy label: %d additional hours would be high risk if secondary weather conditions had been kept "
                   "(%.0f%% more positives) - a labelling bias introduced by de-duplication", lost,
                   100 * lost / max(result["positives"], 1))
    return result


# ---------------------------------------------------------------------------
# 3. Error distribution
# ---------------------------------------------------------------------------
def error_distribution(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    test = df[df["split"] == "test"].copy()
    X = test[MODEL_FEATURES].astype(float)
    regressor = joblib.load(MODELS_DIR / "traffic_volume_regressor.joblib")
    classifier = joblib.load(MODELS_DIR / "traffic_risk_classifier.joblib")
    threshold = json.loads((MODELS_DIR / "traffic_risk_classifier.json").read_text(encoding="utf-8"))["decision_threshold"]
    test["error"] = regressor.predict(X) - test["traffic_volume"]
    test["risk_pred"] = (classifier.predict_proba(X)[:, 1] >= threshold).astype(int)

    reg_rows, clf_rows = [], []
    overall_mae = test["error"].abs().mean()
    for column in ("hour_band", "day_type", "weather_group", "season"):
        for segment, group in test.groupby(column):
            mae = group["error"].abs().mean()
            reg_rows.append({"segment_type": column, "segment": segment, "hours": len(group), "mae": mae,
                             "relative_mae": mae / group["traffic_volume"].mean(), "bias": group["error"].mean(),
                             "mae_vs_overall": mae / overall_mae})
            positives, negatives = group["high_risk"] == 1, group["high_risk"] == 0
            clf_rows.append({"segment_type": column, "segment": segment, "hours": len(group),
                             "positives": int(positives.sum()),
                             "recall": group.loc[positives, "risk_pred"].mean() if positives.any() else np.nan,
                             "false_positive_rate": group.loc[negatives, "risk_pred"].mean() if negatives.any() else np.nan,
                             "precision": group.loc[group["risk_pred"] == 1, "high_risk"].mean()
                             if group["risk_pred"].any() else np.nan})
    regression = pd.DataFrame(reg_rows)
    classification = pd.DataFrame(clf_rows)
    regression.to_csv(METRICS_DIR / "fairness_regression_errors_by_segment.csv", index=False, float_format="%.4f")
    classification.to_csv(METRICS_DIR / "fairness_classifier_errors_by_segment.csv", index=False, float_format="%.4f")

    worst = regression.sort_values("mae_vs_overall", ascending=False).iloc[0]
    logger.info("Regression error by segment: worst %s=%s MAE %.0f (%.2fx overall %.0f)", worst["segment_type"],
                worst["segment"], worst["mae"], worst["mae_vs_overall"], overall_mae)
    uneven = regression[regression["mae_vs_overall"] > 1.25]
    if len(uneven):
        logger.warning("Uneven regression error: %d segment(s) exceed 1.25x overall MAE: %s", len(uneven),
                       ", ".join(f"{r.segment} ({r.mae_vs_overall:.2f}x)" for r in uneven.itertuples()))
    _plot_errors(regression, classification, overall_mae)
    return {"regression": regression, "classification": classification}


def _plot_errors(regression: pd.DataFrame, classification: pd.DataFrame, overall_mae: float) -> None:
    order = ["hour_band", "day_type", "weather_group", "season"]
    titles = {"hour_band": "Time of day", "day_type": "Day type", "weather_group": "Weather", "season": "Season"}
    fig, axes = P.plt.subplots(1, 4, figsize=(13, 4.6), sharex=True)
    for ax, column in zip(axes, order):
        data = regression[regression["segment_type"] == column].sort_values("mae")
        colours = [P.SERIES_2 if v > 1.25 else P.SERIES_1 for v in data["mae_vs_overall"]]
        ax.barh(data["segment"], data["mae"], color=colours, edgecolor=P.SURFACE, linewidth=2, height=0.7)
        ax.axvline(overall_mae, color=P.INK_MUTED, linestyle="--", linewidth=1)
        for i, (mae, n) in enumerate(zip(data["mae"], data["hours"])):
            ax.text(mae, i, f" {mae:,.0f} (n={n:,})", va="center", fontsize=7.5, color=P.INK_SECONDARY)
        ax.set_title(titles[column], fontsize=11)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=8.5)
    axes[0].set_xlim(0, regression["mae"].max() * 1.6)
    fig.supxlabel("Test MAE (vehicles/hour); dashed = overall; orange > 1.25x overall", fontsize=9.5, color=P.INK_SECONDARY)
    fig.suptitle("Regression errors are uneven: largest at peaks, on holidays and in snow", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    P.save_figure(fig, "task7_regression_error_by_segment.png", "task7_responsible_ai")

    fig, axes = P.plt.subplots(1, 4, figsize=(13, 4.6), sharex=True)
    for ax, column in zip(axes, order):
        data = classification[classification["segment_type"] == column]
        y = np.arange(len(data))
        ax.barh(y + 0.2, data["recall"].fillna(0), height=0.36, color=P.SERIES_1, label="Recall")
        ax.barh(y - 0.2, data["false_positive_rate"].fillna(0), height=0.36, color=P.SERIES_2, label="False positive rate")
        for i, (rec, fpr, pos) in enumerate(zip(data["recall"], data["false_positive_rate"], data["positives"])):
            ax.text((0 if np.isnan(rec) else rec) + 0.03, i + 0.2,
                    "recall n/a (0 positives)" if np.isnan(rec) else f"{rec:.2f} (n+={pos})",
                    va="center", fontsize=7.5, color=P.INK_SECONDARY)
            ax.text((0 if np.isnan(fpr) else fpr) + 0.03, i - 0.2, f"FPR {fpr:.2f}", va="center", fontsize=7.5,
                    color=P.INK_SECONDARY)
        ax.set_yticks(y, data["segment"], fontsize=8.5)
        ax.set_xlim(0, 1.6)
        ax.set_title(titles[column], fontsize=11)
        ax.grid(axis="y", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncols=2, fontsize=9, frameon=False)
    fig.suptitle("Proxy-risk classifier: positives exist only in risky weather, so fairness metrics are undefined elsewhere",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    P.save_figure(fig, "task7_classifier_error_by_segment.png", "task7_responsible_ai")


# ---------------------------------------------------------------------------
# 4. Footprint
# ---------------------------------------------------------------------------
def footprint(df: pd.DataFrame) -> pd.DataFrame:
    registry = json.loads(MODEL_REGISTRY_FILE.read_text(encoding="utf-8"))
    rows = []
    for model_name, block in registry.items():
        for v in block["versions"]:
            seconds = float(v.get("train_seconds", 0))
            kwh = seconds * CPU_POWER_WATTS / 3.6e6
            rows.append({"registered_model": model_name, "version": v["version"], "candidate": v["candidate"],
                         "train_seconds": seconds, "energy_wh": kwh * 1000, "co2_g": kwh * GRID_KG_CO2_PER_KWH * 1000,
                         "test_mae_or_f1": v["test"].get("mae", v["test"].get("f1"))})
    table = pd.DataFrame(rows)

    sample = df[df["split"] == "test"][MODEL_FEATURES].astype(float).head(1000)
    latency = {}
    for stem in ("traffic_volume_regressor", "traffic_risk_classifier"):
        model = joblib.load(MODELS_DIR / f"{stem}.joblib")
        start = time.perf_counter()
        for _ in range(5):
            model.predict(sample)
        latency[stem] = (time.perf_counter() - start) / 5 / len(sample) * 1e6
    lstm_state = torch.load(MODELS_DIR / "lstm_traffic_volume.pt", map_location="cpu")
    lstm_params = int(sum(t.numel() for t in lstm_state.values()))
    sizes = {p.name: p.stat().st_size / 1e6 for p in MODELS_DIR.iterdir() if p.suffix in (".joblib", ".pt")}
    extra = pd.DataFrame([{"artifact": k, "size_mb": v} for k, v in sizes.items()])
    extra.to_csv(METRICS_DIR / "sustainability_model_sizes.csv", index=False, float_format="%.3f")
    table.to_csv(METRICS_DIR / "sustainability_training_footprint.csv", index=False, float_format="%.5f")
    (METRICS_DIR / "sustainability_inference.json").write_text(json.dumps(
        {"microseconds_per_prediction": latency, "lstm_parameters": lstm_params,
         "assumptions": {"cpu_power_watts": CPU_POWER_WATTS, "grid_kg_co2_per_kwh": GRID_KG_CO2_PER_KWH}}, indent=2),
        encoding="utf-8")
    logger.info("Footprint: total training %.0f s ≈ %.2f Wh ≈ %.2f g CO2 across %d model versions; inference %s µs/prediction",
                table["train_seconds"].sum(), table["energy_wh"].sum(), table["co2_g"].sum(), len(table),
                {k: round(v, 1) for k, v in latency.items()})
    return table


def run(df: pd.DataFrame | None = None) -> dict[str, Any]:
    df = add_segments(load_modelling_table() if df is None else df)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    return {"coverage": coverage(df), "proxy": proxy_label_analysis(df), "errors": error_distribution(df),
            "footprint": footprint(df)}


if __name__ == "__main__":
    from logging_config import configure_logging

    configure_logging("fairness_analysis")
    run()
