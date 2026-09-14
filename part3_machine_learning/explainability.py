"""
explainability.py - Task 3 (explainability): SHAP explanations.

Why a surrogate for the LSTM?
-----------------------------
SHAP's model-agnostic explainers on a recurrent network need thousands of forward passes
over 24x10 input tensors and yield attributions per time step, which are slow and hard to
communicate. As permitted by the brief, the LSTM is explained through a *comparable*
HistGradientBoosting model trained on the same next-hour problem, the same samples and the
same information (lagged volume summaries + calendar + weather). TreeSHAP gives exact,
fast Shapley values for it. deep_learning.py shows the two models reach similar accuracy,
which is what makes the surrogate's explanations a reasonable proxy for the LSTM's behaviour.

Models explained
----------------
1. Next-hour surrogate (lag features)  -> what drives short-term demand
2. Task 1 regression champion          -> what drives demand without recent history
3. Task 1 proxy-risk classifier        -> exposes that the label is defined by weather + hour
"""

from __future__ import annotations

import logging
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import shap

import plotting as P
from config import EXPERIMENT_DEEP_LEARNING, METRICS_DIR, MODEL_FEATURES, MODELS_DIR, RANDOM_SEED
from data_prep import load_modelling_table
from deep_learning import LAG_FEATURES, build_sequences
from mlflow_utils import setup_experiment

logger = logging.getLogger(__name__)

SHAP_SAMPLE = 2000
SHAP_CMAP = P.plt.matplotlib.colors.LinearSegmentedColormap.from_list("shap_blue_orange", ["#2a78d6", "#c3c2b7", "#eb6834"])


def _sample(frame: pd.DataFrame, n: int = SHAP_SAMPLE) -> pd.DataFrame:
    return frame.sample(n=min(n, len(frame)), random_state=RANDOM_SEED)


def _save_importance(explanation: shap.Explanation, name: str) -> pd.DataFrame:
    importance = (pd.DataFrame({"feature": explanation.feature_names,
                                "mean_abs_shap": np.abs(explanation.values).mean(axis=0)})
                  .sort_values("mean_abs_shap", ascending=False))
    importance["share"] = importance["mean_abs_shap"] / importance["mean_abs_shap"].sum()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / f"shap_importance_{name}.csv"
    importance.to_csv(path, index=False, float_format="%.5f")
    top = ", ".join(f"{r.feature} ({r.share:.0%})" for r in importance.head(5).itertuples())
    logger.info("SHAP top features for %s: %s", name, top)
    return importance


def _beeswarm(explanation: shap.Explanation, title: str, subtitle: str, filename: str, xlabel: str) -> None:
    shap.plots.beeswarm(explanation, max_display=12, show=False, color=SHAP_CMAP, plot_size=(10, 5.6))
    fig = P.plt.gcf()
    ax = P.plt.gca()
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    P.subtitle(ax, subtitle)
    P.save_figure(fig, filename, "task3_explainability")


def _bar(importance: pd.DataFrame, title: str, subtitle: str, filename: str, unit: str) -> None:
    top = importance.head(12).iloc[::-1]
    fig, ax = P.plt.subplots(figsize=(10, 5))
    ax.barh(top["feature"], top["mean_abs_shap"], color=P.SERIES_1, edgecolor=P.SURFACE, linewidth=2, height=0.72)
    for i, (value, share) in enumerate(zip(top["mean_abs_shap"], top["share"])):
        ax.text(value, i, f"  {value:,.2f} ({share:.0%})" if value < 10 else f"  {value:,.0f} ({share:.0%})",
                va="center", fontsize=8.5, color=P.INK_SECONDARY)
    ax.set_xlim(0, top["mean_abs_shap"].max() * 1.25)
    ax.set_xlabel(f"Mean |SHAP value| ({unit})")
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    P.subtitle(ax, subtitle)
    P.save_figure(fig, filename, "task3_explainability")


def _waterfall(single: shap.Explanation, row: pd.Series, top_n: int = 9) -> None:
    """Local explanation as a waterfall: average prediction -> feature pushes -> this prediction."""
    base = float(np.ravel(single.base_values)[0])
    contributions = pd.Series(single.values, index=single.feature_names)
    data_values = pd.Series(single.data, index=single.feature_names)
    order = contributions.abs().sort_values(ascending=False).index
    top = contributions[order[:top_n]]
    other = contributions[order[top_n:]].sum()
    steps = list(top.items()) + [(f"{len(order) - top_n} other features", other)]
    prediction = base + contributions.sum()

    fig, ax = P.plt.subplots(figsize=(10, 5.6))
    running = base
    for i, (name, value) in enumerate(steps):
        colour = P.SERIES_2 if value >= 0 else P.SERIES_1
        ax.barh(i, value, left=running, color=colour, height=0.66, edgecolor=P.SURFACE, linewidth=1.5)
        label_x = running + value if value >= 0 else running + value
        ax.text(label_x + (25 if value >= 0 else -25), i, f"{value:+,.0f}", va="center",
                ha="left" if value >= 0 else "right", fontsize=8.5, color=P.INK_PRIMARY)
        running += value
    labels = []
    for name, _ in steps:
        if name in data_values.index:
            value = data_values[name]
            shown = f"{value:,.0f}" if abs(value) >= 10 else f"{value:.2f}".rstrip("0").rstrip(".")
            labels.append(f"{name} = {shown}")
        else:
            labels.append(name)
    ax.set_yticks(range(len(steps)), labels)
    ax.set_ylim(len(steps) - 0.4, -1.4)  # inverted, with headroom above the first bar for the reference labels
    cumulative = base + np.cumsum([v for _, v in steps])
    low, high = min(base, cumulative.min()), max(base, cumulative.max())
    pad = (high - low) * 0.12
    ax.set_xlim(low - pad, high + pad)
    ax.axvline(base, color=P.INK_MUTED, linestyle="--", linewidth=1)
    ax.axvline(prediction, color=P.INK_PRIMARY, linewidth=1)
    base_left = base < prediction
    ax.text(base, -1.0, f"average prediction {base:,.0f} ", ha="right" if base_left else "left", va="center",
            fontsize=8.5, color=P.INK_SECONDARY)
    ax.text(prediction, -1.0, f" this prediction {prediction:,.0f}", ha="left" if base_left else "right", va="center",
            fontsize=8.5, color=P.INK_PRIMARY)
    ax.xaxis.set_major_formatter(P.thousands)
    ax.set_xlabel("Predicted vehicles per hour")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Local explanation: {row['date_time']:%a %d %b %Y %H:%M}, {row['weather_main'].lower()} "
                 f"(actual {int(row['traffic_volume']):,})")
    P.subtitle(ax, "Each bar pushes the prediction up (orange) or down (blue) from the average; next-hour surrogate")
    P.save_figure(fig, "task3_shap_surrogate_waterfall.png", "task3_explainability")


def explain_surrogate(df: pd.DataFrame) -> dict[str, Any]:
    bundle = joblib.load(MODELS_DIR / "lag_surrogate_regressor.joblib")
    model, features = bundle["model"], bundle["features"]
    data = build_sequences(df)
    test = data["tabular"][data["split"] == "test"][features]
    X = _sample(test)
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)
    logger.info("Computed SHAP values for next-hour surrogate on %d test samples (base value %.0f vehicles/hour)",
                len(X), float(np.ravel(explanation.base_values)[0]))
    importance = _save_importance(explanation, "next_hour_surrogate")
    lag_share = importance.loc[importance["feature"].isin(LAG_FEATURES), "share"].sum()

    _beeswarm(explanation, "Next-hour demand is driven by the last hour's volume and the time of day",
              "SHAP beeswarm for the HistGradientBoosting + lags surrogate of the LSTM (2,000 test hours)",
              "task3_shap_surrogate_beeswarm.png", "SHAP value (impact on predicted vehicles/hour)")
    _bar(importance, f"Recent-history features account for {lag_share:.0%} of total attribution",
         "Mean absolute SHAP value per feature, next-hour surrogate model", "task3_shap_surrogate_importance.png",
         "vehicles/hour")

    # Local explanation: one weekday morning-peak hour.
    frame = data["frame"][data["split"] == "test"].reset_index(drop=True)
    candidates = frame.index[(frame["hour"] == 8) & (frame["is_weekend"] == 0) & (frame["weather_main"] == "Snow")]
    idx = int(candidates[0]) if len(candidates) else int(frame.index[(frame["hour"] == 8)][0])
    single = explainer(test.iloc[[idx]])[0]
    _waterfall(single, frame.loc[idx])

    dependence_feature = "hour_cos"
    fig, ax = P.plt.subplots(figsize=(10, 4.6))
    hours = X["hour"] if "hour" in X else None
    column = list(X.columns).index(dependence_feature)
    scatter = ax.scatter(hours, explanation.values[:, column], c=X["is_weekend"], cmap=P.plt.matplotlib.colors.ListedColormap(
        [P.SERIES_1, P.SERIES_2]), s=10, alpha=0.6, linewidths=0)
    ax.set_xticks(range(0, 24, 3), [f"{h:02d}:00" for h in range(0, 24, 3)])
    ax.set(xlabel="Hour of day", ylabel="SHAP value of hour_cos (vehicles/hour)")
    ax.axhline(0, color=P.INK_MUTED, linewidth=1)
    handles = [P.plt.Line2D([], [], marker="o", linestyle="", color=c, label=l)
               for c, l in ((P.SERIES_1, "Weekday"), (P.SERIES_2, "Weekend"))]
    ax.legend(handles=handles, loc="upper right")
    ax.set_title("Time-of-day effect: the cyclical hour encoding lifts predictions by day and lowers them at night")
    P.subtitle(ax, "SHAP dependence of hour_cos in the next-hour surrogate, by hour and day type")
    P.save_figure(fig, "task3_shap_surrogate_dependence_hour.png", "task3_explainability")
    return {"importance": importance, "lag_share": lag_share}


def explain_regressor(df: pd.DataFrame) -> pd.DataFrame:
    model = joblib.load(MODELS_DIR / "traffic_volume_regressor.joblib")
    X = _sample(df.loc[df["split"] == "test", MODEL_FEATURES].astype(float), 1000)
    explanation = shap.TreeExplainer(model)(X, check_additivity=False)
    importance = _save_importance(explanation, "volume_regressor")
    _bar(importance, "Without recent history, hour and day type explain most of the demand model",
         "Mean absolute SHAP value per feature, Task 1 regression champion (1,000 test hours)",
         "task3_shap_regressor_importance.png", "vehicles/hour")
    return importance


def explain_classifier(df: pd.DataFrame) -> pd.DataFrame:
    model = joblib.load(MODELS_DIR / "traffic_risk_classifier.joblib")
    X = _sample(df.loc[df["split"] == "test", MODEL_FEATURES].astype(float))
    explanation = shap.TreeExplainer(model)(X)
    importance = _save_importance(explanation, "risk_classifier")
    weather_label_share = importance.loc[importance["feature"].isin(
        ["is_low_visibility", "weather_Mist", "weather_Haze", "weather_Fog", "weather_Smoke", "weather_Thunderstorm",
         "weather_Squall", "weather_severity", "is_severe_weather"]), "share"].sum()
    _beeswarm(explanation, f"Label-defining weather features carry {weather_label_share:.0%} of the risk model's attribution",
              "SHAP beeswarm for the proxy high-risk classifier (log-odds). The model mostly re-learns the label rule.",
              "task3_shap_classifier_beeswarm.png", "SHAP value (impact on log-odds of proxy high risk)")
    logger.warning("Proxy-label circularity: %.0f%% of classifier SHAP attribution comes from the weather features "
                   "that define the label", 100 * weather_label_share)
    return importance


def run(df: pd.DataFrame | None = None) -> dict[str, Any]:
    df = load_modelling_table() if df is None else df
    setup_experiment(EXPERIMENT_DEEP_LEARNING)
    with mlflow.start_run(run_name="shap_explainability"):
        mlflow.set_tags({"task": "explainability", "method": f"SHAP TreeExplainer {shap.__version__}"})
        surrogate = explain_surrogate(df)
        regressor = explain_regressor(df)
        classifier = explain_classifier(df)
        mlflow.log_metric("surrogate_lag_feature_share", surrogate["lag_share"])
        for name in ("next_hour_surrogate", "volume_regressor", "risk_classifier"):
            mlflow.log_artifact(str(METRICS_DIR / f"shap_importance_{name}.csv"), artifact_path="shap")
    return {"surrogate": surrogate, "regressor": regressor, "classifier": classifier}


if __name__ == "__main__":
    from logging_config import configure_logging

    configure_logging("explainability")
    run()
