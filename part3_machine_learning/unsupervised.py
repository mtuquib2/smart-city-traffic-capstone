"""
unsupervised.py - Task 2: K-means clustering of traffic conditions and association rule mining.

K-means
-------
Clusters hourly traffic conditions on: cyclical hour (sin/cos), weekend flag, weather
severity and traffic volume (all standardised). k is chosen from 2-9 using the silhouette
score (computed on a fixed random sample for speed) together with the inertia elbow.
Each cluster is profiled and given a descriptive name from its centroid characteristics.

Association rules
-----------------
Each hour becomes a "transaction" of discrete items: time of day, day type, weather group,
temperature band and congestion level. Apriori (mlxtend) mines frequent itemsets; rules whose
consequent is a single congestion level are ranked by lift and translated to plain language.
"""

from __future__ import annotations

import logging
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import plotting as P
from config import CONGESTION_ORDER, EXPERIMENT_UNSUPERVISED, METRICS_DIR, RANDOM_SEED
from data_prep import load_modelling_table
from mlflow_utils import setup_experiment

logger = logging.getLogger(__name__)

CLUSTER_FEATURES = ["hour_sin", "hour_cos", "weather_severity", "traffic_volume"]
K_RANGE = range(2, 9)
SILHOUETTE_TOLERANCE = 0.01   # prefer the smallest k whose silhouette is within this of the best
SILHOUETTE_SAMPLE = 8000
MIN_SUPPORT = 0.005          # itemset must occur in >= 0.5% of hours (~200 hours)
MIN_CONFIDENCE = 0.5
TOP_RULES = 10

TIME_OF_DAY_BINS = [(0, 5, "Night (00-05)"), (6, 9, "Morning peak (06-09)"), (10, 14, "Midday (10-14)"),
                    (15, 18, "Afternoon peak (15-18)"), (19, 23, "Evening (19-23)")]
WEATHER_GROUPS = {"Clear": "Clear", "Clouds": "Cloudy", "Rain": "Rain/Drizzle", "Drizzle": "Rain/Drizzle",
                  "Snow": "Snow", "Mist": "Low visibility", "Fog": "Low visibility", "Haze": "Low visibility",
                  "Smoke": "Low visibility", "Thunderstorm": "Thunderstorm/Squall", "Squall": "Thunderstorm/Squall"}


# ---------------------------------------------------------------------------
# K-means
# ---------------------------------------------------------------------------
def circular_mean_hour(hours: pd.Series) -> float:
    angles = 2 * np.pi * hours / 24
    mean_angle = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    return float((mean_angle * 24 / (2 * np.pi)) % 24)


def hour_concentration(hours: pd.Series) -> float:
    """Mean resultant length (0 = hours spread evenly around the clock, 1 = all the same hour)."""
    angles = 2 * np.pi * hours / 24
    return float(np.hypot(np.sin(angles).mean(), np.cos(angles).mean()))


def name_cluster(profile: pd.Series) -> str:
    hour = profile["typical_hour"]
    if profile["hour_concentration"] < 0.3:
        period = "all hours"
    elif hour >= 21 or hour < 5:
        period = "night"
    elif hour < 14:
        period = "morning-midday"
    else:
        period = "afternoon-evening"
    volume = profile["mean_volume"]
    level = "Low" if volume < 1500 else "Moderate" if volume < 4000 else "Heavy"
    parts = [f"{level} traffic, {period}"]
    parts.append("weekend" if profile["weekend_share"] > 0.6 else "weekday" if profile["weekend_share"] < 0.15 else "mixed days")
    if profile["mean_weather_severity"] >= 1.5:
        parts.append("adverse weather")
    return ", ".join(parts)


def run_kmeans(df: pd.DataFrame) -> dict[str, Any]:
    X = StandardScaler().fit_transform(df[CLUSTER_FEATURES].astype(float))
    rng = np.random.default_rng(RANDOM_SEED)
    sample_idx = rng.choice(len(X), size=min(SILHOUETTE_SAMPLE, len(X)), replace=False)

    scores = []
    for k in K_RANGE:
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED).fit(X)
        silhouette = silhouette_score(X[sample_idx], model.labels_[sample_idx])
        scores.append({"k": k, "inertia": model.inertia_, "silhouette": silhouette})
        logger.debug("k=%d inertia=%.0f silhouette=%.4f", k, model.inertia_, silhouette)
    scores_df = pd.DataFrame(scores)
    # k=2 trivially splits day/night, so require k >= 3. The silhouette curve is flat, so apply
    # parsimony: the smallest k whose silhouette is within SILHOUETTE_TOLERANCE of the best.
    eligible = scores_df[scores_df["k"] >= 3]
    best_silhouette = eligible["silhouette"].max()
    best_k = int(eligible.loc[eligible["silhouette"] >= best_silhouette - SILHOUETTE_TOLERANCE, "k"].min())
    logger.info("K-means model selection: k=%d (silhouette=%.3f; best in range %.3f, tolerance %.2f)", best_k,
                scores_df.set_index("k").loc[best_k, "silhouette"], best_silhouette, SILHOUETTE_TOLERANCE)

    model = KMeans(n_clusters=best_k, n_init=20, random_state=RANDOM_SEED).fit(X)
    labelled = df.assign(cluster=model.labels_)
    profiles = labelled.groupby("cluster").agg(
        hours=("traffic_volume", "size"),
        mean_volume=("traffic_volume", "mean"),
        typical_hour=("hour", circular_mean_hour),
        hour_concentration=("hour", hour_concentration),
        weekend_share=("is_weekend", "mean"),
        mean_weather_severity=("weather_severity", "mean"),
        adverse_weather_share=("weather_severity", lambda s: (s >= 2).mean()),
        severe_congestion_share=("congestion_category", lambda s: (s == "Severe").mean()),
        proxy_high_risk_share=("high_risk", "mean"),
        dominant_weather=("weather_main", lambda s: s.mode().iloc[0]),
        mean_temp_c=("temp_c", "mean"),
    )
    profiles["share_of_hours"] = profiles["hours"] / len(labelled)
    profiles["name"] = profiles.apply(name_cluster, axis=1)
    # Order clusters by volume so ids read low -> high in every output.
    profiles = profiles.sort_values("mean_volume")
    relabel = {old: new for new, old in enumerate(profiles.index)}
    profiles.index = [relabel[i] for i in profiles.index]
    profiles.index.name = "cluster"
    labelled["cluster"] = labelled["cluster"].map(relabel)
    for cluster_id, row in profiles.iterrows():
        logger.info("Cluster %d (%.1f%% of hours): %s | mean volume %.0f, typical hour %.1f, weekend %.0f%%, "
                    "adverse weather %.0f%%", cluster_id, 100 * row["share_of_hours"], row["name"], row["mean_volume"],
                    row["typical_hour"], 100 * row["weekend_share"], 100 * row["adverse_weather_share"])

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(METRICS_DIR / "kmeans_cluster_profiles.csv", float_format="%.4f")
    scores_df.to_csv(METRICS_DIR / "kmeans_model_selection.csv", index=False, float_format="%.4f")
    logger.info("Saved cluster profiles and k-selection scores to reports/metrics/")

    with mlflow.start_run(run_name="kmeans_traffic_conditions"):
        mlflow.set_tags({"task": "clustering", "algorithm": "KMeans"})
        mlflow.log_params({"features": ",".join(CLUSTER_FEATURES), "k_range": f"{K_RANGE.start}-{K_RANGE.stop - 1}",
                           "selected_k": best_k, "n_init": 20, "scaler": "StandardScaler"})
        for row in scores:
            mlflow.log_metric("silhouette_by_k", row["silhouette"], step=row["k"])
            mlflow.log_metric("inertia_by_k", row["inertia"], step=row["k"])
        mlflow.log_metric("silhouette", float(scores_df.set_index("k").loc[best_k, "silhouette"]))
        mlflow.log_artifact(str(METRICS_DIR / "kmeans_cluster_profiles.csv"))

    _plot_kmeans(scores_df, best_k, labelled, profiles)
    return {"best_k": best_k, "profiles": profiles, "scores": scores_df}


def _plot_kmeans(scores: pd.DataFrame, best_k: int, labelled: pd.DataFrame, profiles: pd.DataFrame) -> None:
    fig, (ax_in, ax_sil) = P.plt.subplots(1, 2, figsize=(11, 4.2))
    ax_in.plot(scores["k"], scores["inertia"], color=P.SERIES_1, marker="o")
    ax_sil.plot(scores["k"], scores["silhouette"], color=P.SERIES_1, marker="o")
    for ax in (ax_in, ax_sil):
        ax.axvline(best_k, color=P.SERIES_2, linestyle="--", linewidth=1.2)
        ax.set_xlabel("Number of clusters k")
        ax.grid(axis="x", visible=False)
    ax_in.set_ylabel("Inertia (within-cluster sum of squares)")
    ax_in.yaxis.set_major_formatter(P.thousands)
    ax_sil.set_ylabel("Silhouette score (8,000-hour sample)")
    ax_in.set_title(f"k = {best_k}: smallest k ≥ 3 within {SILHOUETTE_TOLERANCE} of the best silhouette")
    P.subtitle(ax_in, "K-means on hour (sin/cos), weather severity and traffic volume (standardised)")
    P.save_figure(fig, "task2_kmeans_model_selection.png", "task2_unsupervised")

    sample = labelled.sample(n=min(9000, len(labelled)), random_state=RANDOM_SEED)
    jitter = np.random.default_rng(RANDOM_SEED).uniform(-0.35, 0.35, len(sample))
    fig, ax = P.plt.subplots(figsize=(11, 5.4))
    # Largest clusters first so small ones (e.g. adverse weather) are drawn on top and stay visible.
    for cluster_id, row in profiles.sort_values("share_of_hours", ascending=False).iterrows():
        points = sample[sample["cluster"] == cluster_id]
        small = row["share_of_hours"] < 0.15
        ax.scatter(points["hour"] + jitter[sample["cluster"].to_numpy() == cluster_id], points["traffic_volume"],
                   s=14 if small else 6, alpha=0.8 if small else 0.4, color=P.CATEGORICAL[cluster_id], linewidths=0,
                   label=f"{cluster_id}: {row['name']} ({row['share_of_hours']:.0%})", zorder=3 if small else 2)
    ax.set_xticks(range(0, 24, 2), [f"{h:02d}:00" for h in range(0, 24, 2)])
    ax.set_xlabel("Hour of day (jittered)")
    ax.set_ylabel("Vehicles per hour")
    ax.yaxis.set_major_formatter(P.thousands)
    handles, labels = ax.get_legend_handles_labels()
    order = np.argsort([int(label.split(":")[0]) for label in labels])
    legend = ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left",
                       bbox_to_anchor=(1.0, 1.0), markerscale=2, fontsize=8.5, title="Cluster")
    for handle in legend.legend_handles:
        handle.set_alpha(1)
    ax.set_title("Traffic conditions form distinct time-of-day and weather regimes")
    P.subtitle(ax, "Hourly records coloured by K-means cluster (9,000-hour sample)")
    P.save_figure(fig, "task2_kmeans_clusters_hour_volume.png", "task2_unsupervised")

    columns = ["mean_volume", "typical_hour", "weekend_share", "adverse_weather_share", "severe_congestion_share",
               "proxy_high_risk_share", "mean_temp_c"]
    labels = ["Mean volume", "Typical hour", "Weekend share", "Adverse weather", "Severe congestion", "Proxy high risk", "Mean temp °C"]
    table = profiles[columns]
    normalised = (table - table.min()) / (table.max() - table.min()).replace(0, 1)
    fig, ax = P.plt.subplots(figsize=(11, 0.75 * len(profiles) + 1.8))
    ax.imshow(normalised.to_numpy(), aspect="auto",
              cmap=P.plt.matplotlib.colors.LinearSegmentedColormap.from_list("b", P.BLUE_RAMP[:5]))
    formats = ["{:,.0f}", "{:.1f}", "{:.0%}", "{:.0%}", "{:.0%}", "{:.1%}", "{:.1f}"]
    for i in range(len(table)):
        for j, fmt in enumerate(formats):
            value = table.iloc[i, j]
            text = fmt.format(value)
            if columns[j] == "typical_hour" and profiles["hour_concentration"].iloc[i] < 0.3:
                text = "spread"  # hours distributed around the clock: a mean hour would mislead
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    color="white" if normalised.iloc[i, j] > 0.6 else P.INK_PRIMARY)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(table)), [f"{i}: {n}" for i, n in zip(table.index, profiles["name"])])
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Cluster profiles")
    P.subtitle(ax, "Shading is relative within each column (darker = higher)")
    P.save_figure(fig, "task2_kmeans_cluster_profiles.png", "task2_unsupervised")


# ---------------------------------------------------------------------------
# Association rules
# ---------------------------------------------------------------------------
def to_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot boolean item matrix: one row per hour, one column per item."""
    time_of_day = pd.Series("", index=df.index)
    for start, end, label in TIME_OF_DAY_BINS:
        time_of_day[df["hour"].between(start, end)] = label
    day_type = np.where(df["is_holiday"] == 1, "Holiday", np.where(df["is_weekend"] == 1, "Weekend", "Weekday"))
    weather = df["weather_main"].map(WEATHER_GROUPS).fillna("Other")
    temp_band = pd.cut(df["temp_c"], [-100, 0, 10, 20, 100], labels=["Freezing (<=0C)", "Cold (0-10C)",
                                                                    "Mild (10-20C)", "Warm (>20C)"]).astype(str)
    items = pd.DataFrame({
        "time": "time=" + time_of_day,
        "day": "day=" + pd.Series(day_type, index=df.index),
        "weather": "weather=" + weather,
        "temp": "temp=" + temp_band,
        "congestion": "congestion=" + df["congestion_category"].astype(str),
    })
    transactions = pd.get_dummies(items, prefix="", prefix_sep="").astype(bool)
    logger.info("Built %d transactions with %d distinct items", *transactions.shape)
    return transactions


def describe_items(items: frozenset) -> str:
    """Turn an antecedent itemset into a readable clause ordered time -> day -> weather -> temperature."""
    parts = dict(item.split("=", 1) for item in items)
    phrases = []
    if "time" in parts:
        name, hours = parts["time"].split(" (")
        phrases.append(f"during the {name.lower()} ({hours.rstrip(')').replace('-', '–')})")
    if "day" in parts:
        phrases.append({"Weekday": "on weekdays", "Weekend": "at weekends", "Holiday": "on public holidays"}[parts["day"]])
    if "weather" in parts:
        phrases.append(f"in {parts['weather'].lower()} weather")
    if "temp" in parts:
        phrases.append("with " + parts["temp"].split(" (")[0].lower() + " temperatures")
    text = " ".join(phrases)
    return text[0].upper() + text[1:]


def plain_language(rule: pd.Series, n_hours: int) -> str:
    consequent = next(iter(rule["consequents"])).split("=", 1)[1]
    return (f"{describe_items(rule['antecedents'])}, congestion is {consequent} {rule['confidence']:.0%} of the time, "
            f"{rule['lift']:.1f}x the overall rate of {rule['consequent support']:.0%} "
            f"(based on {int(round(rule['support'] * n_hours)):,} hours).")


def run_association_rules(df: pd.DataFrame) -> pd.DataFrame:
    transactions = to_transactions(df)
    itemsets = apriori(transactions, min_support=MIN_SUPPORT, use_colnames=True, max_len=4)
    logger.info("Apriori found %d frequent itemsets (min_support=%.3f)", len(itemsets), MIN_SUPPORT)
    rules = association_rules(itemsets, num_itemsets=len(transactions), metric="confidence", min_threshold=MIN_CONFIDENCE)

    is_congestion = lambda items: all(i.startswith("congestion=") for i in items)  # noqa: E731
    no_congestion = lambda items: not any(i.startswith("congestion=") for i in items)  # noqa: E731
    rules = rules[rules["consequents"].apply(lambda c: len(c) == 1 and is_congestion(c))
                  & rules["antecedents"].apply(no_congestion)].copy()
    logger.info("%d rules predict a congestion level (confidence >= %.2f)", len(rules), MIN_CONFIDENCE)

    # Remove redundant rules: drop a rule if a simpler rule with the same consequent has >= confidence.
    rules = rules.sort_values(["lift", "confidence"], ascending=False)
    kept = []
    for _, rule in rules.iterrows():
        redundant = any(k["consequents"] == rule["consequents"] and k["antecedents"] < rule["antecedents"]
                        and k["confidence"] >= rule["confidence"] - 0.01 for k in kept)
        if not redundant:
            kept.append(rule)
    logger.debug("Non-redundant rules kept: %d of %d", len(kept), len(rules))
    rules = pd.DataFrame(kept)

    rules["consequent"] = rules["consequents"].apply(lambda c: next(iter(c)).split("=", 1)[1])
    rules["antecedent_text"] = rules["antecedents"].apply(lambda a: " & ".join(sorted(a)))
    rules["explanation"] = rules.apply(plain_language, axis=1, n_hours=len(transactions))

    top = rules.head(TOP_RULES)
    per_level = rules.sort_values("lift", ascending=False).groupby("consequent", observed=True).head(3)
    per_level = per_level.sort_values(["consequent", "lift"], key=lambda s: s.map(
        {c: i for i, c in enumerate(CONGESTION_ORDER)}) if s.name == "consequent" else -s)

    columns = ["antecedent_text", "consequent", "support", "confidence", "lift", "explanation"]
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    top[columns].to_csv(METRICS_DIR / "association_rules_top_lift.csv", index=False, float_format="%.4f")
    per_level[columns].to_csv(METRICS_DIR / "association_rules_by_congestion_level.csv", index=False, float_format="%.4f")
    for rank, (_, rule) in enumerate(top.iterrows(), start=1):
        logger.info("Rule %d (lift %.2f): %s", rank, rule["lift"], rule["explanation"])

    with mlflow.start_run(run_name="apriori_congestion_rules"):
        mlflow.set_tags({"task": "association_rules", "algorithm": "Apriori (mlxtend)"})
        mlflow.log_params({"min_support": MIN_SUPPORT, "min_confidence": MIN_CONFIDENCE, "max_len": 4,
                           "items": "time, day type, weather group, temperature band, congestion"})
        mlflow.log_metrics({"frequent_itemsets": len(itemsets), "congestion_rules": len(rules),
                            "top_rule_lift": float(top["lift"].iloc[0]), "top_rule_confidence": float(top["confidence"].iloc[0])})
        mlflow.log_artifact(str(METRICS_DIR / "association_rules_top_lift.csv"))

    _plot_rules(per_level)
    return rules


def _plot_rules(per_level: pd.DataFrame) -> None:
    data = per_level.iloc[::-1]
    colours = [P.ORDINAL_BLUES[CONGESTION_ORDER.index(c)] for c in data["consequent"]]
    fig, ax = P.plt.subplots(figsize=(11, 0.5 * len(data) + 1.6))
    ax.barh(range(len(data)), data["lift"], color=colours, edgecolor=P.SURFACE, linewidth=2, height=0.75)
    labels = [f"{a.replace('time=', '').replace('day=', '').replace('weather=', '').replace('temp=', '')}  →  {c}"
              for a, c in zip(data["antecedent_text"], data["consequent"])]
    ax.set_yticks(range(len(data)), labels, fontsize=8.5)
    for i, (lift, conf) in enumerate(zip(data["lift"], data["confidence"])):
        ax.text(lift + 0.03, i, f"lift {lift:.2f} · conf {conf:.0%}", va="center", fontsize=8, color=P.INK_SECONDARY)
    ax.axvline(1, color=P.INK_MUTED, linestyle="--", linewidth=1)
    ax.set_xlim(0, data["lift"].max() * 1.3)
    ax.set_xlabel("Lift (1 = no better than the overall rate)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Time of day and day type anchor every strong congestion rule; weather only refines them")
    P.subtitle(ax, "Top 3 non-redundant rules by lift for each congestion level (bar shade = predicted level)")
    P.save_figure(fig, "task2_association_rules.png", "task2_unsupervised")


def run(df: pd.DataFrame | None = None) -> dict[str, Any]:
    df = load_modelling_table() if df is None else df
    setup_experiment(EXPERIMENT_UNSUPERVISED)
    logger.info("Task 2: unsupervised learning on %d hourly records", len(df))
    return {"kmeans": run_kmeans(df), "rules": run_association_rules(df)}


if __name__ == "__main__":
    from logging_config import configure_logging

    configure_logging("unsupervised")
    run()
