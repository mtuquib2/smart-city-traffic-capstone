"""
statistical_analysis.py - Part 1 statistics, reproduced from the SQLite database.

Re-computes the statistics from sql/part1_queries.sql in Python (so the SQL results can be
verified) and extends them with formal hypothesis tests:

1. Descriptive statistics for traffic volume and temperature
2. Pearson and Spearman correlation: temperature vs traffic volume
3. Probability analysis of congestion (> 5,500 vehicles/hour) and weather:
   joint, conditional and independence check, odds and odds ratio (clear vs clouds)
4. Chi-square test of independence: weather_main x congestion
5. Welch t-test: weekday vs weekend hourly volume

Run from part1_data_analytics/:
    python statistics/statistical_analysis.py
Outputs: statistics/statistical_results.md and statistics/statistical_results.json
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

PART1_ROOT = Path(__file__).resolve().parents[1]
DATABASE = PART1_ROOT / "sql" / "metro_traffic.db"
OUTPUT_DIR = Path(__file__).resolve().parent
CONGESTION_THRESHOLD = 5500
HIGH_TEMP_K = 292


def load_data() -> pd.DataFrame:
    try:
        with sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True) as connection:
            df = pd.read_sql_query("SELECT * FROM Metro_Interstate_Traffic_Volume", connection)
    except sqlite3.Error as exc:
        raise RuntimeError(f"Could not read {DATABASE.name}: {exc}") from exc
    df["date_time"] = pd.to_datetime(df["date_time"])
    logger.info("Loaded %d rows, %d columns from %s", *df.shape, DATABASE.name)
    return df


def descriptive(df: pd.DataFrame) -> dict:
    summary = df[["traffic_volume", "temp"]].describe().T
    summary["skewness"] = df[["traffic_volume", "temp"]].skew()
    logger.info("Descriptive statistics computed (mean volume %.0f, median %.0f)",
                summary.loc["traffic_volume", "mean"], summary.loc["traffic_volume", "50%"])
    return summary.round(3).to_dict(orient="index")


def correlation(df: pd.DataFrame) -> dict:
    # 0 K readings are sensor faults (see Part 2); report the raw SQL figure and the cleaned one.
    raw_r, raw_p = stats.pearsonr(df["temp"], df["traffic_volume"])
    valid = df[df["temp"] > 200]
    if len(valid) < len(df):
        logger.warning("Excluded %d rows with impossible temperatures (<= 200 K) from the cleaned correlation",
                       len(df) - len(valid))
    r, p = stats.pearsonr(valid["temp"], valid["traffic_volume"])
    rho, rho_p = stats.spearmanr(valid["temp"], valid["traffic_volume"])
    logger.info("Pearson r raw=%.4f cleaned=%.4f; Spearman rho=%.4f", raw_r, r, rho)
    return {"pearson_r_raw": raw_r, "pearson_p_raw": raw_p, "pearson_r_cleaned": r, "pearson_p_cleaned": p,
            "spearman_rho_cleaned": rho, "spearman_p_cleaned": rho_p, "r_squared_cleaned": r ** 2}


def probability(df: pd.DataFrame) -> dict:
    congestion = df["traffic_volume"] > CONGESTION_THRESHOLD
    clear = df["weather_main"] == "Clear"
    clouds = df["weather_main"] == "Clouds"
    p_cong, p_clear = congestion.mean(), clear.mean()
    p_joint = (congestion & clear).mean()

    def odds(mask: pd.Series) -> float:
        p = congestion[mask].mean()
        return p / (1 - p)

    result = {
        "P_congestion": p_cong,
        "P_clear": p_clear,
        "P_congestion_and_clear": p_joint,
        "P_clear_given_congestion": p_joint / p_cong,
        "P_high_temp_given_congestion": (congestion & (df["temp"] > HIGH_TEMP_K)).mean() / p_cong,
        "P_congestion_times_P_clear": p_cong * p_clear,
        "odds_congestion_clear": odds(clear),
        "odds_congestion_clouds": odds(clouds),
    }
    result["odds_ratio_clear_vs_clouds"] = result["odds_congestion_clear"] / result["odds_congestion_clouds"]
    logger.info("P(congestion)=%.4f, odds ratio clear vs clouds=%.4f", p_cong, result["odds_ratio_clear_vs_clouds"])
    return result


def chi_square(df: pd.DataFrame) -> dict:
    table = pd.crosstab(df["weather_main"], df["traffic_volume"] > CONGESTION_THRESHOLD)
    # Categories with expected counts < 5 violate the test's assumption; pool them.
    expected_min = stats.contingency.expected_freq(table).min(axis=1)
    rare = table.index[expected_min < 5]
    if len(rare):
        logger.warning("Pooled %d rare weather categories into 'Other' for the chi-square test: %s", len(rare), list(rare))
        table.loc["Other"] = table.loc[rare].sum()
        table = table.drop(index=rare)
    chi2, p, dof, _ = stats.chi2_contingency(table)
    cramers_v = np.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape) - 1)))
    logger.info("Chi-square weather x congestion: chi2=%.1f dof=%d p=%.3g Cramer's V=%.3f", chi2, dof, p, cramers_v)
    return {"chi2": chi2, "dof": int(dof), "p_value": p, "cramers_v": cramers_v}


def weekday_weekend(df: pd.DataFrame) -> dict:
    weekend = df["date_time"].dt.dayofweek >= 5
    a, b = df.loc[~weekend, "traffic_volume"], df.loc[weekend, "traffic_volume"]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    pooled_sd = np.sqrt((a.var() + b.var()) / 2)
    logger.info("Welch t-test weekday vs weekend: t=%.1f p=%.3g", t, p)
    return {"weekday_mean": a.mean(), "weekend_mean": b.mean(), "welch_t": t, "p_value": p,
            "cohens_d": (a.mean() - b.mean()) / pooled_sd}


def write_report(results: dict) -> None:
    c, pr, chi, tt = results["correlation"], results["probability"], results["chi_square"], results["weekday_weekend"]
    d = results["descriptive"]["traffic_volume"]
    lines = [
        "# Part 1 – Statistical Analysis Results",
        "",
        "Generated by `statistics/statistical_analysis.py` from `sql/metro_traffic.db` (raw data, 48,204 rows).",
        "Sections 1–3 reproduce the SQL analyses in `sql/part1_queries.sql`; sections 4–5 add formal hypothesis tests.",
        "",
        "## 1. Descriptive statistics – traffic volume",
        "",
        f"Mean {d['mean']:,.0f} · median {d['50%']:,.0f} · std {d['std']:,.0f} · min {d['min']:,.0f} · max {d['max']:,.0f} · "
        f"skewness {d['skewness']:.2f}. The distribution is nearly symmetric overall but multi-modal (quiet nights vs "
        "busy daytime hours), so the mean alone is a poor summary.",
        "",
        "## 2. Correlation – temperature vs traffic volume",
        "",
        "| Measure | Value | p-value |",
        "|---|---|---|",
        f"| Pearson r (raw, as in SQL) | {c['pearson_r_raw']:.4f} | {c['pearson_p_raw']:.2e} |",
        f"| Pearson r (0 K faults removed) | {c['pearson_r_cleaned']:.4f} | {c['pearson_p_cleaned']:.2e} |",
        f"| Spearman ρ (cleaned) | {c['spearman_rho_cleaned']:.4f} | {c['spearman_p_cleaned']:.2e} |",
        "",
        f"A **weak positive** relationship: statistically significant because n is large, but temperature explains only "
        f"{c['r_squared_cleaned']:.1%} of the variance in volume. Correlation is not causation; time of day and season "
        "drive both variables.",
        "",
        f"## 3. Probability analysis – congestion (> {CONGESTION_THRESHOLD:,} vehicles/hour) and weather",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| P(Congestion) | {pr['P_congestion']:.4f} |",
        f"| P(Clear) | {pr['P_clear']:.4f} |",
        f"| P(Congestion ∩ Clear) | {pr['P_congestion_and_clear']:.4f} |",
        f"| P(Congestion) × P(Clear) | {pr['P_congestion_times_P_clear']:.4f} |",
        f"| P(Clear \\| Congestion) | {pr['P_clear_given_congestion']:.4f} |",
        f"| P(High temp > {HIGH_TEMP_K} K \\| Congestion) | {pr['P_high_temp_given_congestion']:.4f} |",
        f"| Odds of congestion, clear | {pr['odds_congestion_clear']:.4f} |",
        f"| Odds of congestion, clouds | {pr['odds_congestion_clouds']:.4f} |",
        f"| **Odds ratio clear vs clouds** | **{pr['odds_ratio_clear_vs_clouds']:.4f}** |",
        "",
        "The joint probability is close to the product of the marginals, so congestion and clear weather are roughly "
        f"independent. The odds of congestion are about {1 - pr['odds_ratio_clear_vs_clouds']:.0%} lower in clear than in "
        "cloudy weather.",
        "",
        "## 4. Chi-square test of independence – weather × congestion",
        "",
        f"χ² = {chi['chi2']:,.1f}, df = {chi['dof']}, p = {chi['p_value']:.2e}, **Cramér's V = {chi['cramers_v']:.3f}**.",
        "",
        "The association is statistically significant, but its strength is very small (V < 0.1). Weather is not a "
        "practically important driver of congestion on its own.",
        "",
        "## 5. Welch t-test – weekday vs weekend hourly volume",
        "",
        f"Weekday mean {tt['weekday_mean']:,.0f} vs weekend mean {tt['weekend_mean']:,.0f}; t = {tt['welch_t']:.1f}, "
        f"p {'< 1e-300' if tt['p_value'] == 0 else '= ' + format(tt['p_value'], '.2e')}, **Cohen's d = {tt['cohens_d']:.2f}**.",
        "",
        "Weekdays carry significantly more traffic, with a medium effect size. Pooling all hours dilutes the much "
        "larger peak-hour difference, which the Power BI dashboard shows directly.",
        "",
    ]
    (OUTPUT_DIR / "statistical_results.md").write_text("\n".join(lines), encoding="utf-8")
    (OUTPUT_DIR / "statistical_results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    logger.info("Wrote statistics/statistical_results.md and statistical_results.json")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(module)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
    try:
        df = load_data()
        results = {"descriptive": descriptive(df), "correlation": correlation(df), "probability": probability(df),
                   "chi_square": chi_square(df), "weekday_weekend": weekday_weekend(df)}
        write_report(results)
    except RuntimeError:
        logger.error("Statistical analysis failed", exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
