"""
monitoring.py - Tasks 6.4 and 6.5: model monitoring simulation with PASS / ALERT status.

Simulation design
-----------------
* The champion regressor was trained on data up to Dec 2017. Jan-Sep 2018 (the untouched
  test period) is replayed month by month as "production" batches.
* Feature distribution drift: Population Stability Index (PSI) of each monitored feature
  against the SAME calendar month of the reference year (2017, the last pre-deployment year).
  - A seasonal reference avoids false alarms such as "January is colder than the all-year average".
  - A recent reference matters: a first version compared against all of 2012-2017 and raised
    cloud-cover alerts in every month, because the weather feed started quantising clouds_all to
    {1, 20, 40, 75, 90} in 2017. That upstream change is documented, not re-alerted monthly.
  - Low-cardinality numeric features (e.g. clouds_all) are compared as categories.
* Prediction error drift: batch MAE compared with the model's validation MAE, plus bias.
* Two clearly-labelled synthetic incidents are injected to prove the alerts fire:
    - Sensor under-count: traffic counts from a real batch reduced by 35%.
    - Weather feed unit fault: a real batch's temperatures delivered in Fahrenheit.

Rules (tiered: alert on model impact or broken data, not on weather simply being unusual)
-----
ALERT  error drift       : batch MAE > 1.30 x validation MAE, or |mean error| > 10% of mean volume
ALERT  data integrity    : > 1% of a feature's values fall outside the range seen in training (<= 2017)
WATCH  feature drift     : PSI > 0.25 vs same month of 2017 - logged at WARNING and shown on the
                           dashboard, but weather legitimately varies year to year (March-May 2018 were
                           record cold/warm) so drift alone does not change the status.
A batch is PASS only if no ALERT rule fires. Alerts and drift warnings are logged at WARNING.

Outputs: monitoring/monitoring_report.json, monitoring/dashboard.html,
figures/task6_monitoring/*.png, and one MLflow run per batch (experiment traffic-monitoring).
"""

from __future__ import annotations

import base64
import html
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd

import plotting as P
from config import EXPERIMENT_MONITORING, FIGURES_DIR, MODEL_FEATURES, MODELS_DIR, MONITORING_DIR, PROJECT_ROOT
from data_prep import load_modelling_table
from features import make_feature_frame
from mlflow_utils import setup_experiment

logger = logging.getLogger(__name__)

NUMERIC_DRIFT_FEATURES = ["temp_c", "traffic_volume", "predicted_volume"]
CATEGORICAL_DRIFT_FEATURES = ["weather_main", "clouds_all"]
REFERENCE_SPLIT = "validation"   # 2017: most recent full year before the 2018 "production" period
PSI_WATCH, PSI_ALERT = 0.10, 0.25   # PSI_ALERT is the drift-warning level (see tiered rules above)
MAE_RATIO_ALERT = 1.30
BIAS_ALERT_SHARE = 0.10
RANGE_FEATURES = ["temp_c", "clouds_all", "rain_1h", "snow_1h"]
OUT_OF_RANGE_ALERT_SHARE = 0.01
N_BINS = 10


# ---------------------------------------------------------------------------
# Drift statistics
# ---------------------------------------------------------------------------
def psi_numeric(reference: pd.Series, current: pd.Series, bins: int = N_BINS) -> float:
    """PSI with quantile bins from the reference (duplicate edges merged, open-ended outer bins)."""
    edges = np.unique(np.quantile(reference.dropna(), np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.unique(np.concatenate([edges, [reference.min() - 1, reference.max() + 1]]))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(reference, edges)[0] / len(reference)
    cur_share = np.histogram(current, edges)[0] / len(current)
    return _psi(ref_share, cur_share)


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    categories = sorted(set(reference) | set(current))
    ref_share = reference.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    cur_share = current.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    return _psi(ref_share, cur_share)


def _psi(ref_share: np.ndarray, cur_share: np.ndarray, eps: float = 1e-4) -> float:
    ref_share = np.clip(ref_share, eps, None)
    cur_share = np.clip(cur_share, eps, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------
def evaluate_batch(name: str, batch: pd.DataFrame, reference: pd.DataFrame, baseline_mae: float,
                   training_ranges: dict[str, tuple[float, float]], kind: str = "real") -> dict[str, Any]:
    month = int(batch["date_time"].dt.month.mode().iloc[0])
    ref_month = reference[reference["date_time"].dt.month == month]
    drift = {f: psi_numeric(ref_month[f], batch[f]) for f in NUMERIC_DRIFT_FEATURES}
    drift.update({f: psi_categorical(ref_month[f], batch[f]) for f in CATEGORICAL_DRIFT_FEATURES})
    out_of_range = {f: float(((batch[f] < lo) | (batch[f] > hi)).mean()) for f, (lo, hi) in training_ranges.items()}

    error = batch["predicted_volume"] - batch["traffic_volume"]
    mae = float(error.abs().mean())
    bias = float(error.mean())
    mae_ratio = mae / baseline_mae
    bias_share = abs(bias) / float(batch["traffic_volume"].mean())

    alerts = []
    if mae_ratio > MAE_RATIO_ALERT:
        alerts.append(f"Error drift: MAE {mae:,.0f} is {mae_ratio:.2f}x the validation MAE ({baseline_mae:,.0f})")
    if bias_share > BIAS_ALERT_SHARE:
        alerts.append(f"Error drift: mean error {bias:+,.0f} is {bias_share:.0%} of mean volume")
    alerts += [f"Data integrity: {share:.0%} of {f} outside training range {training_ranges[f][0]:g} to "
               f"{training_ranges[f][1]:g}" for f, share in out_of_range.items() if share > OUT_OF_RANGE_ALERT_SHARE]
    drift_warnings = [f"{f} PSI={v:.2f}" for f, v in drift.items() if v > PSI_ALERT]
    status = "ALERT" if alerts else "PASS"

    result = {"batch": name, "kind": kind, "rows": len(batch), "reference_month": month, "status": status,
              "mae": mae, "mae_ratio": mae_ratio, "bias": bias, "psi": drift, "out_of_range_share": out_of_range,
              "alerts": alerts, "drift_warnings": drift_warnings}
    if status == "ALERT":
        logger.warning("MONITORING ALERT [%s]: %s", name, " | ".join(alerts))
    else:
        logger.info("Monitoring PASS [%s]: MAE %.0f (%.2fx validation), no integrity issues", name, mae, mae_ratio)
    if drift_warnings:
        logger.warning("Feature drift noted [%s] (watch, no status change): %s", name, ", ".join(drift_warnings))
    return result


def _predict(model: Any, frame: pd.DataFrame) -> np.ndarray:
    return np.clip(model.predict(frame[MODEL_FEATURES].astype(float)), 0, None)


def simulate_incidents(test: pd.DataFrame, model: Any) -> list[tuple[str, pd.DataFrame]]:
    """Two synthetic incidents built from real 2018 batches (clearly labelled as simulations)."""
    sensor = test[test["date_time"].dt.month == 9].copy()
    sensor["traffic_volume"] = (sensor["traffic_volume"] * 0.65).round()
    logger.warning("Simulated incident injected: sensor under-count (Sep 2018 volumes x0.65)")

    units = test[test["date_time"].dt.month == 7].copy()
    units["temp_c"] = units["temp_c"] * 9 / 5 + 32
    rebuilt = make_feature_frame(units)
    units[MODEL_FEATURES] = rebuilt[MODEL_FEATURES]
    units["predicted_volume"] = _predict(model, units)
    logger.warning("Simulated incident injected: weather feed unit fault (Jul 2018 temperatures in Fahrenheit)")
    return [("SIM: sensor under-count (Sep 2018)", sensor), ("SIM: temperature feed in °F (Jul 2018)", units)]


def run(df: pd.DataFrame | None = None) -> dict[str, Any]:
    df = load_modelling_table() if df is None else df
    model = joblib.load(MODELS_DIR / "traffic_volume_regressor.joblib")
    meta = json.loads((MODELS_DIR / "traffic_volume_regressor.json").read_text(encoding="utf-8"))
    baseline_mae = float(meta["validation"]["mae"])

    reference = df[df["split"] == REFERENCE_SPLIT].copy()
    reference["predicted_volume"] = _predict(model, reference)
    test = df[df["split"] == "test"].copy()
    test["predicted_volume"] = _predict(model, test)
    logger.info("Monitoring %s v%s: reference=%d hours (2017, same calendar month per batch), production=%d hours "
                "(2018), baseline validation MAE=%.0f", meta["candidate"], meta["version"], len(reference), len(test),
                baseline_mae)

    training = df[df["split"] != "test"]
    training_ranges = {f: (float(training[f].min()), float(training[f].max())) for f in RANGE_FEATURES}
    logger.debug("Training ranges for integrity checks: %s", training_ranges)

    batches = [(str(period), group) for period, group in test.groupby(test["date_time"].dt.to_period("M"))]
    results = [evaluate_batch(name, batch, reference, baseline_mae, training_ranges) for name, batch in batches]
    results += [evaluate_batch(name, batch, reference, baseline_mae, training_ranges, kind="simulated")
                for name, batch in simulate_incidents(test, model)]

    real = [r for r in results if r["kind"] == "real"]
    latest = real[-1]
    system_status = latest["status"]   # current state = most recent production batch
    period_alerts = [r["batch"] for r in real if r["status"] == "ALERT"]
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {"name": "traffic-volume-regressor", "candidate": meta["candidate"], "version": meta["version"]},
        "latest_batch": latest["batch"],
        "alerts_in_period": period_alerts,
        "simulated_incidents_detected": sum(r["status"] == "ALERT" for r in results if r["kind"] == "simulated"),
        "rules": {"alert_mae_ratio": MAE_RATIO_ALERT, "alert_bias_share": BIAS_ALERT_SHARE,
                  "alert_out_of_range_share": OUT_OF_RANGE_ALERT_SHARE, "drift_warning_psi": PSI_ALERT,
                  "baseline_validation_mae": baseline_mae, "training_ranges": training_ranges,
                  "drift_reference": "same calendar month of 2017"},
        "system_status": system_status,
        "batches": results,
    }
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    (MONITORING_DIR / "monitoring_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved monitoring report to monitoring/monitoring_report.json")
    if system_status == "ALERT":
        logger.warning("SYSTEM STATUS: ALERT - latest batch %s requires investigation", latest["batch"])
    else:
        logger.info("SYSTEM STATUS: PASS - latest batch %s is normal", latest["batch"])
    if period_alerts:
        logger.warning("%d of %d production batches raised alerts during the period: %s", len(period_alerts), len(real),
                       ", ".join(period_alerts))

    setup_experiment(EXPERIMENT_MONITORING)
    for r in results:
        with mlflow.start_run(run_name=f"monitor_{r['batch']}"):
            mlflow.set_tags({"status": r["status"], "batch_kind": r["kind"], "model_version": meta["version"]})
            mlflow.log_metrics({"mae": r["mae"], "mae_ratio": r["mae_ratio"], "bias": r["bias"],
                                **{f"psi_{k}": v for k, v in r["psi"].items()},
                                **{f"out_of_range_{k}": v for k, v in r["out_of_range_share"].items()}})

    charts = _plot(results, baseline_mae)
    _write_dashboard(report, charts)
    return report


# ---------------------------------------------------------------------------
# Figures and dashboard
# ---------------------------------------------------------------------------
def _plot(results: list[dict[str, Any]], baseline_mae: float) -> list[str]:
    names = [r["batch"] for r in results]
    fig, ax = P.plt.subplots(figsize=(11, 4.6))
    colours = [P.SERIES_2 if r["status"] == "ALERT" else P.SERIES_1 for r in results]
    ax.bar(range(len(results)), [r["mae"] for r in results], color=colours, edgecolor=P.SURFACE, linewidth=2, width=0.75)
    ax.axhline(baseline_mae, color=P.INK_MUTED, linestyle="--", linewidth=1)
    ax.axhline(baseline_mae * MAE_RATIO_ALERT, color=P.INK_PRIMARY, linestyle="--", linewidth=1)
    ax.text(len(results) - 0.5, baseline_mae, " validation MAE", va="bottom", ha="right", fontsize=8.5, color=P.INK_SECONDARY)
    ax.text(len(results) - 0.5, baseline_mae * MAE_RATIO_ALERT, f" alert threshold ({MAE_RATIO_ALERT}x)", va="bottom",
            ha="right", fontsize=8.5, color=P.INK_PRIMARY)
    for i, r in enumerate(results):
        ax.text(i, r["mae"], f"{r['status']}\n{r['mae']:,.0f}", ha="center", va="bottom", fontsize=8,
                color=P.INK_PRIMARY if r["status"] == "ALERT" else P.INK_SECONDARY)
    ax.set_xticks(range(len(results)), [n.replace("SIM: ", "SIM:\n").replace(" (", "\n(") for n in names], fontsize=8)
    ax.set_ylim(0, max(r["mae"] for r in results) * 1.25)
    ax.set_ylabel("Batch MAE (vehicles/hour)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Error drift flags the Jan and Apr 2018 winter anomalies and the injected sensor fault")
    P.subtitle(ax, "Monthly 2018 batches and simulated incidents (orange = ALERT)")
    error_path = P.save_figure(fig, "task6_monitoring_error_drift.png", "task6_monitoring")

    features = NUMERIC_DRIFT_FEATURES + CATEGORICAL_DRIFT_FEATURES
    matrix = np.array([[r["psi"][f] for f in features] for r in results])
    fig, ax = P.plt.subplots(figsize=(11, 0.45 * len(results) + 1.8))
    cmap = P.plt.matplotlib.colors.ListedColormap(["#cde2fb", "#6da7ec", "#eb6834"])
    norm = P.plt.matplotlib.colors.BoundaryNorm([0, PSI_WATCH, PSI_ALERT, 100], cmap.N)
    ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8.5,
                color="white" if value > PSI_ALERT else P.INK_PRIMARY)
    ax.set_xticks(range(len(features)), features)
    ax.set_yticks(range(len(results)), names, fontsize=8.5)
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Feature drift (PSI vs the same calendar month of 2017)")
    P.subtitle(ax, f"Light < {PSI_WATCH} stable · mid {PSI_WATCH}-{PSI_ALERT} minor · orange > {PSI_ALERT} drift warning (watch)")
    psi_path = P.save_figure(fig, "task6_monitoring_feature_drift_psi.png", "task6_monitoring")
    return [str(error_path), str(psi_path)]


def _write_dashboard(report: dict[str, Any], charts: list[str]) -> None:
    def badge(status: str) -> str:
        return f'<span class="badge {status.lower()}">{"✔ PASS" if status == "PASS" else "▲ ALERT"}</span>'

    rows = []
    for r in report["batches"]:
        detail = "<br>".join(html.escape(a) for a in r["alerts"]) or "Error and data-integrity checks passed"
        if r["drift_warnings"]:
            detail += "<br><span class='watch'>Drift watch: " + html.escape(", ".join(r["drift_warnings"])) + "</span>"
        rows.append(f"<tr class='{r['kind']}'><td>{html.escape(r['batch'])}</td><td>{r['kind']}</td><td>{badge(r['status'])}</td>"
                    f"<td class='num'>{r['rows']:,}</td><td class='num'>{r['mae']:,.0f}</td><td class='num'>{r['mae_ratio']:.2f}</td>"
                    f"<td class='num'>{r['bias']:+,.0f}</td><td class='num'>{max(r['psi'].values()):.2f}</td><td>{detail}</td></tr>")
    images = "".join(
        f'<figure><img alt="monitoring chart" src="data:image/png;base64,{base64.b64encode(open(p, "rb").read()).decode()}"></figure>'
        for p in charts)
    real = [r for r in report["batches"] if r["kind"] == "real"]
    simulated = [r for r in report["batches"] if r["kind"] == "simulated"]
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Traffic Model Monitoring</title>
<style>
:root {{ --surface:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --line:#e1e0d9;
        --good:#0ca30c; --critical:#d03b3b; }}
body {{ margin:0; background:var(--page); color:var(--ink); font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1100px; margin:0 auto; padding:24px 16px 48px; }}
h1 {{ font-size:22px; margin:0 0 4px; }} p.sub {{ color:var(--ink2); margin:0 0 20px; }}
.banner {{ display:flex; gap:16px; align-items:center; padding:16px 20px; border-radius:10px; background:var(--surface);
          border:1px solid var(--line); margin-bottom:20px; flex-wrap:wrap; }}
.banner .big {{ font-size:28px; font-weight:700; }} .banner.pass .big {{ color:var(--good); }} .banner.alert .big {{ color:var(--critical); }}
.badge {{ display:inline-block; padding:2px 8px; border-radius:999px; font-weight:600; font-size:12px; white-space:nowrap; }}
.badge.pass {{ background:#e3f4e3; color:#006300; }} .badge.alert {{ background:#fbe3e3; color:#9b1c1c; }}
.table-wrap {{ overflow-x:auto; background:var(--surface); border:1px solid var(--line); border-radius:10px; }}
table {{ border-collapse:collapse; width:100%; min-width:820px; }}
th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); vertical-align:top; }}
th {{ color:var(--ink2); font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.03em; }}
td.num {{ text-align:right; font-variant-numeric:tabular-nums; }} tr.simulated td {{ background:#faf6ee; }}
figure {{ margin:20px 0 0; background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:12px; }}
img {{ max-width:100%; height:auto; display:block; }} .rules {{ color:var(--ink2); font-size:13px; margin-top:16px; }}
.watch {{ color:#8a5a00; }}
</style></head><body><main>
<h1>Traffic model monitoring</h1>
<p class="sub">Model <b>{html.escape(report['model']['name'])}</b> ({html.escape(report['model']['candidate'])}, version {report['model']['version']}) ·
generated {report['generated_utc']}</p>
<div class="banner {report['system_status'].lower()}"><div class="big">{"PASS / Normal" if report['system_status'] == "PASS" else "ALERT / Requires investigation"}</div>
<div>Current status from latest batch <b>{html.escape(report['latest_batch'])}</b> ·
{sum(r['status'] == 'PASS' for r in real)} of {len(real)} production batches passed
{("(alerts: " + html.escape(", ".join(report['alerts_in_period'])) + ")") if report['alerts_in_period'] else ""} ·
{sum(r['status'] == 'ALERT' for r in simulated)} of {len(simulated)} simulated incidents detected</div></div>
<div class="table-wrap"><table><thead><tr><th>Batch</th><th>Type</th><th>Status</th><th>Hours</th><th>MAE</th><th>MAE ratio</th>
<th>Bias</th><th>Max PSI</th><th>Findings</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="rules">ALERT when MAE ratio &gt; {MAE_RATIO_ALERT}, |bias| &gt; {BIAS_ALERT_SHARE:.0%} of mean volume, or &gt; {OUT_OF_RANGE_ALERT_SHARE:.0%} of a feature's
values fall outside the training range. Feature drift (PSI &gt; {PSI_ALERT} against the same calendar month of 2017) is shown as a watch note:
weather legitimately varies year to year, so drift alone does not require investigation unless it persists or errors rise.
Rows shaded beige are synthetic incidents injected to test alerting.</p>
{images}
</main></body></html>"""
    path = MONITORING_DIR / "dashboard.html"
    path.write_text(page, encoding="utf-8")
    logger.info("Wrote monitoring dashboard to %s", path.relative_to(PROJECT_ROOT).as_posix())


def main() -> int:
    report = run()
    print(f"\nSYSTEM STATUS: {report['system_status']} (latest batch {report['latest_batch']}; "
          f"alerts during period: {', '.join(report['alerts_in_period']) or 'none'})\n")
    print(f"{'Batch':42} {'Status':7} {'MAE':>6} {'Ratio':>6} {'Max PSI':>8}")
    for r in report["batches"]:
        print(f"{r['batch']:42} {r['status']:7} {r['mae']:6,.0f} {r['mae_ratio']:6.2f} {max(r['psi'].values()):8.2f}")
    print(f"\nDashboard: {(MONITORING_DIR / 'dashboard.html').relative_to(PROJECT_ROOT).as_posix()}")
    return 0 if report["system_status"] == "PASS" else 3


if __name__ == "__main__":
    from logging_config import configure_logging

    configure_logging("monitoring")
    sys.exit(main())
