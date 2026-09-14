"""
supervised_models.py - Task 1: classification (proxy accident risk) and regression (traffic volume).

Both tasks share the common feature set in config.MODEL_FEATURES (time features, cyclical
hour/day/month encodings, holiday flag, weather encodings and indicators).

Protocol (identical for every candidate)
----------------------------------------
1. Fit on TRAIN (Oct 2012 - Dec 2016) and score on VALIDATION (2017) -> model selection.
2. Refit on TRAIN + VALIDATION and score once on TEST (Jan - Sep 2018) -> reported performance.
3. Log parameters, metrics, timing and the refitted model to MLflow and register it as a
   new model version. The best validation candidate receives the 'champion' alias and is
   exported to models/ for the API, recommender and monitoring.

Candidates
----------
Classification: logistic regression (baseline), random forest, histogram gradient boosting (x2 configs)
Regression    : ridge regression (baseline), random forest, histogram gradient boosting (x2 configs)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import plotting as P
from config import (
    EXPERIMENT_CLASSIFICATION,
    EXPERIMENT_REGRESSION,
    METRICS_DIR,
    MODEL_FEATURES,
    MODELS_DIR,
    RANDOM_SEED,
    REGISTERED_CLASSIFIER,
    REGISTERED_REGRESSOR,
)
from data_prep import load_modelling_table
from mlflow_utils import log_sklearn_model, set_champion, setup_experiment, update_local_registry

logger = logging.getLogger(__name__)

CLASSIFICATION_TARGET = "high_risk"
REGRESSION_TARGET = "traffic_volume"


@dataclass
class Candidate:
    name: str
    algorithm: str
    estimator: Any
    params: dict[str, Any] = field(default_factory=dict)


def classification_candidates() -> list[Candidate]:
    return [
        Candidate("logistic_regression_v1", "LogisticRegression (baseline)",
                  make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", C=1.0, max_iter=3000)),
                  {"C": 1.0, "class_weight": "balanced", "scaler": "StandardScaler"}),
        Candidate("random_forest_v1", "RandomForestClassifier",
                  RandomForestClassifier(n_estimators=200, max_leaf_nodes=512, min_samples_leaf=5,
                                         class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_SEED),
                  {"n_estimators": 200, "max_leaf_nodes": 512, "min_samples_leaf": 5, "class_weight": "balanced_subsample"}),
        Candidate("hist_gradient_boosting_v1", "HistGradientBoostingClassifier",
                  HistGradientBoostingClassifier(learning_rate=0.1, max_iter=200, max_leaf_nodes=31,
                                                 class_weight="balanced", random_state=RANDOM_SEED),
                  {"learning_rate": 0.1, "max_iter": 200, "max_leaf_nodes": 31, "class_weight": "balanced"}),
        Candidate("hist_gradient_boosting_v2", "HistGradientBoostingClassifier (tuned)",
                  HistGradientBoostingClassifier(learning_rate=0.05, max_iter=500, max_leaf_nodes=15, min_samples_leaf=40,
                                                 l2_regularization=1.0, class_weight="balanced", random_state=RANDOM_SEED),
                  {"learning_rate": 0.05, "max_iter": 500, "max_leaf_nodes": 15, "min_samples_leaf": 40,
                   "l2_regularization": 1.0, "class_weight": "balanced"}),
    ]


def regression_candidates() -> list[Candidate]:
    return [
        Candidate("ridge_regression_v1", "Ridge (baseline)",
                  make_pipeline(StandardScaler(), Ridge(alpha=1.0)), {"alpha": 1.0, "scaler": "StandardScaler"}),
        Candidate("random_forest_v1", "RandomForestRegressor",
                  RandomForestRegressor(n_estimators=200, max_leaf_nodes=512, min_samples_leaf=5, n_jobs=-1,
                                        random_state=RANDOM_SEED),
                  {"n_estimators": 200, "max_leaf_nodes": 512, "min_samples_leaf": 5}),
        Candidate("hist_gradient_boosting_v1", "HistGradientBoostingRegressor",
                  HistGradientBoostingRegressor(learning_rate=0.1, max_iter=200, max_leaf_nodes=31, random_state=RANDOM_SEED),
                  {"learning_rate": 0.1, "max_iter": 200, "max_leaf_nodes": 31}),
        Candidate("hist_gradient_boosting_v2", "HistGradientBoostingRegressor (tuned)",
                  HistGradientBoostingRegressor(learning_rate=0.05, max_iter=600, max_leaf_nodes=63, min_samples_leaf=30,
                                                l2_regularization=1.0, random_state=RANDOM_SEED),
                  {"learning_rate": 0.05, "max_iter": 600, "max_leaf_nodes": 63, "min_samples_leaf": 30,
                   "l2_regularization": 1.0}),
    ]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def classification_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
    }


def regression_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, pred),
        "rmse": root_mean_squared_error(y_true, pred),
        "r2": r2_score(y_true, pred),
    }


def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def split_xy(df: pd.DataFrame, target: str) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    parts = {}
    for name in ("train", "validation", "test"):
        subset = df[df["split"] == name]
        parts[name] = (subset[MODEL_FEATURES].astype(float), subset[target])
    parts["train_validation"] = (
        pd.concat([parts["train"][0], parts["validation"][0]]),
        pd.concat([parts["train"][1], parts["validation"][1]]),
    )
    return parts


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------
def _fit_timed(estimator: Any, X: pd.DataFrame, y: pd.Series) -> tuple[Any, float]:
    start = time.perf_counter()
    estimator.fit(X, y)
    return estimator, time.perf_counter() - start


def run_classification(df: pd.DataFrame) -> dict[str, Any]:
    setup_experiment(EXPERIMENT_CLASSIFICATION)
    data = split_xy(df, CLASSIFICATION_TARGET)
    X_train, y_train = data["train"]
    X_val, y_val = data["validation"]
    X_trval, y_trval = data["train_validation"]
    X_test, y_test = data["test"]
    logger.info("Classification data: train=%d (%.2f%% positive), validation=%d (%.2f%%), test=%d (%.2f%%)",
                len(y_train), 100 * y_train.mean(), len(y_val), 100 * y_val.mean(), len(y_test), 100 * y_test.mean())

    results, entries = [], []
    for cand in classification_candidates():
        with mlflow.start_run(run_name=cand.name):
            model, _ = _fit_timed(clone(cand.estimator), X_train, y_train)
            val_proba = model.predict_proba(X_val)[:, 1]
            val_metrics = classification_metrics(y_val, val_proba)
            tuned_threshold = best_f1_threshold(y_val, val_proba)

            final_model, train_seconds = _fit_timed(clone(cand.estimator), X_trval, y_trval)
            test_proba = final_model.predict_proba(X_test)[:, 1]
            test_metrics = classification_metrics(y_test, test_proba)
            test_metrics_tuned = classification_metrics(y_test, test_proba, tuned_threshold)

            mlflow.set_tags({"task": "classification", "algorithm": cand.algorithm, "label": "PROXY high_risk",
                             "features": "common MODEL_FEATURES", "split": "time-based"})
            mlflow.log_params({**cand.params, "n_features": len(MODEL_FEATURES), "decision_threshold_tuned": round(tuned_threshold, 4)})
            mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            mlflow.log_metrics({f"test_tuned_{k}": v for k, v in test_metrics_tuned.items()})
            mlflow.log_metric("train_seconds", train_seconds)
            version = log_sklearn_model(final_model, X_test, REGISTERED_CLASSIFIER)
            run_id = mlflow.active_run().info.run_id

        logger.info("%-28s val PR-AUC=%.3f F1=%.3f | test ROC-AUC=%.3f PR-AUC=%.3f F1=%.3f (%.1fs)", cand.name,
                    val_metrics["pr_auc"], val_metrics["f1"], test_metrics["roc_auc"], test_metrics["pr_auc"],
                    test_metrics["f1"], train_seconds)
        results.append({"candidate": cand, "model": final_model, "test_proba": test_proba, "threshold": tuned_threshold,
                        "val": val_metrics, "test": test_metrics, "test_tuned": test_metrics_tuned})
        entries.append({"version": version, "candidate": cand.name, "algorithm": cand.algorithm, "params": cand.params,
                        "validation": val_metrics, "test": test_metrics, "test_at_tuned_threshold": test_metrics_tuned,
                        "decision_threshold": tuned_threshold, "train_seconds": train_seconds, "mlflow_run_id": run_id,
                        "trained_on": "2012-10-02 to 2017-12-31", "is_champion": False})

    rule = _rule_reference(df)

    best_idx = int(np.argmax([r["val"]["pr_auc"] for r in results]))
    champion = results[best_idx]
    entries[best_idx]["is_champion"] = True
    set_champion(REGISTERED_CLASSIFIER, entries[best_idx]["version"], "highest validation PR-AUC")
    update_local_registry(REGISTERED_CLASSIFIER, entries)
    _export_champion(champion["model"], "traffic_risk_classifier", entries[best_idx])
    logger.info("Classification champion: %s (validation PR-AUC=%.3f, tuned threshold=%.3f)",
                champion["candidate"].name, champion["val"]["pr_auc"], champion["threshold"])

    _plot_classification(results, y_test.to_numpy(), champion)
    _save_metrics_table(results, "classification", extra_rows=[rule])
    return {"results": results, "champion": champion, "entries": entries, "rule_reference": rule}


def _rule_reference(df: pd.DataFrame) -> dict[str, Any]:
    """Non-ML reference: flag risky weather during daytime hours (06:00-18:59).

    The proxy label is defined from weather_main/is_low_visibility (model inputs) and
    congestion (largely a function of hour). If a two-condition rule scores almost as
    well as the models, the classifier is mostly re-learning the label definition -
    a circularity that is discussed in the bias and fairness report.
    """
    rows = {}
    for split in ("validation", "test"):
        subset = df[df["split"] == split]
        risky = subset["weather_Thunderstorm"].eq(1) | subset["weather_Squall"].eq(1) | subset["is_low_visibility"].eq(1)
        flag = (risky & subset["hour"].between(6, 18)).astype(int).to_numpy()
        rows[split] = classification_metrics(subset[CLASSIFICATION_TARGET].to_numpy(), flag.astype(float))
    with mlflow.start_run(run_name="rule_reference_daytime_risky_weather"):
        mlflow.set_tags({"task": "classification", "algorithm": "Heuristic rule (no ML)", "label": "PROXY high_risk"})
        mlflow.log_param("rule", "risky_weather AND 06:00<=hour<=18:59")
        mlflow.log_metrics({f"val_{k}": v for k, v in rows["validation"].items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in rows["test"].items()})
    logger.info("Rule reference (risky weather AND daytime): test F1=%.3f precision=%.3f recall=%.3f",
                rows["test"]["f1"], rows["test"]["precision"], rows["test"]["recall"])
    return {"candidate": "rule_reference_daytime_risky_weather", "algorithm": "Heuristic rule (no ML)",
            "val": rows["validation"], "test": rows["test"]}


def run_regression(df: pd.DataFrame) -> dict[str, Any]:
    setup_experiment(EXPERIMENT_REGRESSION)
    data = split_xy(df, REGRESSION_TARGET)
    X_train, y_train = data["train"]
    X_val, y_val = data["validation"]
    X_trval, y_trval = data["train_validation"]
    X_test, y_test = data["test"]

    results, entries = [], []
    for cand in regression_candidates():
        with mlflow.start_run(run_name=cand.name):
            model, _ = _fit_timed(clone(cand.estimator), X_train, y_train)
            val_metrics = regression_metrics(y_val, model.predict(X_val))
            final_model, train_seconds = _fit_timed(clone(cand.estimator), X_trval, y_trval)
            test_pred = final_model.predict(X_test)
            test_metrics = regression_metrics(y_test, test_pred)

            mlflow.set_tags({"task": "regression", "algorithm": cand.algorithm, "target": REGRESSION_TARGET,
                             "features": "common MODEL_FEATURES", "split": "time-based"})
            mlflow.log_params({**cand.params, "n_features": len(MODEL_FEATURES)})
            mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            mlflow.log_metric("train_seconds", train_seconds)
            version = log_sklearn_model(final_model, X_test, REGISTERED_REGRESSOR)
            run_id = mlflow.active_run().info.run_id

        logger.info("%-28s val MAE=%.0f R2=%.3f | test MAE=%.0f RMSE=%.0f R2=%.3f (%.1fs)", cand.name,
                    val_metrics["mae"], val_metrics["r2"], test_metrics["mae"], test_metrics["rmse"], test_metrics["r2"],
                    train_seconds)
        results.append({"candidate": cand, "model": final_model, "test_pred": test_pred, "val": val_metrics, "test": test_metrics})
        entries.append({"version": version, "candidate": cand.name, "algorithm": cand.algorithm, "params": cand.params,
                        "validation": val_metrics, "test": test_metrics, "train_seconds": train_seconds,
                        "mlflow_run_id": run_id, "trained_on": "2012-10-02 to 2017-12-31", "is_champion": False})

    best_idx = int(np.argmin([r["val"]["mae"] for r in results]))
    champion = results[best_idx]
    entries[best_idx]["is_champion"] = True
    set_champion(REGISTERED_REGRESSOR, entries[best_idx]["version"], "lowest validation MAE")
    update_local_registry(REGISTERED_REGRESSOR, entries)
    _export_champion(champion["model"], "traffic_volume_regressor", entries[best_idx])
    logger.info("Regression champion: %s (validation MAE=%.0f)", champion["candidate"].name, champion["val"]["mae"])

    test_frame = df[df["split"] == "test"]
    _plot_regression(results, test_frame, champion)
    _save_metrics_table(results, "regression")
    return {"results": results, "champion": champion, "entries": entries}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _export_champion(model: Any, stem: str, entry: dict[str, Any]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / f"{stem}.joblib", compress=3)
    metadata = {**entry, "features": MODEL_FEATURES}
    (MODELS_DIR / f"{stem}.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    logger.info("Saved champion model to models/%s.joblib (+ metadata json)", stem)


def _save_metrics_table(results: list[dict[str, Any]], task: str, extra_rows: list[dict[str, Any]] | None = None) -> None:
    rows = []
    for extra in extra_rows or []:
        row = {"candidate": extra["candidate"], "algorithm": extra["algorithm"]}
        row.update({f"val_{k}": v for k, v in extra["val"].items()})
        row.update({f"test_{k}": v for k, v in extra["test"].items()})
        rows.append(row)
    for r in results:
        row = {"candidate": r["candidate"].name, "algorithm": r["candidate"].algorithm}
        row.update({f"val_{k}": v for k, v in r["val"].items()})
        row.update({f"test_{k}": v for k, v in r["test"].items()})
        if "test_tuned" in r:
            row.update({f"test_tuned_{k}": v for k, v in r["test_tuned"].items()})
            row["tuned_threshold"] = r["threshold"]
        rows.append(row)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / f"{task}_metrics.csv"
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.4f")
    logger.info("Saved %s metrics table to reports/metrics/%s", task, path.name)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _short_name(candidate: Candidate) -> str:
    return {
        "logistic_regression_v1": "Logistic reg. (baseline)",
        "ridge_regression_v1": "Ridge (baseline)",
        "random_forest_v1": "Random forest",
        "hist_gradient_boosting_v1": "Hist GB v1",
        "hist_gradient_boosting_v2": "Hist GB v2 (tuned)",
    }[candidate.name]


def _plot_classification(results: list[dict[str, Any]], y_test: np.ndarray, champion: dict[str, Any]) -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "ROC AUC", "PR AUC"]
    fig, ax = P.plt.subplots(figsize=(11, 5.2))
    width = 0.8 / len(results)
    x = np.arange(len(metrics))
    for i, r in enumerate(results):
        values = [r["test"][m] for m in metrics]
        bars = ax.bar(x + (i - (len(results) - 1) / 2) * width, values, width=width * 0.92, color=P.CATEGORICAL[i],
                      label=_short_name(r["candidate"]), edgecolor=P.SURFACE, linewidth=1)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.2f}", ha="center", va="bottom",
                    fontsize=7, color=P.INK_SECONDARY, rotation=90)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score on test set (Jan–Sep 2018)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncols=4, bbox_to_anchor=(0, 1.0))
    ax.set_title("Every model separates the proxy label almost perfectly: the label is largely defined by its inputs")
    P.subtitle(ax, "Proxy high-risk classification, test-set metrics at a 0.5 decision threshold. Trees mainly add precision.")
    P.save_figure(fig, "task1_classification_metrics.png", "task1_supervised")

    fig, (ax_roc, ax_pr) = P.plt.subplots(1, 2, figsize=(11, 4.8))
    for i, r in enumerate(results):
        fpr, tpr, _ = roc_curve(y_test, r["test_proba"])
        prec, rec, _ = precision_recall_curve(y_test, r["test_proba"])
        name = _short_name(r["candidate"])
        ax_roc.plot(fpr, tpr, color=P.CATEGORICAL[i], label=f"{name} (AUC {r['test']['roc_auc']:.3f})")
        ax_pr.plot(rec, prec, color=P.CATEGORICAL[i], label=f"{name} (AP {r['test']['pr_auc']:.3f})")
    ax_roc.plot([0, 1], [0, 1], color=P.INK_MUTED, linestyle="--", linewidth=1)
    ax_pr.axhline(y_test.mean(), color=P.INK_MUTED, linestyle="--", linewidth=1)
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate")
    ax_pr.set(xlabel="Recall", ylabel="Precision")
    ax_roc.set_title("ROC curves (test)")
    ax_pr.set_title("Precision–recall curves (test)")
    ax_roc.legend(loc="lower right", fontsize=8)
    ax_pr.legend(loc="lower left", bbox_to_anchor=(0.0, 0.12), fontsize=8)
    ax_pr.text(0.01, y_test.mean() + 0.015, f"Prevalence {y_test.mean():.1%}", fontsize=8, color=P.INK_SECONDARY)
    P.save_figure(fig, "task1_classification_roc_pr.png", "task1_supervised")

    pred = (champion["test_proba"] >= champion["threshold"]).astype(int)
    cm = confusion_matrix(y_test, pred)
    fig, ax = P.plt.subplots(figsize=(5.4, 4.6))
    ax.imshow(cm, cmap=P.plt.matplotlib.colors.LinearSegmentedColormap.from_list("b", P.BLUE_RAMP))
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, f"{value:,}", ha="center", va="center", fontsize=12,
                color="white" if value > cm.max() / 2 else P.INK_PRIMARY)
    ax.set_xticks([0, 1], ["Predicted normal", "Predicted high risk"])
    ax.set_yticks([0, 1], ["Actual normal", "Actual high risk"])
    ax.grid(False)
    ax.set_title(f"Champion confusion matrix (threshold {champion['threshold']:.2f})")
    P.subtitle(ax, f"{_short_name(champion['candidate'])}, test set Jan–Sep 2018")
    P.save_figure(fig, "task1_classification_confusion_matrix.png", "task1_supervised")


def _plot_regression(results: list[dict[str, Any]], test_frame: pd.DataFrame, champion: dict[str, Any]) -> None:
    names = [_short_name(r["candidate"]) for r in results]
    fig, (ax_mae, ax_r2) = P.plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, key, label, fmt in ((ax_mae, "mae", "MAE (vehicles/hour, lower is better)", "{:,.0f}"),
                                (ax_r2, "r2", "R² (higher is better)", "{:.3f}")):
        values = [r["test"][key] for r in results]
        colours = [P.SERIES_2 if r is champion else P.SERIES_1 for r in results]
        ax.barh(names, values, color=colours, edgecolor=P.SURFACE, linewidth=2, height=0.7)
        for i, value in enumerate(values):
            ax.text(value, i, "  " + fmt.format(value), va="center", fontsize=9, color=P.INK_SECONDARY)
        ax.set_xlabel(label)
        ax.set_xlim(0, max(values) * 1.25)
        ax.grid(axis="y", visible=False)
        ax.invert_yaxis()
    ax_r2.set_xlim(0, 1.12)
    ax_r2.set_xticks(np.arange(0, 1.01, 0.2))
    ax_r2.set_yticklabels([])
    reduction = 1 - champion["test"]["mae"] / results[0]["test"]["mae"]
    ax_mae.set_title(f"Tree ensembles cut the linear baseline's error by {reduction:.0%}")
    P.subtitle(ax_mae, "Traffic volume regression, test set Jan–Sep 2018 (champion in orange)")
    P.save_figure(fig, "task1_regression_metrics.png", "task1_supervised")

    y = test_frame["traffic_volume"].to_numpy()
    pred = champion["test_pred"]
    fig, (ax_scatter, ax_hour) = P.plt.subplots(1, 2, figsize=(11, 4.8))
    ax_scatter.hexbin(y, pred, gridsize=40, mincnt=1, cmap=P.plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "b", [P.SURFACE] + P.BLUE_RAMP), linewidths=0)
    ax_scatter.plot([0, 7500], [0, 7500], color=P.SERIES_2, linewidth=1.5)
    ax_scatter.set(xlabel="Actual vehicles/hour", ylabel="Predicted vehicles/hour", xlim=(0, 7500), ylim=(0, 7500))
    ax_scatter.xaxis.set_major_formatter(P.thousands)
    ax_scatter.yaxis.set_major_formatter(P.thousands)
    ax_scatter.set_title("Predicted vs actual (champion)")
    ax_scatter.grid(False)

    baseline = results[0]
    hours = test_frame["hour"].to_numpy()
    for r, colour, label in ((baseline, P.SERIES_1, _short_name(baseline["candidate"])),
                             (champion, P.SERIES_2, _short_name(champion["candidate"]))):
        mae_by_hour = pd.Series(np.abs(y - r["test_pred"])).groupby(hours).mean()
        ax_hour.plot(mae_by_hour.index, mae_by_hour.values, color=colour, marker="o", markersize=3.5, label=label)
    ax_hour.set(xlabel="Hour of day", ylabel="MAE (vehicles/hour)")
    ax_hour.set_xticks(range(0, 24, 3), [f"{h:02d}:00" for h in range(0, 24, 3)])
    ax_hour.yaxis.set_major_formatter(P.thousands)
    ax_hour.set_ylim(0, None)
    ax_hour.legend(loc="upper right")
    ax_hour.set_title("Error by hour of day (test)")
    P.save_figure(fig, "task1_regression_diagnostics.png", "task1_supervised")


def run(df: pd.DataFrame | None = None) -> dict[str, Any]:
    df = load_modelling_table() if df is None else df
    logger.info("Task 1: supervised learning with %d common features", len(MODEL_FEATURES))
    return {"classification": run_classification(df), "regression": run_regression(df)}


if __name__ == "__main__":
    from logging_config import configure_logging

    configure_logging("supervised_models")
    run()
