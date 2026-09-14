"""
deep_learning.py - Task 3: LSTM for next-hour traffic volume (sequential behaviour).

Problem
-------
Given the previous 24 hours (traffic volume, calendar and weather) plus the *known*
calendar and forecast weather for the next hour, predict traffic volume in the next hour.

Data handling
-------------
* The series has gaps (sensor outages, the 2014-15 gap). A window is only used if all
  25 hours (24 inputs + target) are consecutive, so the LSTM never learns across a gap.
* Samples are assigned to train/validation/test by the TARGET timestamp using the same
  time-based boundaries as Task 1. Scalers are fitted on training rows only.

Models compared on identical test samples
-----------------------------------------
* Persistence baseline: next hour = current hour.
* Seasonal naive baseline: next hour = same hour yesterday.
* Task 1 champion regressor (no lag features).
* HistGradientBoosting with lag features: the *explainable surrogate* for the LSTM. It
  solves the same next-hour problem with the same information, so SHAP on it explains
  which signals drive next-hour demand (see explainability.py and the report).
* LSTM v1 (1 layer, 32 units) and LSTM v2 (2 layers, 64 units, dropout).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import plotting as P
from config import (
    EXPERIMENT_DEEP_LEARNING,
    METRICS_DIR,
    MODEL_FEATURES,
    MODELS_DIR,
    RANDOM_SEED,
    REGISTERED_LSTM,
    TRAIN_END,
    VALIDATION_END,
)
from data_prep import load_modelling_table
from features import make_feature_frame
from mlflow_utils import set_champion, setup_experiment, update_local_registry
from supervised_models import regression_metrics

logger = logging.getLogger(__name__)

LOOKBACK = 24
SEQUENCE_FEATURES = ["traffic_volume", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_holiday", "temp_c",
                     "weather_severity", "is_precipitation", "clouds_all"]
NEXT_HOUR_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "is_holiday", "is_rush_hour",
                      "temp_c", "weather_severity", "is_precipitation", "is_low_visibility"]
LAG_FEATURES = ["lag_1", "lag_2", "lag_3", "lag_24", "rolling_mean_24", "rolling_std_24"]
MAX_EPOCHS = 40
PATIENCE = 5
BATCH_SIZE = 256
MAX_FILL_GAP_HOURS = 3   # 2,452 of 2,588 gaps are <= 3 hours; longer gaps still break sequences


@dataclass
class LSTMConfig:
    name: str
    hidden_size: int
    num_layers: int
    dropout: float
    learning_rate: float


LSTM_CONFIGS = [
    LSTMConfig("lstm_v1_small", hidden_size=32, num_layers=1, dropout=0.0, learning_rate=2e-3),
    LSTMConfig("lstm_v2_stacked", hidden_size=64, num_layers=2, dropout=0.2, learning_rate=1e-3),
]


class TrafficLSTM(nn.Module):
    """LSTM encoder over the last 24 hours, concatenated with next-hour known features."""

    def __init__(self, n_sequence_features: int, n_next_hour_features: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(n_sequence_features, hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + n_next_hour_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, sequence: torch.Tensor, next_hour: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.lstm(sequence)
        return self.head(torch.cat([encoded[:, -1, :], next_hour], dim=1)).squeeze(1)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def fill_short_gaps(df: pd.DataFrame, max_gap_hours: int = MAX_FILL_GAP_HOURS) -> pd.DataFrame:
    """Insert rows for missing hours inside gaps of <= max_gap_hours so short outages do not break sequences.

    Volume and numeric weather are linearly interpolated, weather labels carried forward, and
    calendar/weather features recomputed from the timestamp with the shared feature builder.
    Inserted rows are flagged `is_interpolated` and are never used as prediction targets.
    """
    base_cols = ["date_time", "traffic_volume", "temp_c", "rain_1h", "snow_1h", "clouds_all", "weather_main",
                 "weather_description", "is_holiday", "date"]
    base = df[base_cols].sort_values("date_time").set_index("date_time")
    full_index = pd.date_range(base.index.min(), base.index.max(), freq="h")
    full = base.reindex(full_index)
    missing = full["traffic_volume"].isna()
    run_id = (~missing).cumsum()
    run_length = missing.groupby(run_id).transform("sum")
    fillable = missing & (run_length <= max_gap_hours)

    numeric = ["traffic_volume", "temp_c", "rain_1h", "snow_1h", "clouds_all"]
    interpolated = full[numeric].interpolate(method="linear", limit=max_gap_hours, limit_area="inside")
    full.loc[fillable, numeric] = interpolated.loc[fillable, numeric]
    for column in ("weather_main", "weather_description"):
        full[column] = full[column].ffill(limit=max_gap_hours)
    holiday_dates = set(base.loc[base["is_holiday"] == 1, "date"])
    full["date"] = full.index.normalize()
    full["is_holiday"] = full["date"].isin(holiday_dates).astype(int)
    full = full.loc[~missing | fillable].copy()
    full["is_interpolated"] = fillable.loc[full.index].to_numpy()
    full.index.name = "date_time"
    full = full.reset_index()

    features = make_feature_frame(full)
    full = pd.concat([full, features.drop(columns=[c for c in features.columns if c in full.columns])], axis=1)
    full["split"] = np.select([full["date_time"] <= TRAIN_END, full["date_time"] <= VALIDATION_END],
                              ["train", "validation"], default="test")
    logger.warning("Interpolated %d missing hour(s) in %d gap(s) of <= %d hours so sequences can span short outages "
                   "(inputs only, never targets); %d hour(s) in longer gaps left missing",
                   int(fillable.sum()), int((fillable & ~fillable.shift(fill_value=False)).sum()), max_gap_hours,
                   int((missing & ~fillable).sum()))
    return full


def build_sequences(df: pd.DataFrame) -> dict[str, Any]:
    """Create (sequence, next-hour features, target) arrays for every valid contiguous window."""
    df = fill_short_gaps(df)
    gap = df["date_time"].diff().ne(pd.Timedelta(hours=1))
    segment = gap.cumsum()
    position = df.groupby(segment).cumcount()
    # The 24 preceding rows are in the same contiguous segment, and the target itself is a real observation.
    valid_target = (position >= LOOKBACK) & ~df["is_interpolated"]
    target_idx = np.flatnonzero(valid_target.to_numpy())
    logger.info("Sequence windows: %d valid next-hour targets from %d rows (%d contiguous segments)",
                len(target_idx), len(df), int(segment.max()))

    train_rows = df["split"].eq("train").to_numpy()
    seq_mean = df.loc[train_rows, SEQUENCE_FEATURES].mean()
    seq_std = df.loc[train_rows, SEQUENCE_FEATURES].std().replace(0, 1)
    nxt_mean = df.loc[train_rows, NEXT_HOUR_FEATURES].mean()
    nxt_std = df.loc[train_rows, NEXT_HOUR_FEATURES].std().replace(0, 1)
    logger.debug("Volume scaling from training rows: mean=%.1f std=%.1f", seq_mean["traffic_volume"], seq_std["traffic_volume"])

    seq_values = ((df[SEQUENCE_FEATURES] - seq_mean) / seq_std).to_numpy(dtype=np.float32)
    nxt_values = ((df[NEXT_HOUR_FEATURES] - nxt_mean) / nxt_std).to_numpy(dtype=np.float32)
    offsets = np.arange(-LOOKBACK, 0)
    sequences = seq_values[target_idx[:, None] + offsets[None, :]]           # (n, 24, n_features)
    next_hour = nxt_values[target_idx]
    volume = df["traffic_volume"].to_numpy(dtype=np.float32)
    target_scaled = (volume[target_idx] - seq_mean["traffic_volume"]) / seq_std["traffic_volume"]

    # Lag features for the tree surrogate and naive baselines (same rows, same information).
    lags = pd.DataFrame({
        "lag_1": volume[target_idx - 1],
        "lag_2": volume[target_idx - 2],
        "lag_3": volume[target_idx - 3],
        "lag_24": volume[target_idx - 24],
        "rolling_mean_24": volume[target_idx[:, None] + offsets[None, :]].mean(axis=1),
        "rolling_std_24": volume[target_idx[:, None] + offsets[None, :]].std(axis=1),
    })
    tabular = pd.concat([lags, df.loc[target_idx, MODEL_FEATURES].reset_index(drop=True)], axis=1)

    return {
        "frame": df.loc[target_idx].reset_index(drop=True),
        "sequences": sequences,
        "next_hour": next_hour,
        "target_scaled": target_scaled.astype(np.float32),
        "target": volume[target_idx],
        "tabular": tabular,
        "split": df.loc[target_idx, "split"].to_numpy(),
        "scaling": {"sequence_mean": seq_mean.to_dict(), "sequence_std": seq_std.to_dict(),
                    "next_hour_mean": nxt_mean.to_dict(), "next_hour_std": nxt_std.to_dict()},
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _loader(data: dict[str, Any], mask: np.ndarray, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(data["sequences"][mask]), torch.from_numpy(data["next_hour"][mask]),
                            torch.from_numpy(data["target_scaled"][mask]))
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, generator=generator)


def _predict(model: TrafficLSTM, data: dict[str, Any], mask: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        scaled = model(torch.from_numpy(data["sequences"][mask]), torch.from_numpy(data["next_hour"][mask])).numpy()
    mean = data["scaling"]["sequence_mean"]["traffic_volume"]
    std = data["scaling"]["sequence_std"]["traffic_volume"]
    return np.clip(scaled * std + mean, 0, None)


def train_lstm(config: LSTMConfig, data: dict[str, Any]) -> dict[str, Any]:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    train_mask, val_mask, test_mask = (data["split"] == s for s in ("train", "validation", "test"))
    model = TrafficLSTM(len(SEQUENCE_FEATURES), len(NEXT_HOUR_FEATURES), config.hidden_size, config.num_layers, config.dropout)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.5, patience=2)
    loss_fn = nn.HuberLoss(delta=1.0)
    train_loader = _loader(data, train_mask, shuffle=True)
    n_params = sum(p.numel() for p in model.parameters())

    history, best_state, best_val, bad_epochs = [], None, np.inf, 0
    start = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running = 0.0
        for seq, nxt, target in train_loader:
            optimiser.zero_grad()
            loss = loss_fn(model(seq, nxt), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            running += loss.item() * len(target)
        train_loss = running / int(train_mask.sum())
        val_mae = float(np.mean(np.abs(_predict(model, data, val_mask) - data["target"][val_mask])))
        scheduler.step(val_mae)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_mae": val_mae})
        mlflow.log_metrics({"train_huber_loss": train_loss, "val_mae_epoch": val_mae}, step=epoch)
        logger.debug("%s epoch %02d: train_loss=%.4f val_MAE=%.1f", config.name, epoch, train_loss, val_mae)
        if val_mae < best_val - 0.5:
            best_val, bad_epochs = val_mae, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                logger.info("%s early stopping at epoch %d (best validation MAE %.1f)", config.name, epoch, best_val)
                break
    train_seconds = time.perf_counter() - start
    model.load_state_dict(best_state)

    val_metrics = regression_metrics(data["target"][val_mask], _predict(model, data, val_mask))
    test_pred = _predict(model, data, test_mask)
    test_metrics = regression_metrics(data["target"][test_mask], test_pred)
    logger.info("%-16s params=%d epochs=%d val MAE=%.0f | test MAE=%.0f RMSE=%.0f R2=%.3f (%.0fs)", config.name, n_params,
                len(history), val_metrics["mae"], test_metrics["mae"], test_metrics["rmse"], test_metrics["r2"], train_seconds)
    return {"config": config, "model": model, "history": history, "val": val_metrics, "test": test_metrics,
            "test_pred": test_pred, "train_seconds": train_seconds, "n_params": n_params}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run(df: pd.DataFrame | None = None) -> dict[str, Any]:
    df = load_modelling_table() if df is None else df
    torch.set_num_threads(max(1, torch.get_num_threads()))
    setup_experiment(EXPERIMENT_DEEP_LEARNING)
    data = build_sequences(df)
    train_mask, val_mask, test_mask = (data["split"] == s for s in ("train", "validation", "test"))
    logger.info("Sequence samples: train=%d validation=%d test=%d", train_mask.sum(), val_mask.sum(), test_mask.sum())
    y_test = data["target"][test_mask]
    tab = data["tabular"]

    comparisons: dict[str, dict[str, float]] = {}
    comparisons["Persistence (last hour)"] = regression_metrics(y_test, tab.loc[test_mask, "lag_1"])
    comparisons["Seasonal naive (same hour yesterday)"] = regression_metrics(y_test, tab.loc[test_mask, "lag_24"])
    task1 = joblib.load(MODELS_DIR / "traffic_volume_regressor.joblib")
    comparisons["Task 1 regressor (no lags)"] = regression_metrics(y_test, task1.predict(tab.loc[test_mask, MODEL_FEATURES]))

    # Explainable surrogate: gradient boosting with lag features on exactly the same samples.
    surrogate_features = LAG_FEATURES + MODEL_FEATURES
    trval = train_mask | val_mask
    with mlflow.start_run(run_name="hgb_lag_surrogate"):
        surrogate = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=600, max_leaf_nodes=63, min_samples_leaf=30,
                                                  l2_regularization=1.0, random_state=RANDOM_SEED)
        surrogate.fit(tab.loc[train_mask, surrogate_features], data["target"][train_mask])
        surrogate_val = regression_metrics(data["target"][val_mask], surrogate.predict(tab.loc[val_mask, surrogate_features]))
        start = time.perf_counter()
        surrogate.fit(tab.loc[trval, surrogate_features], data["target"][trval])
        surrogate_seconds = time.perf_counter() - start
        surrogate_pred = surrogate.predict(tab.loc[test_mask, surrogate_features])
        comparisons["Hist GB + lag features (surrogate)"] = regression_metrics(y_test, surrogate_pred)
        mlflow.set_tags({"task": "next-hour regression", "algorithm": "HistGradientBoostingRegressor",
                         "role": "explainable surrogate for LSTM"})
        mlflow.log_params({"features": len(surrogate_features), "lag_features": ",".join(LAG_FEATURES)})
        mlflow.log_metrics({**{f"val_{k}": v for k, v in surrogate_val.items()},
                            **{f"test_{k}": v for k, v in comparisons["Hist GB + lag features (surrogate)"].items()},
                            "train_seconds": surrogate_seconds})
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": surrogate, "features": surrogate_features}, MODELS_DIR / "lag_surrogate_regressor.joblib", compress=3)
    logger.info("Surrogate (HGB + lags): test MAE=%.0f R2=%.3f; saved models/lag_surrogate_regressor.joblib",
                comparisons["Hist GB + lag features (surrogate)"]["mae"], comparisons["Hist GB + lag features (surrogate)"]["r2"])

    lstm_results, entries = [], []
    for config in LSTM_CONFIGS:
        with mlflow.start_run(run_name=config.name):
            mlflow.set_tags({"task": "next-hour regression", "algorithm": "PyTorch LSTM", "framework": f"torch {torch.__version__}"})
            mlflow.log_params({"lookback_hours": LOOKBACK, "hidden_size": config.hidden_size, "num_layers": config.num_layers,
                               "dropout": config.dropout, "learning_rate": config.learning_rate, "batch_size": BATCH_SIZE,
                               "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "loss": "Huber",
                               "sequence_features": ",".join(SEQUENCE_FEATURES), "next_hour_features": ",".join(NEXT_HOUR_FEATURES)})
            result = train_lstm(config, data)
            mlflow.log_metrics({**{f"val_{k}": v for k, v in result["val"].items()},
                                **{f"test_{k}": v for k, v in result["test"].items()},
                                "train_seconds": result["train_seconds"], "n_parameters": result["n_params"],
                                "epochs_trained": len(result["history"])})
            # Two-input model (sequence + next-hour tensors): the pickle format avoids pt2's single-TensorSpec signature.
            info = mlflow.pytorch.log_model(result["model"].eval(), name="model", serialization_format="pickle",
                                            registered_model_name=REGISTERED_LSTM)
            version = str(info.registered_model_version)
            run_id = mlflow.active_run().info.run_id
        comparisons[f"LSTM {config.name.split('_', 1)[1].replace('_', ' ')}"] = result["test"]
        lstm_results.append(result)
        entries.append({"version": version, "candidate": config.name, "algorithm": "PyTorch LSTM",
                        "params": {"hidden_size": config.hidden_size, "num_layers": config.num_layers, "dropout": config.dropout,
                                   "learning_rate": config.learning_rate, "lookback": LOOKBACK},
                        "validation": result["val"], "test": result["test"], "train_seconds": result["train_seconds"],
                        "n_parameters": result["n_params"], "epochs_trained": len(result["history"]),
                        "mlflow_run_id": run_id, "trained_on": "2012-10-02 to 2016-12-31 (early stopping on 2017)",
                        "is_champion": False})

    best = int(np.argmin([r["val"]["mae"] for r in lstm_results]))
    entries[best]["is_champion"] = True
    set_champion(REGISTERED_LSTM, entries[best]["version"], "lowest validation MAE")
    update_local_registry(REGISTERED_LSTM, entries)
    champion = lstm_results[best]
    torch.save(champion["model"].state_dict(), MODELS_DIR / "lstm_traffic_volume.pt")
    (MODELS_DIR / "lstm_traffic_volume.json").write_text(json.dumps({
        **entries[best], "sequence_features": SEQUENCE_FEATURES, "next_hour_features": NEXT_HOUR_FEATURES,
        "scaling": data["scaling"]}, indent=2, default=str), encoding="utf-8")
    logger.info("LSTM champion %s saved to models/lstm_traffic_volume.pt", champion["config"].name)

    table = pd.DataFrame(comparisons).T
    table.index.name = "model"
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(METRICS_DIR / "deep_learning_comparison.csv", float_format="%.4f")
    logger.info("Saved next-hour model comparison to reports/metrics/deep_learning_comparison.csv")
    _plot(lstm_results, table, data, test_mask, champion, surrogate_pred)
    return {"comparison": table, "lstm": lstm_results, "data": data}


def _plot(lstm_results, table, data, test_mask, champion, surrogate_pred) -> None:
    fig, ax = P.plt.subplots(figsize=(10, 4.4))
    for i, result in enumerate(lstm_results):
        history = pd.DataFrame(result["history"])
        ax.plot(history["epoch"], history["val_mae"], color=P.CATEGORICAL[i], marker="o", markersize=3.5,
                label=f"{result['config'].name} ({result['n_params']:,} params)")
    ax.set(xlabel="Epoch", ylabel="Validation MAE (vehicles/hour)")
    ax.set_ylim(0, None)
    ax.legend(loc="upper right")
    ax.set_title("Both LSTMs converge within a few epochs; early stopping picks the best validation epoch")
    P.subtitle(ax, "Validation (2017) mean absolute error per training epoch")
    P.save_figure(fig, "task3_lstm_training_curves.png", "task3_deep_learning")

    ordered = table.sort_values("mae", ascending=False)
    fig, ax = P.plt.subplots(figsize=(10, 4.8))
    colours = [P.SERIES_2 if name.startswith("LSTM") else P.SERIES_1 for name in ordered.index]
    ax.barh(ordered.index, ordered["mae"], color=colours, edgecolor=P.SURFACE, linewidth=2, height=0.7)
    for i, (mae, r2) in enumerate(zip(ordered["mae"], ordered["r2"])):
        ax.text(mae, i, f"  MAE {mae:,.0f} · R² {r2:.3f}", va="center", fontsize=9, color=P.INK_SECONDARY)
    ax.set_xlim(0, ordered["mae"].max() * 1.35)
    ax.set_xlabel("Test MAE, Jan–Sep 2018 (vehicles/hour)")
    ax.grid(axis="y", visible=False)
    best_name = table["mae"].idxmin()
    ax.set_title(f"Recent history matters: best next-hour model is {best_name}")
    P.subtitle(ax, "Next-hour traffic volume on identical test windows (LSTMs in orange)")
    P.save_figure(fig, "task3_next_hour_model_comparison.png", "task3_deep_learning")

    frame = data["frame"][test_mask].reset_index(drop=True)
    window = frame["date_time"].between("2018-03-05", "2018-03-18 23:00")
    fig, ax = P.plt.subplots(figsize=(11, 4.6))
    ax.plot(frame.loc[window, "date_time"], frame.loc[window, "traffic_volume"], color=P.INK_MUTED, linewidth=2.5, label="Actual")
    ax.plot(frame.loc[window, "date_time"], champion["test_pred"][window.to_numpy()], color=P.SERIES_2, linewidth=1.5,
            label=f"LSTM ({champion['config'].name})")
    ax.plot(frame.loc[window, "date_time"], surrogate_pred[window.to_numpy()], color=P.SERIES_1, linewidth=1.2,
            linestyle="--", label="Hist GB + lags (surrogate)")
    ax.yaxis.set_major_formatter(P.thousands)
    ax.set_ylabel("Vehicles per hour")
    ax.legend(loc="upper left", ncols=3)
    ax.set_ylim(0, frame.loc[window, "traffic_volume"].max() * 1.2)
    ax.set_title("Two test weeks: both models track the daily commuter cycle and weekend dips")
    P.subtitle(ax, "Next-hour predictions vs actual traffic, 5–18 March 2018")
    P.save_figure(fig, "task3_lstm_test_weeks.png", "task3_deep_learning")


if __name__ == "__main__":
    from logging_config import configure_logging

    configure_logging("deep_learning")
    run()
