"""
mlflow_utils.py - MLflow experiment tracking, model registry and version export helpers.

Backend: SQLite (`mlflow.db`, required for the Model Registry) with artifacts in
`mlartifacts/`. Browse with:

    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from config import (  # noqa: E402
    CHAMPION_ALIAS,
    METRICS_DIR,
    MLFLOW_ARTIFACT_ROOT,
    MLFLOW_TRACKING_URI,
    MODEL_REGISTRY_FILE,
    PROJECT_ROOT,
    REPORTS_DIR,
)

logger = logging.getLogger(__name__)


def setup_experiment(name: str) -> str:
    """Point MLflow at the local SQLite store and create/select an experiment. Returns its id."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        artifact_location = (MLFLOW_ARTIFACT_ROOT / name).as_uri()
        experiment_id = client.create_experiment(name, artifact_location=artifact_location)
        logger.info("Created MLflow experiment '%s' (id=%s)", name, experiment_id)
    else:
        experiment_id = experiment.experiment_id
        logger.info("Using MLflow experiment '%s' (id=%s)", name, experiment_id)
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def log_sklearn_model(model: Any, X_example: pd.DataFrame, registered_name: str | None) -> str | None:
    """Log a scikit-learn model with signature; register it (new version) if a name is given.

    Returns the registered version number as a string, or None if not registered.
    """
    example = X_example.head(5)
    signature = infer_signature(example, model.predict(example))
    info = mlflow.sklearn.log_model(
        model, name="model", signature=signature, input_example=example, registered_model_name=registered_name
    )
    version = getattr(info, "registered_model_version", None)
    if registered_name:
        logger.info("Registered '%s' version %s (run %s)", registered_name, version, mlflow.active_run().info.run_id[:8])
    return None if version is None else str(version)


def set_champion(registered_name: str, version: str, reason: str) -> None:
    client = MlflowClient()
    client.set_registered_model_alias(registered_name, CHAMPION_ALIAS, version)
    client.set_model_version_tag(registered_name, version, "selection_reason", reason)
    logger.info("Set alias '%s' -> %s v%s (%s)", CHAMPION_ALIAS, registered_name, version, reason)


def update_local_registry(model_name: str, entries: list[dict[str, Any]]) -> None:
    """Mirror model versions into models/model_registry.json (human-readable, git-friendly)."""
    registry: dict[str, Any] = {}
    if MODEL_REGISTRY_FILE.exists():
        registry = json.loads(MODEL_REGISTRY_FILE.read_text(encoding="utf-8"))
    registry[model_name] = {
        "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": entries,
    }
    MODEL_REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODEL_REGISTRY_FILE.write_text(json.dumps(registry, indent=2, default=str), encoding="utf-8")
    logger.info("Updated local model registry for '%s' (%d versions)", model_name, len(entries))


def export_runs_summary() -> pd.DataFrame:
    """Export every tracked run (params + metrics) to reports/metrics/mlflow_runs_summary.csv."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    experiments = [e for e in client.search_experiments() if e.name != "Default"]
    runs = mlflow.search_runs(experiment_ids=[e.experiment_id for e in experiments], output_format="pandas")
    if runs.empty:
        logger.warning("No MLflow runs found to export")
        return runs
    names = {e.experiment_id: e.name for e in experiments}
    runs.insert(0, "experiment", runs["experiment_id"].map(names))
    keep = ["experiment", "run_id", "tags.mlflow.runName", "start_time"] + sorted(
        c for c in runs.columns if c.startswith(("params.", "metrics."))
    )
    summary = runs[[c for c in keep if c in runs.columns]].sort_values(["experiment", "start_time"])
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / "mlflow_runs_summary.csv"
    summary.to_csv(path, index=False)
    logger.info("Exported %d MLflow runs across %d experiments to %s", len(summary), len(experiments),
                path.relative_to(PROJECT_ROOT).as_posix())
    return summary


def export_model_versions_markdown() -> None:
    """Write reports/MODEL_VERSIONS.md from the local registry mirror (Task 6.1)."""
    if not MODEL_REGISTRY_FILE.exists():
        logger.warning("No local model registry found; skipping MODEL_VERSIONS.md")
        return
    registry = json.loads(MODEL_REGISTRY_FILE.read_text(encoding="utf-8"))
    lines = [
        "# Model Versions",
        "",
        "Generated from `models/model_registry.json`, which mirrors the MLflow Model Registry in `mlflow.db`.",
        f"The version tagged **{CHAMPION_ALIAS}** is the one served by the API. Champions are selected on the",
        "**validation** split (2017); test metrics (Jan–Sep 2018) are reported for transparency only.",
        "",
    ]
    for model_name, block in registry.items():
        versions = block["versions"]
        lines += [f"## {model_name}", "", f"_Last updated {block['updated_utc']}_", ""]
        metric_keys = list(versions[0]["validation"].keys())
        header = ["Version", "Candidate", "Algorithm"] + [f"val {k}" for k in metric_keys] + [f"test {k}" for k in metric_keys] + ["Train s", "Status"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for v in versions:
            status = f"**{CHAMPION_ALIAS}**" if v.get("is_champion") else "archived"
            row = [str(v.get("version")), v["candidate"], v["algorithm"]]
            row += [f"{v['validation'][k]:.4f}" if abs(v['validation'][k]) < 10 else f"{v['validation'][k]:,.1f}" for k in metric_keys]
            row += [f"{v['test'][k]:.4f}" if abs(v['test'][k]) < 10 else f"{v['test'][k]:,.1f}" for k in metric_keys]
            row += [f"{v.get('train_seconds', 0):.1f}", status]
            lines.append("| " + " | ".join(row) + " |")
        lines += ["", "Hyperparameters:", ""]
        for v in versions:
            lines.append(f"* **v{v.get('version')} {v['candidate']}**: `{json.dumps(v['params'])}`")
        lines.append("")
    path = REPORTS_DIR / "MODEL_VERSIONS.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote model version history to %s", path.relative_to(PROJECT_ROOT).as_posix())
