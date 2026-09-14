"""
run_all.py - Entry point that reproduces every Part 3 result end to end.

    python run_all.py                       # all stages, INFO logging -> logs/run_all.log
    python run_all.py --fresh-mlflow        # wipe mlflow/mlflow.db + mlflow/mlartifacts first (clean version numbers)
    python run_all.py --stages monitoring fairness --log-level DEBUG

Stages (in order)
-----------------
data         build the modelling table (Part 2 cleaned data + features + proxy label + split)
supervised   Task 1  classification + regression candidates, MLflow tracking and registry
unsupervised Task 2  K-means clustering + association rules
deep         Task 3  LSTM + lag-feature surrogate
explain      Task 3  SHAP explanations
recommend    Task 5  serving reference + recommendation scenarios
api          Task 6.3 exercise the FastAPI app in-process and save responses
monitoring   Task 6.4/6.5 drift + error monitoring, PASS/ALERT dashboard
fairness     Task 7  coverage, proxy-label, segment-error and footprint analysis
exports      Task 6.1/6.2 export MLflow runs summary and MODEL_VERSIONS.md
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
import warnings

from config import MLFLOW_ARTIFACT_ROOT, MLFLOW_DB, MODEL_REGISTRY_FILE
from logging_config import configure_logging

logger = logging.getLogger(__name__)

STAGES = ["data", "supervised", "unsupervised", "deep", "explain", "recommend", "api", "monitoring", "fairness", "exports"]


def run_stage(name: str, df):
    if name == "data":
        import data_prep
        return data_prep.build_modelling_table(save=True)
    if name == "supervised":
        import supervised_models
        supervised_models.run(df)
    elif name == "unsupervised":
        import unsupervised
        unsupervised.run(df)
    elif name == "deep":
        import deep_learning
        deep_learning.run(df)
    elif name == "explain":
        import explainability
        explainability.run(df)
    elif name == "recommend":
        from recommendation_system import recommender
        recommender.run(df)
    elif name == "api":
        from deployment import demo_client
        demo_client.run()
    elif name == "monitoring":
        from monitoring import monitoring
        monitoring.run(df)
    elif name == "fairness":
        import fairness_analysis
        fairness_analysis.run(df)
    elif name == "exports":
        import mlflow_utils
        mlflow_utils.export_runs_summary()
        mlflow_utils.export_model_versions_markdown()
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce all Part 3 results")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    parser.add_argument("--fresh-mlflow", action="store_true", help="Delete the MLflow store and model registry first")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)
    log_path = configure_logging("run_all", args.log_level)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*Pickle or CloudPickle.*")
    logger.info("Part 3 pipeline started: stages=%s, log=%s", " -> ".join(args.stages), log_path.name)

    if args.fresh_mlflow:
        MLFLOW_DB.unlink(missing_ok=True)
        shutil.rmtree(MLFLOW_ARTIFACT_ROOT, ignore_errors=True)
        MODEL_REGISTRY_FILE.unlink(missing_ok=True)
        logger.warning("Deleted existing MLflow store, artifacts and local model registry (--fresh-mlflow)")

    df = None
    if "data" not in args.stages:
        import data_prep
        df = data_prep.load_modelling_table()

    total_start = time.perf_counter()
    for stage in args.stages:
        start = time.perf_counter()
        logger.info("===== Stage '%s' started =====", stage)
        try:
            df = run_stage(stage, df)
        except Exception:  # log full context and stop: later stages depend on earlier outputs
            logger.error("Stage '%s' failed; pipeline stopped", stage, exc_info=True)
            return 1
        logger.info("===== Stage '%s' finished in %.1f s =====", stage, time.perf_counter() - start)
    logger.info("Part 3 pipeline finished successfully in %.1f s", time.perf_counter() - total_start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
