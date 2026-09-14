# Model Versions

Generated from `models/model_registry.json`, which mirrors the MLflow Model Registry in `mlflow.db`.
The version tagged **champion** is the one served by the API. Champions are selected on the
**validation** split (2017); test metrics (Jan–Sep 2018) are reported for transparency only.

## traffic-risk-classifier

_Last updated 2026-09-14T09:16:51+00:00_

| Version | Candidate | Algorithm | val accuracy | val precision | val recall | val f1 | val roc_auc | val pr_auc | test accuracy | test precision | test recall | test f1 | test roc_auc | test pr_auc | Train s | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | logistic_regression_v1 | LogisticRegression (baseline) | 0.9889 | 0.8460 | 0.9962 | 0.9150 | 0.9995 | 0.9911 | 0.9891 | 0.8329 | 1.0000 | 0.9089 | 0.9992 | 0.9811 | 0.1 | archived |
| 2 | random_forest_v1 | RandomForestClassifier | 0.9960 | 0.9553 | 0.9790 | 0.9670 | 0.9997 | 0.9953 | 0.9948 | 0.9188 | 0.9915 | 0.9538 | 0.9996 | 0.9930 | 0.7 | archived |
| 3 | hist_gradient_boosting_v1 | HistGradientBoostingClassifier | 0.9952 | 0.9513 | 0.9695 | 0.9603 | 0.9997 | 0.9958 | 0.9949 | 0.9280 | 0.9831 | 0.9547 | 0.9996 | 0.9927 | 0.3 | archived |
| 4 | hist_gradient_boosting_v2 | HistGradientBoostingClassifier (tuned) | 0.9955 | 0.9450 | 0.9828 | 0.9635 | 0.9998 | 0.9967 | 0.9948 | 0.9167 | 0.9944 | 0.9539 | 0.9997 | 0.9942 | 0.4 | **champion** |

Hyperparameters:

* **v1 logistic_regression_v1**: `{"C": 1.0, "class_weight": "balanced", "scaler": "StandardScaler"}`
* **v2 random_forest_v1**: `{"n_estimators": 200, "max_leaf_nodes": 512, "min_samples_leaf": 5, "class_weight": "balanced_subsample"}`
* **v3 hist_gradient_boosting_v1**: `{"learning_rate": 0.1, "max_iter": 200, "max_leaf_nodes": 31, "class_weight": "balanced"}`
* **v4 hist_gradient_boosting_v2**: `{"learning_rate": 0.05, "max_iter": 500, "max_leaf_nodes": 15, "min_samples_leaf": 40, "l2_regularization": 1.0, "class_weight": "balanced"}`

## traffic-volume-regressor

_Last updated 2026-09-14T09:17:27+00:00_

| Version | Candidate | Algorithm | val mae | val rmse | val r2 | test mae | test rmse | test r2 | Train s | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ridge_regression_v1 | Ridge (baseline) | 570.0 | 726.7 | 0.8656 | 550.1 | 709.0 | 0.8711 | 0.0 | archived |
| 2 | random_forest_v1 | RandomForestRegressor | 231.8 | 360.2 | 0.9670 | 218.3 | 369.5 | 0.9650 | 2.0 | archived |
| 3 | hist_gradient_boosting_v1 | HistGradientBoostingRegressor | 231.5 | 365.6 | 0.9660 | 217.3 | 363.9 | 0.9660 | 0.6 | **champion** |
| 4 | hist_gradient_boosting_v2 | HistGradientBoostingRegressor (tuned) | 240.1 | 375.0 | 0.9642 | 221.5 | 366.8 | 0.9655 | 2.4 | archived |

Hyperparameters:

* **v1 ridge_regression_v1**: `{"alpha": 1.0, "scaler": "StandardScaler"}`
* **v2 random_forest_v1**: `{"n_estimators": 200, "max_leaf_nodes": 512, "min_samples_leaf": 5}`
* **v3 hist_gradient_boosting_v1**: `{"learning_rate": 0.1, "max_iter": 200, "max_leaf_nodes": 31}`
* **v4 hist_gradient_boosting_v2**: `{"learning_rate": 0.05, "max_iter": 600, "max_leaf_nodes": 63, "min_samples_leaf": 30, "l2_regularization": 1.0}`

## traffic-volume-lstm

_Last updated 2026-09-14T09:19:25+00:00_

| Version | Candidate | Algorithm | val mae | val rmse | val r2 | test mae | test rmse | test r2 | Train s | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | lstm_v1_small | PyTorch LSTM | 168.5 | 254.8 | 0.9835 | 156.2 | 233.2 | 0.9861 | 18.2 | **champion** |
| 2 | lstm_v2_stacked | PyTorch LSTM | 169.3 | 260.4 | 0.9828 | 158.2 | 241.7 | 0.9850 | 80.5 | archived |

Hyperparameters:

* **v1 lstm_v1_small**: `{"hidden_size": 32, "num_layers": 1, "dropout": 0.0, "learning_rate": 0.002, "lookback": 24}`
* **v2 lstm_v2_stacked**: `{"hidden_size": 64, "num_layers": 2, "dropout": 0.2, "learning_rate": 0.001, "lookback": 24}`
