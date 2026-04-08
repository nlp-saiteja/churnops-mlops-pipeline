import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, accuracy_score, confusion_matrix
)
import yaml
import os
import json


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_processed_data(processed_path: str):
    X_train = pd.read_csv(f"{processed_path}/X_train.csv")
    X_test = pd.read_csv(f"{processed_path}/X_test.csv")
    y_train = pd.read_csv(f"{processed_path}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{processed_path}/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "f1_score": round(f1_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
    }


def train_and_log(config: dict):
    processed_path = config["data"]["processed_path"]
    model_params = config["model"]["params"]
    mlflow_config = config["mlflow"]

    # Load data
    X_train, X_test, y_train, y_test = load_processed_data(processed_path)
    print(f"Training on {X_train.shape[0]} samples with {X_train.shape[1]} features")

    # Calculate scale_pos_weight to handle class imbalance
    # Formula: (number of negative samples) / (number of positive samples)
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = round(neg / pos, 2)
    print(f"Class imbalance ratio (scale_pos_weight): {scale_pos_weight}")

    # Setup MLflow
    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run() as run:
        print(f"MLflow Run ID: {run.info.run_id}")

        # Build model — scale_pos_weight tells XGBoost to pay more attention to churners
        model = XGBClassifier(
            n_estimators=model_params["n_estimators"],
            max_depth=model_params["max_depth"],
            learning_rate=model_params["learning_rate"],
            subsample=model_params["subsample"],
            colsample_bytree=model_params["colsample_bytree"],
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=model_params["random_state"],
            verbosity=0,
        )

        # Train
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Predict
        threshold = config["model"]["threshold"]
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        # Metrics
        metrics = compute_metrics(y_test, y_pred, y_prob)
        print("\n--- Model Performance ---")
        for name, val in metrics.items():
            print(f"  {name}: {val}")

        # Log params and metrics to MLflow
        mlflow.log_params({
            "n_estimators": model_params["n_estimators"],
            "max_depth": model_params["max_depth"],
            "learning_rate": model_params["learning_rate"],
            "subsample": model_params["subsample"],
            "colsample_bytree": model_params["colsample_bytree"],
            "scale_pos_weight": scale_pos_weight,
            "threshold": threshold,
        })
        mlflow.log_metrics(metrics)

        # Log the model itself to MLflow
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=mlflow_config["model_name"],
        )

        # Save metrics locally too (used later by monitoring)
        os.makedirs("models", exist_ok=True)
        with open("models/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\nModel registered in MLflow as '{mlflow_config['model_name']}'")
        print(f"View at: {mlflow_config['tracking_uri']}")

        return model, metrics, run.info.run_id


if __name__ == "__main__":
    config = load_config()
    model, metrics, run_id = train_and_log(config)
