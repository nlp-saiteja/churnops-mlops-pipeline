import pandas as pd
import mlflow
import mlflow.xgboost
from sklearn.metrics import classification_report, confusion_matrix
import yaml
import json


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_production_model(config: dict):
    """Load the latest registered model from MLflow Model Registry."""
    mlflow_config = config["mlflow"]
    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])

    model_uri = f"models:/{mlflow_config['model_name']}/latest"
    model = mlflow.xgboost.load_model(model_uri)
    print(f"Loaded model: {model_uri}")
    return model


def evaluate(config: dict = None):
    if config is None:
        config = load_config()

    processed_path = config["data"]["processed_path"]
    threshold = config["model"]["threshold"]

    # Load test data
    X_test = pd.read_csv(f"{processed_path}/X_test.csv")
    y_test = pd.read_csv(f"{processed_path}/y_test.csv").squeeze()

    # Load model
    model = load_production_model(config)

    # Predict
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Print full report
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    print("--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  True Negatives  (correctly said no churn):  {cm[0][0]}")
    print(f"  False Positives (wrongly flagged as churn): {cm[0][1]}")
    print(f"  False Negatives (missed actual churners):   {cm[1][0]}")
    print(f"  True Positives  (correctly caught churners):{cm[1][1]}")

    # Load saved metrics
    try:
        with open("models/metrics.json") as f:
            metrics = json.load(f)
        print("\n--- Saved Metrics ---")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    evaluate()
