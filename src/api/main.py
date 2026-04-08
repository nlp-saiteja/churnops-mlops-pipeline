import os
import pandas as pd
import mlflow
import mlflow.xgboost
import yaml
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from src.api.schemas import CustomerFeatures, PredictionResponse, HealthResponse


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def encode_input(data: CustomerFeatures) -> pd.DataFrame:
    """
    Convert raw human-readable input into the same numeric format
    the model was trained on.
    """
    yes_no = {"Yes": 1, "No": 0, "No phone service": 0, "No internet service": 0}

    row = {
        "gender": 1 if data.gender == "Male" else 0,
        "SeniorCitizen": data.SeniorCitizen,
        "Partner": yes_no[data.Partner],
        "Dependents": yes_no[data.Dependents],
        "tenure": data.tenure,
        "PhoneService": yes_no[data.PhoneService],
        "MultipleLines": yes_no[data.MultipleLines],
        "OnlineSecurity": yes_no[data.OnlineSecurity],
        "OnlineBackup": yes_no[data.OnlineBackup],
        "DeviceProtection": yes_no[data.DeviceProtection],
        "TechSupport": yes_no[data.TechSupport],
        "StreamingTV": yes_no[data.StreamingTV],
        "StreamingMovies": yes_no[data.StreamingMovies],
        "PaperlessBilling": yes_no[data.PaperlessBilling],
        "MonthlyCharges": data.MonthlyCharges,
        "TotalCharges": data.TotalCharges,
        # One-hot encoded columns (drop_first=True, so DSL/Month-to-month/Bank transfer are baseline=0)
        "InternetService_Fiber optic": 1 if data.InternetService == "Fiber optic" else 0,
        "InternetService_No": 1 if data.InternetService == "No" else 0,
        "Contract_One year": 1 if data.Contract == "One year" else 0,
        "Contract_Two year": 1 if data.Contract == "Two year" else 0,
        "PaymentMethod_Credit card (automatic)": 1 if data.PaymentMethod == "Credit card (automatic)" else 0,
        "PaymentMethod_Electronic check": 1 if data.PaymentMethod == "Electronic check" else 0,
        "PaymentMethod_Mailed check": 1 if data.PaymentMethod == "Mailed check" else 0,
    }

    return pd.DataFrame([row])


# Global model holder — loaded once when the server starts
model_store = {"model": None, "config": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model when the API starts up."""
    config = load_config()
    # MLFLOW_TRACKING_URI env var overrides config (used inside Docker)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{config['mlflow']['model_name']}/latest"
    try:
        model_store["model"] = mlflow.xgboost.load_model(model_uri)
        model_store["config"] = config
        print(f"Model loaded from MLflow: {model_uri}")
    except Exception as e:
        print(f"WARNING: Could not load model from MLflow: {e}")
        print("Start the MLflow server and register a model first.")
    yield
    # Cleanup on shutdown (nothing needed here)


app = FastAPI(
    title="ChurnOps — Customer Churn Prediction API",
    description="Predicts whether a telecom customer will churn using XGBoost.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check if the API and model are ready."""
    config = model_store.get("config") or load_config()
    return HealthResponse(
        status="ok",
        model_loaded=model_store["model"] is not None,
        mlflow_uri=config["mlflow"]["tracking_uri"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    """
    Predict churn for a single customer.

    Send customer features → get back churn probability and risk level.
    """
    if model_store["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Ensure MLflow server is running and a model is registered.",
        )

    config = model_store["config"]
    threshold = config["model"]["threshold"]

    # Encode input
    X = encode_input(customer)

    # Predict
    prob = float(model_store["model"].predict_proba(X)[0][1])
    prediction = int(prob >= threshold)

    # Risk level
    if prob < 0.3:
        risk = "Low"
    elif prob < 0.6:
        risk = "Medium"
    else:
        risk = "High"

    return PredictionResponse(
        churn_prediction=prediction,
        churn_probability=round(prob, 4),
        risk_level=risk,
    )
