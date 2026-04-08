import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np


@pytest.fixture
def mock_model():
    """Mock XGBoost model that returns a fixed probability."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.3, 0.7]])
    return model


@pytest.fixture
def client(mock_model):
    """Creates a test client with the model pre-loaded (no MLflow needed)."""
    from src.api.main import app, model_store
    model_store["model"] = mock_model
    model_store["config"] = {
        "mlflow": {"tracking_uri": "http://localhost:5001", "model_name": "churn-model"},
        "model": {"threshold": 0.5},
    }
    return TestClient(app)


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_returns_correct_fields(client, sample_customer):
    response = client.post("/predict", json=sample_customer)
    assert response.status_code == 200
    data = response.json()
    assert "churn_prediction" in data
    assert "churn_probability" in data
    assert "risk_level" in data


def test_predict_churn_prediction_is_binary(client, sample_customer):
    response = client.post("/predict", json=sample_customer)
    assert response.json()["churn_prediction"] in [0, 1]


def test_predict_probability_in_range(client, sample_customer):
    response = client.post("/predict", json=sample_customer)
    prob = response.json()["churn_probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_high_risk(client, sample_customer):
    # Mock returns 0.7 probability → should be High risk
    response = client.post("/predict", json=sample_customer)
    data = response.json()
    assert data["churn_prediction"] == 1
    assert data["risk_level"] == "High"


def test_predict_invalid_input(client):
    """Should return 422 validation error for bad input."""
    bad_payload = {"gender": "Unknown", "tenure": -5}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_no_model(sample_customer):
    """Should return 503 when model is not loaded."""
    from src.api.main import app, model_store
    model_store["model"] = None
    client = TestClient(app)
    response = client.post("/predict", json=sample_customer)
    assert response.status_code == 503
