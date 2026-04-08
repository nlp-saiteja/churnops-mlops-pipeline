"""
Shared test fixtures — creates synthetic data matching the real dataset schema.
This means tests run in CI without needing the Kaggle dataset.
"""

import pytest
import pandas as pd
import numpy as np
import os


@pytest.fixture(scope="session")
def synthetic_processed_data():
    """
    Creates small synthetic train/test splits matching our processed schema.
    Saves them to data/processed/ so preprocessing and training code can find them.
    """
    np.random.seed(42)
    n = 200

    df = pd.DataFrame({
        "gender": np.random.randint(0, 2, n),
        "SeniorCitizen": np.random.randint(0, 2, n),
        "Partner": np.random.randint(0, 2, n),
        "Dependents": np.random.randint(0, 2, n),
        "tenure": np.random.randint(0, 72, n),
        "PhoneService": np.random.randint(0, 2, n),
        "MultipleLines": np.random.randint(0, 2, n),
        "OnlineSecurity": np.random.randint(0, 2, n),
        "OnlineBackup": np.random.randint(0, 2, n),
        "DeviceProtection": np.random.randint(0, 2, n),
        "TechSupport": np.random.randint(0, 2, n),
        "StreamingTV": np.random.randint(0, 2, n),
        "StreamingMovies": np.random.randint(0, 2, n),
        "PaperlessBilling": np.random.randint(0, 2, n),
        "MonthlyCharges": np.random.uniform(20, 110, n),
        "TotalCharges": np.random.uniform(0, 8000, n),
        "InternetService_Fiber optic": np.random.randint(0, 2, n),
        "InternetService_No": np.random.randint(0, 2, n),
        "Contract_One year": np.random.randint(0, 2, n),
        "Contract_Two year": np.random.randint(0, 2, n),
        "PaymentMethod_Credit card (automatic)": np.random.randint(0, 2, n),
        "PaymentMethod_Electronic check": np.random.randint(0, 2, n),
        "PaymentMethod_Mailed check": np.random.randint(0, 2, n),
    })

    target = pd.Series(np.random.randint(0, 2, n), name="Churn")

    # Split 80/20
    split = int(n * 0.8)
    X_train = df.iloc[:split]
    X_test = df.iloc[split:]
    y_train = target.iloc[:split]
    y_test = target.iloc[split:]

    # Save to data/processed/
    os.makedirs("data/processed", exist_ok=True)
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="session")
def sample_customer():
    """A sample customer payload for API testing."""
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5,
        "TotalCharges": 1024.5,
    }
