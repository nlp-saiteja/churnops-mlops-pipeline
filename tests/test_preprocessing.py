import pandas as pd
import numpy as np
import pytest
from src.data.preprocess import clean_data, encode_features


def make_raw_sample():
    """Creates a minimal raw dataset matching the Telco CSV schema."""
    return pd.DataFrame({
        "customerID": ["111-AAA", "222-BBB", "333-CCC"],
        "gender": ["Male", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0],
        "Partner": ["Yes", "No", "Yes"],
        "Dependents": ["No", "No", "Yes"],
        "tenure": [12, 0, 60],
        "PhoneService": ["Yes", "No", "Yes"],
        "MultipleLines": ["No", "No phone service", "Yes"],
        "InternetService": ["Fiber optic", "DSL", "No"],
        "OnlineSecurity": ["No", "Yes", "No internet service"],
        "OnlineBackup": ["Yes", "No", "No internet service"],
        "DeviceProtection": ["No", "Yes", "No internet service"],
        "TechSupport": ["No", "No", "No internet service"],
        "StreamingTV": ["Yes", "No", "No internet service"],
        "StreamingMovies": ["Yes", "No", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No", "Yes"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
        "MonthlyCharges": [85.5, 45.0, 20.0],
        "TotalCharges": ["1026.0", " ", "1200.0"],   # space simulates new customer
        "Churn": ["Yes", "No", "No"],
    })


def test_clean_data_drops_customer_id():
    df = make_raw_sample()
    cleaned = clean_data(df)
    assert "customerID" not in cleaned.columns


def test_clean_data_converts_total_charges():
    df = make_raw_sample()
    cleaned = clean_data(df)
    # The space value should become 0.0
    assert cleaned["TotalCharges"].dtype in [float, np.float64]
    assert cleaned["TotalCharges"].isna().sum() == 0


def test_clean_data_encodes_churn():
    df = make_raw_sample()
    cleaned = clean_data(df)
    assert set(cleaned["Churn"].unique()).issubset({0, 1})


def test_encode_features_binary_columns():
    df = make_raw_sample()
    cleaned = clean_data(df)
    encoded = encode_features(cleaned)
    assert encoded["Partner"].isin([0, 1]).all()
    assert encoded["PhoneService"].isin([0, 1]).all()
    assert encoded["gender"].isin([0, 1]).all()


def test_encode_features_one_hot():
    df = make_raw_sample()
    cleaned = clean_data(df)
    encoded = encode_features(cleaned)
    # After drop_first, DSL/Month-to-month/Bank transfer are baseline (not present)
    assert "InternetService_Fiber optic" in encoded.columns
    assert "Contract_One year" in encoded.columns
    assert "PaymentMethod_Electronic check" in encoded.columns
    # Original columns should be gone
    assert "InternetService" not in encoded.columns
    assert "Contract" not in encoded.columns


def test_encode_features_no_nulls():
    df = make_raw_sample()
    cleaned = clean_data(df)
    encoded = encode_features(cleaned)
    assert encoded.isna().sum().sum() == 0


def test_encode_features_column_count():
    df = make_raw_sample()
    # Add a 4th row with the missing PaymentMethod to get full one-hot encoding
    extra = df.iloc[[0]].copy()
    extra["PaymentMethod"] = "Credit card (automatic)"
    df = pd.concat([df, extra], ignore_index=True)
    cleaned = clean_data(df)
    encoded = encode_features(cleaned)
    # Should have exactly 24 columns (23 features + 1 target)
    assert encoded.shape[1] == 24
