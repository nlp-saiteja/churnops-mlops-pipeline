import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
import os


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop customerID — it's just an identifier, not useful for prediction
    df = df.drop(columns=["customerID"])

    # TotalCharges has spaces for new customers (tenure=0) — convert to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Convert target: Yes → 1, No → 0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    # Binary Yes/No columns → 1/0
    binary_cols = [
        "Partner", "Dependents", "PhoneService", "PaperlessBilling",
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0, "No phone service": 0, "No internet service": 0})

    # Gender → binary
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    # Multi-category columns → one-hot encoding (drop_first avoids multicollinearity)
    multi_cols = ["InternetService", "Contract", "PaymentMethod"]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    return df


def split_and_save(df: pd.DataFrame, config: dict) -> tuple:
    target = config["data"]["target_column"]
    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]
    processed_path = config["data"]["processed_path"]

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Save processed splits
    os.makedirs(processed_path, exist_ok=True)
    X_train.to_csv(f"{processed_path}/X_train.csv", index=False)
    X_test.to_csv(f"{processed_path}/X_test.csv", index=False)
    y_train.to_csv(f"{processed_path}/y_train.csv", index=False)
    y_test.to_csv(f"{processed_path}/y_test.csv", index=False)

    print(f"Train set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")
    print(f"Churn rate in train: {y_train.mean():.2%} | test: {y_test.mean():.2%}")
    print(f"Saved processed data to {processed_path}")

    return X_train, X_test, y_train, y_test


def run_preprocessing():
    config = load_config()
    df = load_raw_data(config["data"]["raw_path"])
    df = clean_data(df)
    df = encode_features(df)
    X_train, X_test, y_train, y_test = split_and_save(df, config)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run_preprocessing()
