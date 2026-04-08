"""
Drift Detection using Evidently AI.

Compares reference data (training set) against current production data.
Generates an HTML report and prints a summary of drifted features.

Usage:
    python src/monitoring/drift_detector.py           # with drift
    python src/monitoring/drift_detector.py --no-drift # without drift
"""

import argparse
import os
import pandas as pd
import yaml
from datetime import datetime

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import DatasetDriftMetric, DatasetMissingValuesMetric

from src.monitoring.simulate_drift import simulate_production_data


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_drift_detection(drift: bool = True):
    config = load_config()
    processed_path = config["data"]["processed_path"]
    drift_threshold = config["monitoring"]["drift_threshold"]

    # Reference = training data (what the model learned from)
    reference_data = pd.read_csv(f"{processed_path}/X_train.csv")

    # Current = simulated production data
    current_data = simulate_production_data(drift=drift)

    print(f"\nReference data shape: {reference_data.shape}")
    print(f"Current data shape:   {current_data.shape}")

    # Build Evidently report with two presets:
    # 1. DataDriftPreset   → checks if feature distributions have shifted
    # 2. DataQualityPreset → checks for missing values, outliers, etc.
    report = Report(metrics=[
        DatasetDriftMetric(drift_share=drift_threshold),  # trigger if X% of features drift
        DatasetMissingValuesMetric(),
        DataDriftPreset(),
        DataQualityPreset(),
    ])

    report.run(reference_data=reference_data, current_data=current_data)

    # Save HTML report
    os.makedirs("reports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    drift_label = "drifted" if drift else "stable"
    report_path = f"reports/drift_report_{drift_label}_{timestamp}.html"
    report.save_html(report_path)
    print(f"\nReport saved → {report_path}")

    # Extract drift summary from the report
    result = report.as_dict()
    drift_metric = result["metrics"][0]["result"]

    n_drifted = drift_metric.get("number_of_drifted_columns", 0)
    n_total = drift_metric.get("number_of_columns", 0)
    dataset_drifted = drift_metric.get("dataset_drift", False)
    share_drifted = drift_metric.get("share_of_drifted_columns", 0)

    print("\n--- Drift Detection Summary ---")
    print(f"  Total features checked : {n_total}")
    print(f"  Features drifted       : {n_drifted}")
    print(f"  Share drifted          : {share_drifted:.1%}")
    print(f"  Dataset drift detected : {'YES ⚠️' if dataset_drifted else 'NO ✅'}")

    if dataset_drifted:
        print(f"\n⚠️  ALERT: Drift detected in {n_drifted} features!")
        print("   Consider retraining the model.")
    else:
        print("\n✅ No significant drift detected. Model is healthy.")

    print(f"\nOpen in browser: {report_path}")
    return dataset_drifted, report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-drift", action="store_true", help="Simulate stable data")
    args = parser.parse_args()

    run_drift_detection(drift=not args.no_drift)
