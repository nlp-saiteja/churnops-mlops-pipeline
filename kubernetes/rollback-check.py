"""
Automated rollback script.

Checks the canary deployment's model metrics against thresholds.
If performance drops below threshold → scale canary to 0 (rollback).
If performance is good → promote canary to stable (full rollout).

Usage:
    python kubernetes/rollback-check.py
"""

import json
import subprocess
import sys
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_metrics(metrics_path: str = "models/metrics.json") -> dict:
    with open(metrics_path, "r") as f:
        return json.load(f)


def scale_deployment(name: str, namespace: str, replicas: int):
    cmd = [
        "kubectl", "scale", "deployment", name,
        f"--replicas={replicas}",
        f"--namespace={namespace}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Scaled {name} to {replicas} replicas")
    else:
        print(f"ERROR scaling {name}: {result.stderr}")


def check_and_rollback():
    config = load_config()
    thresholds = config["monitoring"]["performance_threshold"]

    print("--- Canary Rollback Check ---")

    try:
        metrics = load_metrics()
    except FileNotFoundError:
        print("No metrics file found. Skipping check.")
        sys.exit(0)

    roc_auc = metrics.get("roc_auc", 0)
    f1 = metrics.get("f1_score", 0)
    min_roc_auc = thresholds["roc_auc"]
    min_f1 = thresholds["f1_score"]

    print(f"Current metrics  → ROC-AUC: {roc_auc} | F1: {f1}")
    print(f"Minimum required → ROC-AUC: {min_roc_auc} | F1: {min_f1}")

    if roc_auc < min_roc_auc or f1 < min_f1:
        print("\nPERFORMANCE BELOW THRESHOLD — rolling back canary!")
        scale_deployment("churnops-api-canary", "churnops", 0)
        print("Rollback complete. All traffic back to stable.")
        sys.exit(1)
    else:
        print("\nPerformance OK — promoting canary to full stable!")
        # Scale stable down first, then canary up to 9
        scale_deployment("churnops-api-canary", "churnops", 9)
        scale_deployment("churnops-api-stable", "churnops", 0)
        print("Promotion complete. Canary is now the new stable.")
        sys.exit(0)


if __name__ == "__main__":
    check_and_rollback()
