# ChurnOps — End-to-End MLOps Pipeline for Customer Churn Prediction

By Leela Phanidhar Sai Teja Nalanagula 

![Python](https://img.shields.io/badge/Python-3.12-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green)
![Docker](https://img.shields.io/badge/Container-Docker-blue)
![Kubernetes](https://img.shields.io/badge/Orchestration-Kubernetes-326CE5)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

**ChurnOps** is a fully automated, production-grade MLOps pipeline built to predict customer churn for a telecom company. The system trains an XGBoost classifier, tracks every experiment with MLflow, serves predictions via a FastAPI REST API, containerizes the app with Docker, deploys to a Kubernetes cluster with canary release strategies, monitors for data drift using Evidently AI, and automates the entire lifecycle with GitHub Actions CI/CD — all running locally with zero cloud costs.

> Built as a demonstration of end-to-end MLOps engineering — from raw data to automated production deployment.

---

## Table of Contents

- [Architecture](#architecture)
- [Model Performance](#model-performance)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Phase Breakdown](#phase-breakdown)
  - [Phase 1 — Data & Setup](#phase-1--data--setup)
  - [Phase 2 — Model Training & MLflow](#phase-2--model-training--mlflow)
  - [Phase 3 — FastAPI Serving](#phase-3--fastapi-serving)
  - [Phase 4 — Docker](#phase-4--docker)
  - [Phase 5 — Kubernetes & Canary Deployment](#phase-5--kubernetes--canary-deployment)
  - [Phase 6 — Drift Detection](#phase-6--drift-detection)
  - [Phase 7 — GitHub Actions CI/CD](#phase-7--github-actions-cicd)
- [Quickstart](#quickstart)
- [API Reference](#api-reference)
- [CI/CD Pipeline](#cicd-pipeline)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│  Kaggle Telco Dataset → Preprocessing → Train/Test Split        │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                      TRAINING LAYER                             │
│  XGBoost Classifier → MLflow Experiment Tracking                │
│                     → MLflow Model Registry (versioned)         │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                       SERVING LAYER                             │
│  FastAPI REST API → Docker Container → Kubernetes (Minikube)    │
│                                     → Canary Deployment (10%)   │
│                                     → Auto Rollback on Failure  │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                     MONITORING LAYER                            │
│  Evidently AI → Feature Drift Detection → HTML Reports          │
│              → Alert if drift > 10% of features                 │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                       CI/CD LAYER                               │
│  GitHub Actions → Run Tests → Retrain → Build Docker → Deploy   │
│                → Scheduled Weekly Retraining (every Sunday)     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Model Performance

Trained on the **Telco Customer Churn dataset** (7,043 customers, 20 features).

| Metric | Score | Notes |
|--------|-------|-------|
| **ROC-AUC** | **0.8303** | Primary metric — measures ranking ability |
| **Recall** | **0.7032** | 70% of actual churners caught |
| **Precision** | **0.5389** | 54% of predicted churners are real |
| **F1 Score** | **0.6102** | Harmonic mean of precision & recall |
| **Accuracy** | **0.7615** | Overall correctness |

> **Why recall over accuracy?** In churn prediction, missing a real churner (false negative) costs more than a false alarm. The model is optimized for recall using `scale_pos_weight=2.77` to handle class imbalance (73% non-churn, 27% churn).

---

## Tech Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| Language | Python 3.12 | Core language |
| ML Model | XGBoost 2.0 | Gradient boosted classifier |
| Experiment Tracking | MLflow 2.16 | Logs params, metrics, artifacts |
| Model Registry | MLflow Model Registry | Versioned model promotion |
| API Framework | FastAPI | REST API with auto docs |
| Data Validation | Pydantic v2 | Input schema validation |
| Containerization | Docker + OrbStack | Portable deployment |
| Orchestration | Kubernetes + Minikube | Local K8s cluster |
| Drift Detection | Evidently AI 0.4 | Feature distribution monitoring |
| CI/CD | GitHub Actions | Automated test, retrain, deploy |
| Testing | pytest | 14 unit & integration tests |

---

## Project Structure

```
churnops-mlops-pipeline/
│
├── .github/
│   └── workflows/
│       ├── ci.yml              # Runs on every push — test, retrain, build
│       └── retrain.yml         # Scheduled weekly retraining + manual trigger
│
├── configs/
│   └── config.yaml             # Central config: model params, thresholds, paths
│
├── data/
│   ├── raw/                    # Original Kaggle CSV (gitignored)
│   └── processed/              # Train/test splits after preprocessing (gitignored)
│
├── docker/
│   ├── Dockerfile              # FastAPI container image
│   └── docker-compose.yml      # Runs MLflow + API together
│
├── kubernetes/
│   ├── namespace.yaml          # Isolated K8s namespace: churnops
│   ├── mlflow-deployment.yaml  # MLflow server in Kubernetes
│   ├── api-deployment.yaml     # Stable API deployment (9 replicas = 90% traffic)
│   ├── canary-deployment.yaml  # Canary deployment (1 replica = 10% traffic)
│   └── rollback-check.py       # Auto-rollback script based on metric thresholds
│
├── src/
│   ├── data/
│   │   └── preprocess.py       # Load, clean, encode, split dataset
│   ├── models/
│   │   ├── train.py            # XGBoost training + MLflow logging
│   │   └── evaluate.py         # Loads registered model, prints metrics
│   ├── api/
│   │   ├── main.py             # FastAPI app: /health and /predict endpoints
│   │   └── schemas.py          # Pydantic input/output schemas
│   └── monitoring/
│       ├── drift_detector.py   # Runs Evidently drift report
│       └── simulate_drift.py   # Simulates production data with injected drift
│
├── tests/
│   ├── conftest.py             # Shared fixtures (synthetic data, mock model)
│   ├── test_preprocessing.py   # 7 tests for data cleaning & encoding
│   └── test_api.py             # 7 tests for API endpoints
│
├── models/                     # Saved model artifacts (gitignored)
├── reports/                    # Evidently HTML drift reports (gitignored)
├── notebooks/                  # Exploratory analysis
├── .dockerignore               # Excludes venv, data, .claude from Docker builds
├── .gitignore                  # Excludes venv, data, secrets, .claude
├── requirements.txt            # All Python dependencies
└── README.md                   # This file
```

---

## Phase Breakdown

### Phase 1 — Data & Setup

- **Dataset:** Telco Customer Churn (Kaggle) — 7,043 rows, 21 columns
- **Preprocessing steps:**
  - Dropped `customerID` (non-predictive identifier)
  - Fixed `TotalCharges` column (whitespace values for new customers → 0.0)
  - Encoded binary columns (`Yes`/`No` → `1`/`0`)
  - One-hot encoded multi-category columns: `InternetService`, `Contract`, `PaymentMethod`
  - Stratified 80/20 train/test split

---

### Phase 2 — Model Training & MLflow

- **Algorithm:** XGBoost with `scale_pos_weight=2.77` for class imbalance
- **Every training run automatically logs to MLflow:**
  - Hyperparameters: `n_estimators`, `max_depth`, `learning_rate`, etc.
  - Metrics: ROC-AUC, F1, precision, recall, accuracy
  - The trained model artifact (registered in MLflow Model Registry)
- **MLflow UI** available at `http://localhost:5001` — compare runs, visualize metrics

---

### Phase 3 — FastAPI Serving

- REST API with two endpoints:
  - `GET /health` — verifies model is loaded and ready
  - `POST /predict` — accepts customer features, returns churn prediction
- Auto-generated interactive documentation at `/docs`
- Pydantic schemas enforce strict input validation (wrong types → 422 error)
- Model loaded from MLflow Registry on API startup

**Example request:**
```json
{
  "tenure": 12,
  "Contract": "Month-to-month",
  "InternetService": "Fiber optic",
  "MonthlyCharges": 85.5,
  "PaymentMethod": "Electronic check"
}
```

**Example response:**
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.5689,
  "risk_level": "Medium"
}
```

---

### Phase 4 — Docker

- `Dockerfile` packages the FastAPI app with all dependencies into a portable image
- `docker-compose.yml` runs two services together:
  - `mlflow` — tracking server on port 5001
  - `api` — prediction API on port 8000
- Services communicate via Docker network using service names (`http://mlflow:5001`)
- Health checks ensure API only starts after MLflow is ready

```bash
cd docker && docker compose up --build
```

---

### Phase 5 — Kubernetes & Canary Deployment

Deployed to a local Kubernetes cluster using **Minikube**.

**Canary release strategy:**
```
Total 10 pods:
  ├── 9 pods → stable version  (90% traffic)
  └── 1 pod  → canary version  (10% traffic)
```

**Automated rollback** (`kubernetes/rollback-check.py`):
- Reads model metrics from `models/metrics.json`
- If `ROC-AUC < 0.75` or `F1 < 0.55` → scales canary to 0, all traffic back to stable
- If metrics pass → promotes canary to 9 replicas, retires stable

```bash
# Deploy canary
kubectl apply -f kubernetes/canary-deployment.yaml -n churnops

# Run rollback check
python kubernetes/rollback-check.py
```

---

### Phase 6 — Drift Detection

Built with **Evidently AI** to detect when production data shifts away from training data.

**Two scenarios tested:**

| Scenario | Features Drifted | Alert Triggered |
|----------|-----------------|-----------------|
| Injected drift (shorter tenure, higher charges, more Fiber optic) | 6 / 23 (26.1%) | YES |
| Stable production data | 0 / 23 (0%) | NO |

Generates a full **HTML dashboard report** with:
- Per-feature distribution comparison (training vs production)
- Statistical drift scores per column
- Data quality metrics (missing values, outliers)

```bash
# Run with drift
PYTHONPATH=. python src/monitoring/drift_detector.py

# Run without drift (baseline)
PYTHONPATH=. python src/monitoring/drift_detector.py --no-drift
```

---

### Phase 7 — GitHub Actions CI/CD

Two automated workflows:

**`ci.yml` — triggers on every push to `main`:**
```
Push to main
    ├── Job 1: Run Tests       (pytest — 14 tests)
    ├── Job 2: Retrain Model   (download data → preprocess → train → drift check)
    └── Job 3: Build Docker    (build image → verify startup)
```

**`retrain.yml` — runs every Sunday at midnight (or manually triggered):**
```
Scheduled trigger
    ├── Download latest Kaggle dataset
    ├── Preprocess & retrain model
    ├── Validate: ROC-AUC must be ≥ 0.75
    ├── Run drift detection report
    └── Upload reports as GitHub artifacts
```

All 3 CI checks pass on every push.

---

## Quickstart

```bash
# 1. Clone the repository
git clone https://github.com/saitejanlp/churnops-mlops-pipeline.git
cd churnops-mlops-pipeline

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip setuptools
pip install -r requirements.txt

# 4. Download dataset (Kaggle account required)
# Place telco_churn.csv in data/raw/telco_churn.csv

# 5. Preprocess data
PYTHONPATH=. python src/data/preprocess.py

# 6. Start MLflow server (in a separate terminal)
mlflow server --host 0.0.0.0 --port 5001

# 7. Train model
PYTHONPATH=. python src/models/train.py

# 8. Start API
python -m uvicorn src.api.main:app --reload --port 8000

# 9. Open API docs
open http://localhost:8000/docs

# 10. Open MLflow UI
open http://localhost:5001
```

### Run with Docker

```bash
cd docker
docker compose up --build
# API: http://localhost:8000/docs
# MLflow: http://localhost:5001
```

### Run on Kubernetes

```bash
# Start Minikube
minikube start --driver=docker --memory=3500 --cpus=2

# Load image
docker tag docker-api:latest churnops-api:latest
minikube image load churnops-api:latest

# Deploy
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/mlflow-deployment.yaml
kubectl apply -f kubernetes/api-deployment.yaml

# Access
kubectl port-forward service/churnops-api 9000:80 -n churnops
open http://localhost:9000/docs
```

### Run Tests

```bash
PYTHONPATH=. pytest tests/ -v
# 14 passed
```

---

## API Reference

### `GET /health`
Returns the API and model status.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "mlflow_uri": "http://localhost:5001"
}
```

### `POST /predict`
Predicts churn probability for a customer.

**Request body:** Full customer feature set (see `src/api/schemas.py` for all fields)

**Response:**
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.5689,
  "risk_level": "High"
}
```

| `risk_level` | Probability range |
|-------------|------------------|
| Low | < 0.30 |
| Medium | 0.30 – 0.60 |
| High | > 0.60 |

---

## CI/CD Pipeline

```
┌─────────────────────────────────────────────────┐
│             GitHub Actions (ci.yml)             │
│                                                 │
│  on: push to main                               │
│                                                 │
│  ┌──────────┐   ┌──────────┐   ┌─────────────┐ │
│  │   test   │──▶│ retrain  │   │build docker │ │
│  │          │   │          │   │             │ │
│  │ 14 tests │   │ kaggle   │   │ Dockerfile  │ │
│  │  passed  │   │ → train  │   │ + verify    │ │
│  │          │   │ → drift  │   │             │ │
│  └──────────┘   └──────────┘   └─────────────┘ │
└─────────────────────────────────────────────────┘
```

---

