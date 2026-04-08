# ChurnOps — End-to-End MLOps Pipeline for Customer Churn Prediction

A fully automated MLOps pipeline that trains, serves, monitors, and continuously retrains an XGBoost model for predicting customer churn.

## Architecture

```
Data Ingestion → Preprocessing → XGBoost Training → MLflow Registry
                                                          ↓
GitHub Actions CI/CD ← Drift Detected ← Evidently Monitoring ← FastAPI (Kubernetes/Minikube)
```

## Tech Stack

| Layer | Tool |
|-------|------|
| ML Model | XGBoost |
| Experiment Tracking | MLflow |
| Model Registry | MLflow Model Registry |
| API Serving | FastAPI |
| Containerization | Docker (OrbStack) |
| Orchestration | Kubernetes (Minikube) |
| Drift Detection | Evidently AI |
| CI/CD | GitHub Actions |

## Project Structure

```
churnops-mlops-pipeline/
├── configs/           # Configuration files
├── data/
│   ├── raw/           # Original dataset
│   └── processed/     # Cleaned & feature-engineered data
├── src/
│   ├── data/          # Data ingestion & preprocessing
│   ├── models/        # Training, evaluation, registry
│   ├── api/           # FastAPI app
│   └── monitoring/    # Drift detection & alerting
├── tests/             # Unit & integration tests
├── docker/            # Dockerfile & compose files
├── kubernetes/        # K8s manifests (deployment, service, canary)
├── .github/workflows/ # CI/CD pipelines
├── models/            # Saved model artifacts
└── notebooks/         # Exploratory analysis
```

## Quickstart

```bash
# 1. Clone and enter project
git clone <repo-url>
cd churnops-mlops-pipeline

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start MLflow tracking server
mlflow server --host 0.0.0.0 --port 5000

# 5. Train model
python src/models/train.py

# 6. Start API
uvicorn src.api.main:app --reload --port 8000
```

## Features

- Automated model retraining via GitHub Actions
- Canary deployment strategy with automated rollback
- Real-time data drift detection with Evidently AI
- Full experiment lineage tracked in MLflow
- Zero paid cloud services — runs entirely on local Minikube
