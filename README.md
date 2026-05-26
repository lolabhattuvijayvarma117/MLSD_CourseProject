# IEEE-CIS Fraud Detection MLOps Pipeline

This repository implements a production-grade machine learning pipeline for credit card fraud detection using the IEEE-CIS dataset.

## Project Structure
* `model_training/` - Python scripts for model training, metrics evaluation, and MLflow logging.
* `fraud_detection_service/` - FastAPI application serving predictions and health status check.
* `streamlit_app/` - Streamlit application providing interactive transaction evaluation dashboard.
* `docker/` - Docker and Docker Compose setup for packaging the application.
* `kubernetes/` - Kubernetes manifests for multi-replica, self-healing deployments.
* `tests/` - Pytest suite for training validation and API endpoints checks.
* `scripts/` - Shell utilities for local execution.
