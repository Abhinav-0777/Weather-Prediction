# 🌦️ Weather Prediction

![Sydney Opera House](./assets/sydney-opera-house.jpg)

A Machine Learning project that predicts **whether it will rain tomorrow**, using historical weather data from various cities in Australia — served through a fully async, modular FastAPI backend and deployed on Render.

The project covers the full pipeline — data ingestion, transformation, model training, evaluation, orchestration via Airflow, experiment tracking via MLflow, model explainability, and automated model monitoring (data drift, prediction drift, and performance tracking) to keep track of how the model behaves after deployment.

---

## Live Demo

🔗 **[Live App](https://weather-actions-latest.onrender.com/)**
🔗 **[API Docs (Swagger)](https://weather-actions-latest.onrender.com/docs)**

---

## Features

- Predicts `RainTomorrow` (Yes/No) using historical weather observations
- End-to-end pipeline: data ingestion → transformation → model training → evaluation
- Orchestrated as an **Apache Airflow DAG** with retries and task dependencies
- **MLflow** experiment tracking — nested runs per model, hyperparameter logging, model registry, and evaluation-to-training run linkage
- **Model explainability** for interpreting predictions
- **Async FastAPI** backend, fully modular project structure
- **Redis caching** (including cached Unsplash imagery) for faster response times
- Automated model monitoring using Evidently AI (https://www.evidentlyai.com/)
- Prediction logging via **Supabase (Postgres)**
- **Load tested using Locust** to validate API performance under concurrent traffic
- Config-driven setup (model version, paths, etc. via config.yaml)
- Unit and integration tests using pytest
- **CI/CD pipeline** via GitHub Actions (lint, test, Docker build & push) with continuous deployment to Render
- Dockerized for easy setup

---

## Tech Stack

- Python
- scikit-learn — model training
- FastAPI — async web backend
- Apache Airflow — pipeline orchestration
- MLflow — experiment tracking & model registry
- Evidently AI — model monitoring (drift & performance)
- Redis — caching layer
- Unsplash API — imagery, served via Redis cache
- Supabase (Postgres) — prediction logging
- pytest — testing
- Locust — load testing
- Docker
- Render — deployment

---

## Installation

### Prerequisites
- Python 3.10+
- Git

### Steps

1. Clone the repository

   git clone https://github.com/Abhinav-0777/Weather-Prediction.git
   cd Weather-Prediction

2. Create a virtual environment

   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate

3. Install dependencies

   pip install -r requirements.txt

4. Set up configuration

   Update config.yaml with your model version and other settings:

   model_version: v1
   date_format: "%Y-%m-%d_%H-%M-%S"

---

## Running the Pipeline

### Option 1 — Run components individually

   python -m src.components.data_ingestion
   python -m src.components.data_transformation
   python -m src.components.model_trainer
   python -m src.components.model_evaluation

### Option 2 — Run via Airflow DAG (recommended)

   export AIRFLOW_HOME=~/airflow
   export PYTHONPATH=.
   airflow db migrate
   airflow dags test model_pipeline $(date +%F)

---

## Running the API

   uvicorn src.app:app --reload

FastAPI's interactive docs will be available at `/docs`.

---

## Experiment Tracking (MLflow)

All training runs are logged under the weather_prediction_classification experiment — including per-model hyperparameter search results, the best model's parameters and F2-score, and the registered model in MLflow's Model Registry (weather_prediction_classifier).

To view the MLflow UI locally:

   mlflow ui --backend-store-uri file:///path/to/mlruns

---

## Running with Docker

   docker build -t weather-prediction .
   docker run weather-prediction

---

## Running Tests

   pytest tests/ -v

With coverage:

   pytest --cov=src --cov-report=term-missing tests/

---

## Load Testing

API performance under concurrent load is validated using Locust.

   locust -f locustfile.py --host=http://localhost:8000

Open `http://localhost:8089` to configure and run load test scenarios.

---

## Project Structure

Weather-Prediction/
├── artifacts/           # Trained model, preprocessor, and metrics
├── dags/                # Airflow DAG definitions
├── mlruns/              # MLflow tracking data (local runs)
├── src/
│   ├── app/              # FastAPI application (routes, dependencies)
│   ├── components/      # Data ingestion, transformation, trainer, evaluation
│   ├── pipeline/         # Training and monitoring pipeline orchestration
│   ├── monitoring/       # Drift/performance baselines
│   ├── database/         # Prediction logging (Supabase)
│   ├── cache/             # Redis caching layer
│   └── utils.py
├── tests/                # Unit and integration tests
├── locustfile.py         # Load testing scenarios
├── config.yaml           # Project configuration
├── requirements.txt
├── Dockerfile
└── README.md

---

## CI/CD

On every push/PR to main, GitHub Actions runs linting (ruff), the test suite (pytest), and builds/pushes a Docker image to Docker Hub. Merges to main trigger continuous deployment to Render.

---

## Deployment

The application is deployed on Render.

🔗 **Live App:** https://weather-actions-latest.onrender.com/

---

## License

This project is open source and available under the MIT License (LICENSE).