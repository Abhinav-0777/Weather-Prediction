# 🌦️ Weather Prediction

A Machine Learning project that predicts **whether it will rain tomorrow**, using historical weather data from various cities in Australia.

The project covers the full pipeline — data preprocessing, model training, evaluation, and automated model monitoring (data drift, prediction drift, and performance tracking) to keep track of how the model behaves after deployment.

---

## Features

- Predicts `RainTomorrow` (Yes/No) using historical weather observations
- End-to-end pipeline: data transformation → model training → evaluation
- Automated model monitoring using [Evidently AI](https://www.evidentlyai.com/)
- Config-driven setup (model version, paths, etc. via `config.yaml`)
- Unit and integration tests using `pytest`
- Dockerized for easy setup

---

## Tech Stack

- **Python**
- **scikit-learn** — model training
- **Evidently AI** — model monitoring (drift & performance)
- **Supabase (Postgres)** — prediction logging
- **pytest** — testing
- **Docker**

---

## Installation

### Prerequisites
- Python 3.10+
- Git

### Steps

1. **Clone the repository**
```bash
git clone https://github.com/Abhinav-0777/Weather-Prediction.git
cd Weather-Prediction
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up configuration**

Update `config.yaml` with your model version and other settings:
```yaml
model_version: v1
date_format: "%Y-%m-%d_%H-%M-%S"
```

5. **Run the training pipeline**
```bash
python src/components/data_transformation.py
python src/components/model_trainer.py
```

---

## Running with Docker

```bash
docker build -t weather-prediction .
docker run weather-prediction
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
Weather-Prediction/
├── artifacts/          # Trained model and preprocessor files
├── data/                # Raw dataset
├── src/                 # Source code (training, monitoring, utils)
├── tests/                # Unit and integration tests
├── config.yaml           # Project configuration
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## License

This project is open source and available under the [MIT License](LICENSE).
