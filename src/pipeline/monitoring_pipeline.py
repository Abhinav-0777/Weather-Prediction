import os

from src.database.connection import SessionLocal
from src.database.models import ModelPredictionLog
from src.monitoring.evidently.monitor_confidence_score import plotting_confidence_score
from src.monitoring.evidently.monitor_data_drift import plotting_data_drift
from src.monitoring.evidently.monitor_prediction_drift import plotting_predictions_drift
from src.monitoring.evidently.monitor_prediction_performance import (
    plotting_predictions_performance,
)
from src.utils import load_config

config = load_config()

def run_monitor_confidence_score():

    confidence_score_baseline_path = "src/monitoring/baselines/confidence_score_baseline_v1.csv"

    plotting_confidence_score(
        session=SessionLocal,
        table_name=ModelPredictionLog,
        reference_path=confidence_score_baseline_path,
        model_version=config["model_version"],
        date_format=config["date_format"]
    )


def run_monitor_data_drift():

    data_drift_baseline_path = os.path.join("src/monitoring/baselines", f"data_drift_baseline_{config['model_version']}.npy")

    plotting_data_drift(
        session=SessionLocal,
        table_name=ModelPredictionLog,
        reference_path=data_drift_baseline_path,
        model_version=config["model_version"],
        date_format=config["date_format"]
    )


def run_monitor_prediction_drift():

    prediction_drift_baseline_path = os.path.join("src/monitoring/baselines", f"predictions_baseline_{config['model_version']}.csv")

    plotting_predictions_drift(
        session=SessionLocal,
        table_name=ModelPredictionLog,
        reference_path=prediction_drift_baseline_path,
        model_version=config["model_version"],
        date_format=config["date_format"]
    )


def run_monitor_prediction_performance():

    prediction_performance_baseline_path = os.path.join("src/monitoring/baselines", f"predictions_baseline_{config['model_version']}.csv")

    plotting_predictions_performance(
        session=SessionLocal,
        table_name=ModelPredictionLog,
        reference_path=prediction_performance_baseline_path,
        model_version=config["model_version"],
        date_format=config["date_format"]
    )
