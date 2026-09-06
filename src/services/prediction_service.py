import asyncio
import time
import uuid
from datetime import UTC, datetime

import pandas as pd

from src.logger import logging
from src.monitoring.prometheus.metrics import (
    client_requests,
    confidence_score_metric,
    prediction_latency,
)
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.schema import data_validation
from src.services.weather_service import fetch_weather
from src.utils import load_config

config = load_config()

CITY_COORDS = config['CITY_COORDS']

DIRECTIONS = config['DIRECTIONS']

HOURLY_FEATURES = config['HOURLY_FEATURES']
DAILY_FEATURES = config['DAILY_FEATURES']


async def run_prediction(location: str, client_type: str, model_version: str) -> dict:

    """
    Orchestrates the full prediction workflow: fetch live weather,
    validate, run the model, record Prometheus metrics, and return
    the raw result dictionary .
    """

    timestamp = datetime.now(UTC).isoformat()
    request_id = str(uuid.uuid4())

    latitude, longitude = CITY_COORDS.get(location, CITY_COORDS['Melbourne'])

    logging.info(f"Fetching live weather data for request_id={request_id}")

    live_data = await fetch_weather(
        latitude=latitude,
        longitude=longitude,
        hourly_features=HOURLY_FEATURES,
        daily_features=DAILY_FEATURES
    )

    logging.info("Validating the data using pydantic")

    data = data_validation(
        Date=datetime.now().strftime("%Y-%m-%d"),
        Location=next(k for k, v in CITY_COORDS.items() if v == [latitude, longitude]),
        MinTemp=live_data["daily"]["temperature_2m_min"][0],
        MaxTemp=live_data["daily"]["temperature_2m_max"][0],
        Rainfall=live_data["daily"]["precipitation_sum"][0],
        Evaporation=live_data["daily"]["et0_fao_evapotranspiration"][0],
        Sunshine=live_data["daily"]["sunshine_duration"][0] / 3600,
        WindGustDir=DIRECTIONS[int((live_data["daily"]["wind_direction_10m_dominant"][0] + 11.25) / 22.5) % 16],
        WindGustSpeed=max(live_data["hourly"]["wind_gusts_10m"]),
        WindDir9am=DIRECTIONS[int((live_data["hourly"]["wind_direction_10m"][9] + 11.25) / 22.5) % 16],
        WindDir3pm=DIRECTIONS[int((live_data["hourly"]["wind_direction_10m"][15] + 11.25) / 22.5) % 16],
        WindSpeed9am=live_data["hourly"]["wind_speed_10m"][9],
        WindSpeed3pm=live_data["hourly"]["wind_speed_10m"][15],
        Humidity9am=live_data["hourly"]["relative_humidity_2m"][9],
        Humidity3pm=live_data["hourly"]["relative_humidity_2m"][15],
        Pressure9am=live_data["hourly"]["pressure_msl"][9],
        Pressure3pm=live_data["hourly"]["pressure_msl"][15],
        Cloud9am=live_data["hourly"]["cloud_cover"][9] / 10,
        Cloud3pm=live_data["hourly"]["cloud_cover"][15] / 10,
        Temp9am=live_data["hourly"]["temperature_2m"][9],
        Temp3pm=live_data["hourly"]["temperature_2m"][15],
        RainToday="Yes" if live_data["daily"]["precipitation_sum"][0] > 0 else "No"
    )

    df = pd.DataFrame([data.model_dump()])

    logging.info("Getting predictions from pipeline")

    pipeline = PredictionPipeline()
    start = time.perf_counter()
    result = await asyncio.to_thread(pipeline.get_prediction, df)
    end = time.perf_counter()

    logging.info("Successfully predicted the data")

    metrics = {
        "timestamp": timestamp,
        "request_id": request_id,
        "client_type": client_type,
        "model_version": model_version,
        "input_features": result["features"],
        "prediction": result["prediction"],
        "confidence_score": result["confidence"],
        "latency": (end - start),
        "truth_label": None
    }

    prediction_latency.labels(model_version=model_version, client_type=client_type).observe(metrics['latency'])
    confidence_score_metric.labels(model_version=model_version).observe(metrics['confidence_score'])
    client_requests.labels(client_type=client_type).inc()

    logging.info(f"Prometheus metrics recorded — latency={metrics['latency']:.4f}s, "
                 f"confidence_score={metrics['confidence_score']:.4f}, client_type={client_type}")

    return {
        "result": result,
        "metrics": metrics
    }
