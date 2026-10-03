import numpy as np
import pandas as pd
import pytest
from prometheus_client import REGISTRY

from src.services import prediction_service
from src.utils import load_config

config = load_config()

FAKE_WEATHER = {
    "daily": {
        "temperature_2m_min": [10.0],
        "temperature_2m_max": [20.0],
        "precipitation_sum": [0.0],
        "et0_fao_evapotranspiration": [3.0],
        "sunshine_duration": [36000],
        "wind_direction_10m_dominant": [90],
    },
    "hourly": {
        "wind_gusts_10m": [30.0] * 24,
        "wind_direction_10m": [0] * 24,
        "wind_speed_10m": [15.0] * 24,
        "relative_humidity_2m": [60.0] * 24,
        "pressure_msl": [1015.0] * 24,
        "cloud_cover": [50.0] * 24,
        "temperature_2m": [15.0] * 24,
    },
}


@pytest.fixture
def mock_fetch_weather(monkeypatch: pytest.MonkeyPatch):
    """Replace `fetch_weather` in prediction_service with a fake async function.

    Returns the fixed FAKE_WEATHER payload instead of calling the Open-Meteo API,
    so tests stay deterministic and run without network access.
    """

    async def dummy_weather(**kwargs):
        return FAKE_WEATHER

    monkeypatch.setattr(
        prediction_service,
        "fetch_weather",
        dummy_weather
    )


@pytest.mark.anyio
async def test_run_prediction(mock_fetch_weather: None):
    """Check that `run_prediction` returns a well-formed result and metrics dict.

    Verifies:
        - `result` holds a valid prediction, a features DataFrame, and a
          confidence score between 0 and 1.
        - `metrics` holds the timestamp, client type, model version, a positive
          latency, and `truth_label` set to None.
    """

    output = await prediction_service.run_prediction("Sydney", "common_user", config['model_version'])
    result = output["result"]
    metrics = output["metrics"]

    assert isinstance(result['prediction'], (int, np.integer))
    assert isinstance(result['features'], pd.DataFrame)
    assert isinstance(result['confidence'], (float, np.floating))
    assert 0 <= result['confidence'] <= 1

    assert isinstance(metrics['timestamp'], str)
    assert metrics['client_type'] == 'common_user'
    assert metrics['model_version'] == config['model_version']
    assert isinstance(metrics['latency'], (float, np.floating))
    assert metrics['latency'] > 0
    assert metrics['truth_label'] is None


@pytest.mark.anyio
async def test_run_prediction_records_metrics(mock_fetch_weather: None):
    """Check that `run_prediction` records the Prometheus request counter.

    Reads the `client_requests_total` counter before and after the call and
    asserts it increased by exactly 1. A before/after delta is used because
    counters accumulate across tests in the same process.
    """

    labels = {"client_type": "common_user"}
    before = REGISTRY.get_sample_value("client_requests_total", labels) or 0

    await prediction_service.run_prediction("Sydney", "common_user", config['model_version'])

    after = REGISTRY.get_sample_value("client_requests_total", labels)
    assert after == before + 1
