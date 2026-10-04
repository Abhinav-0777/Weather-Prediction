import json
import re
from datetime import timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.exception import CustomException
from src.services import weather_service
from src.utils import load_config

config = load_config()

IST = timezone(timedelta(hours=5, minutes=30))


@pytest.fixture
def mock_redis_client(monkeypatch: pytest.MonkeyPatch):

    dummy_redis_client = AsyncMock()
    dummy_redis_client.get.return_value = None   # default: cache miss

    monkeypatch.setattr(weather_service.redis_client_module, "redis_client", dummy_redis_client)

    return dummy_redis_client


def test_make_cache_key_format():
    key = weather_service.make_cache_key(-33.86, 151.2)
    assert re.fullmatch(r"weather:-33\.86,151\.2:\d{4}-\d{2}-\d{2}-(00|06|12|18)", key)


@pytest.mark.anyio
async def test_get_cached_value_hit(mock_redis_client: None):

    data = {"daily": {"temperature_2m_min": [10.0]}}
    mock_redis_client.get.return_value = json.dumps(data)

    cached_result = await weather_service.get_cached_value("some-key")

    assert cached_result == data

    mock_redis_client.get.assert_awaited_once_with("some-key")


@pytest.mark.anyio
async def test_get_cached_value_miss(mock_redis_client: None):

    mock_redis_client.get.return_value = None

    cached_result = await weather_service.get_cached_value("some-key")

    assert cached_result is None


@pytest.mark.anyio
async def test_get_cached_value_error(mock_redis_client: None):

    mock_redis_client.get.side_effect = Exception("Redis down")

    cached_result = await weather_service.get_cached_value("some-key")

    assert cached_result is None


@pytest.mark.anyio
async def test_cache_new_value_custom_ttl(mock_redis_client: None):

    await weather_service.cache_new_value("some-key", {"a": 1}, ttl=60)

    mock_redis_client.set.assert_awaited_once_with("some-key", json.dumps({"a": 1}), ex=60)


@pytest.mark.anyio
async def test_cache_new_value_error(mock_redis_client: None):

    mock_redis_client.set.side_effect = Exception("Redis down")

    await weather_service.cache_new_value("some-key", {"a": 1})


WEATHER_DATA = {"daily": {"temperature_2m_min": [10.0]}}


@pytest.fixture
def mock_http(monkeypatch: pytest.MonkeyPatch):
    """Fake httpx client whose async `get` returns a fake response."""

    response = MagicMock()
    response.json.return_value = WEATHER_DATA

    fake_client = AsyncMock()
    fake_client.get.return_value = response

    monkeypatch.setattr(weather_service.http_client, "client", fake_client, raising=False)
    return fake_client


@pytest.fixture
def mock_sleep(monkeypatch: pytest.MonkeyPatch):
    """Skip real retry waits without touching the global asyncio.sleep."""

    fake = AsyncMock()
    monkeypatch.setattr(weather_service, "asyncio", SimpleNamespace(sleep=fake))
    return fake


async def call_fetch_weather():

    return await weather_service.fetch_weather(
        latitude=-33.86,
        longitude=151.2,
        hourly_features=["temperature_2m"],
        daily_features=["temperature_2m_min"],
    )


@pytest.mark.anyio
async def test_fetch_weather_cache_hit(mock_redis_client: None, mock_http: None, mock_sleep: None):
    """Cached data is returned and the API is never called."""

    mock_redis_client.get.return_value = json.dumps(WEATHER_DATA)

    result = await call_fetch_weather()

    assert result == WEATHER_DATA
    mock_http.get.assert_not_awaited()
    mock_redis_client.set.assert_not_awaited()


@pytest.mark.anyio
async def test_fetch_weather_cache_miss(mock_redis_client: None, mock_http: None, mock_sleep: None):
    """On a miss, the API is called once and the result is cached."""

    result = await call_fetch_weather()

    assert result == WEATHER_DATA
    mock_http.get.assert_awaited_once()
    mock_redis_client.set.assert_awaited_once()
    assert mock_redis_client.set.await_args.args[1] == json.dumps(WEATHER_DATA)


@pytest.mark.anyio
async def test_fetch_weather_retry_then_success(mock_redis_client: None, mock_http: None, mock_sleep: None):
    """First API call fails, second succeeds: data is returned after one wait."""

    response = mock_http.get.return_value
    mock_http.get.side_effect = [Exception("temporary failure"), response]

    result = await call_fetch_weather()

    assert result == WEATHER_DATA
    assert mock_http.get.await_count == 2
    mock_sleep.assert_awaited_once_with(3)


@pytest.mark.anyio
async def test_fetch_weather_all_retries_fail(mock_redis_client: None, mock_http: None, mock_sleep: None):
    """If every attempt fails, CustomException is raised after 5 tries."""

    mock_http.get.side_effect = Exception("API down")

    with pytest.raises(CustomException):
        await call_fetch_weather()

    assert mock_http.get.await_count == 5
    assert [c.args[0] for c in mock_sleep.await_args_list] == [3, 6, 9, 12]
    mock_redis_client.set.assert_not_awaited()


@pytest.mark.anyio
async def test_fetch_weather_redis_down_still_works(mock_redis_client: None, mock_http: None, mock_sleep: None):
    """Redis failures on get and set are swallowed; live API data is returned."""

    mock_redis_client.get.side_effect = Exception("Redis down")
    mock_redis_client.set.side_effect = Exception("Redis down")

    result = await call_fetch_weather()

    assert result == WEATHER_DATA
    mock_http.get.assert_awaited_once()
