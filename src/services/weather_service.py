import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone

from src.core import http_client
from src.core import redis_client as redis_client_module
from src.exception import CustomException
from src.logger import logging

IST = timezone(timedelta(hours=5, minutes=30))


def make_cache_key(latitude: float, longitude: float) -> str:
    now = datetime.now(IST)
    bucket_hour = (now.hour // 6) * 6
    cache_key = f"weather:{latitude},{longitude}:{now.strftime('%Y-%m-%d')}-{bucket_hour:02d}"
    logging.info(f"Generated cache key: {cache_key}")
    return cache_key


async def get_cached_value(cached_key: str) -> dict | None:
    try:
        logging.info(f"Checking Redis for key: {cached_key}")
        cached_value = await redis_client_module.redis_client.get(cached_key)

        if cached_value:
            logging.info(f"Cache hit for {cached_key}")
            return json.loads(cached_value)

        logging.info(f"Cache miss for {cached_key}")
        return None

    except Exception as e:
        logging.exception(f"Redis unavailable while checking cache for {cached_key}, falling back to live fetch: {e}")
        return None


async def cache_new_value(cached_key: str, caching_value: dict, ttl: int = 21600):
    try:
        await redis_client_module.redis_client.set(cached_key, json.dumps(caching_value), ex=ttl)
        logging.info(f"Cached value for {cached_key} with TTL={ttl}s")

    except Exception as e:
        logging.exception(f"Redis unavailable while caching {cached_key}, continuing without caching: {e}")


async def fetch_weather(
        latitude: float,
        longitude: float,
        hourly_features: list[str],
        daily_features: list[str],
        retries: int = 5
) -> dict:

    cached_key = make_cache_key(latitude=latitude, longitude=longitude)

    cached_data = await get_cached_value(cached_key=cached_key)

    if cached_data:
        return cached_data

    logging.info(f"Fetching fresh data from API for {cached_key}")

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": daily_features,
        "hourly": hourly_features,
        "timezone": "auto",
        "start_date": datetime.now().strftime("%Y-%m-%d"),
        "end_date": datetime.now().strftime("%Y-%m-%d")
    }

    for attempt in range(retries):

        try:
            logging.info(f"Attempt {attempt+1}/{retries}: sending request to Open-Meteo Weather API")

            response = await http_client.client.get(
                url,
                params=params,
                timeout=10
            )
            response.raise_for_status()

            logging.info("Successfully got the data, now converting it to json")

            data = response.json()

            await cache_new_value(cached_key=cached_key, caching_value=data)

            return data

        except Exception as e:

            if attempt == retries - 1:
                logging.exception(f"Due to {e} reason the {attempt}th and last attempt has been failed.")
                raise CustomException(e, sys)

            wait_time = 3 * (attempt + 1)
            logging.exception(f"Attempt {attempt+1}/{retries} failed: {e}. Retrying in {wait_time}s...")

            await asyncio.sleep(wait_time)
