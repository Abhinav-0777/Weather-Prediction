import asyncio
import sys
from datetime import datetime

import httpx

from src.exception import CustomException
from src.logger import logging


logging.info("Creating a client using httpx")
client = httpx.AsyncClient()

async def fetch_weather(
        latitude: float,
        longitude: float,
        hourly_features: list[str],
        daily_features: list[str],
        retries: int = 5
) -> dict:

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

            response = await client.get(
                url,
                params=params,
                timeout=10
            )
            response.raise_for_status()

            logging.info("Successfully got the data, now converting it to json")

            data = response.json()

            return data

        except Exception as e:

            if attempt == retries - 1:
                logging.exception(f"Due to {e} reason the {attempt}th and last attempt has been failed.")
                raise CustomException(e, sys)

            wait_time = 3 * (attempt + 1)
            logging.exception(f"Attempt {attempt+1}/{retries} failed: {e}. Retrying in {wait_time}s...")

            await asyncio.sleep(wait_time)
