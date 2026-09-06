import httpx

from src.logger import logging

client: httpx.AsyncClient | None = None

async def setup_http_client():
    global client
    logging.info("Setting up httpx AsyncClient")
    client = httpx.AsyncClient()
    logging.info("httpx AsyncClient initialized successfully")


async def close_http_client():
    logging.info("Closing httpx AsyncClient")
    await client.aclose()
    logging.info("httpx AsyncClient closed successfully")
