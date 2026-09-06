import redis.asyncio as redis

from src.config import get_env
from src.logger import logging

redis_pool: redis.ConnectionPool | None = None
redis_client: redis.Redis | None = None

config = get_env()

async def setup_redis():
    global redis_pool, redis_client
    logging.info("Setting up Redis connection pool")
    redis_pool = redis.ConnectionPool.from_url(
        config.get("REDIS_URL"),
        max_connections=50,
        decode_responses=True
    )
    redis_client = redis.Redis(connection_pool=redis_pool)
    logging.info("Redis client initialized successfully")


async def close_redis():
    logging.info("Closing Redis client and disconnecting pool")
    await redis_client.aclose()
    await redis_pool.disconnect()
    logging.info("Redis client and pool closed successfully")
