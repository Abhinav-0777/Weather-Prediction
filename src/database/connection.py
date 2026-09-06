import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.config import get_env
from src.exception import CustomException
from src.logger import logging

config = get_env()

logging.info("Getting the database_url")

DATABASE_URL = config.get("DATABASE_URL")

logging.info("Creating the engine for the database")

engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def check_connection():

    """Checks if connection to Supabase was successful or not
    """

    try:
        logging.info("Getting the database connection checked")
        with engine.connect():
            logging.info("Connection successful!")

    except Exception as e:
        logging.exception(f"Failed to connect to Supabase engine: {e}")
        raise CustomException(e,sys)
