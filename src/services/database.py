import os
import sys
from datetime import datetime
from src.logger import logging
from dotenv import load_dotenv
from src.exception import CustomException
from sqlalchemy import create_engine, Column, String, Float, JSON, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

logging.info("Getting the database_url")

DATABASE_URL = os.getenv("DATABASE_URL")

logging.info("Creating the engine for the database")

engine = create_engine(
    DATABASE_URL,
    pool_size=5,             
    max_overflow=10,         
    pool_timeout=30,         
    pool_recycle=1800     
)

def check_connection():

    """Checks if connection to Supabase was successful or not
    """

    try:
        logging.info("Getting the database connection checked")
        with engine.connect():
            logging.info("Connection successful!")

    except Exception as e:
        logging.info(f"Failed to connect to Supabase engine: {e}")
        raise CustomException(e,sys)

check_connection()


logging.info("Configuring the session")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class ModelPredictionLog(Base):
    __tablename__ = "model_prediction_logs"

    request_id = Column(String, primary_key=True, index=True)  
    timestamp = Column(DateTime(timezone=True), nullable=False) 
    client_type = Column(String, nullable=False)
    model_version = Column(String, nullable=False)              
    input_features = Column(JSON, nullable=False)               
    predicted_output = Column(String, nullable=False)          
    confidence_score = Column(Float, nullable=False)             
    latency = Column(Float, nullable=False)                    
    truth_label = Column(String, nullable=True)                

Base.metadata.create_all(bind=engine)


def save_to_database(metrics_data: dict):
    """
    Executes sequentially on a separate background thread pool.
    Isolated database session management prevents web-thread blockages.
    """

    logging.info("Creating a new session")

    db = SessionLocal()
    try:
        
        db_timestamp = datetime.fromisoformat(metrics_data["timestamp"])
        
        log_record = ModelPredictionLog(
            request_id=metrics_data["request_id"],
            timestamp=db_timestamp,
            client_type=metrics_data["client_type"],
            model_version=metrics_data["model_version"],
            input_features=metrics_data["input_features"],
            predicted_output=str(metrics_data["prediction"]), 
            confidence_score=metrics_data["confidence_score"],
            latency=metrics_data["latency"],
            truth_label=metrics_data["truth_label"]
        )
        
        db.add(log_record)
        db.commit()

    except Exception as e:
        db.rollback()
        logging.error(f"[MONITORING FAILED] Couldn't commit metrics log to Supabase: {e}")

    finally:
        db.close() 
