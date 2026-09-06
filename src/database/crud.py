import sys
from datetime import datetime, timedelta

from sqlalchemy import (
    Float,
    Integer,
    cast,
    func,
)

from src.database.connection import SessionLocal
from src.database.models import ModelPredictionLog
from src.exception import CustomException
from src.logger import logging


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

        logging.info("Adding the data to the session's memory")

        db.add(log_record)

        logging.info("Commiting the data successfully to Supabase")

        db.commit()

    except Exception as e:
        db.rollback()
        db.close()
        logging.exception("[MONITORING FAILED] Couldn't commit metrics log to Supabase.")
        raise CustomException(e,sys)


    try:

        location_to_be_filled = metrics_data["input_features"][0]['Location']

        truth_label_value = metrics_data["input_features"][0]['RainToday']

        yesterday = db_timestamp.date() - timedelta(days=1)

        logging.info("Getting yesterday's first row with 'None' truth label")

        previous_row = (db.query(ModelPredictionLog)
                        .filter(
                            cast(cast(ModelPredictionLog.input_features[0]['Location'].astext, Float), Integer) == location_to_be_filled,
                            ModelPredictionLog.truth_label.is_(None)
                        )
                        .filter(
                            func.date(ModelPredictionLog.timestamp) == yesterday
                        )
                        .first()
                    )

        if previous_row:
            previous_row.truth_label = truth_label_value
            logging.info(f"Filled truth_label for {location_to_be_filled} ({yesterday}): {truth_label_value}")

        logging.info("Committing yesterday's truth labels to Supabase")

        db.commit()

    except Exception as e:
        db.rollback()
        logging.exception("An error has occurred while filling yesterday's truth value.")
        raise CustomException(e,sys)

    finally:
        db.close()
